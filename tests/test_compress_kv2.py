"""
Test the cache genes of scripts/compress_search_kv2.py on a tiny random HF Qwen2 with made-up tables. The Triton kernel and
the time table need a CUDA GPU with Triton; the rest runs on CPU.

python -m pytest tests/test_compress_kv2.py -v
"""

import importlib.util

import pytest
import torch

from nanochat import compress_hw
from nanochat.compress_hw import MLP_KEEP, Layout
from nanochat.compress_kv2 import (KEY_KINDS, KV2_WINDOWS, SINKS, VALUE_KINDS, KV2Compressor, cache_mask, cached_keys, cached_values,
                                   dequantize_keys, dequantize_values, key_bytes, kept_tokens, quantize_keys, quantize_values,
                                   recall_score, recall_set, value_bytes)
from tests.test_compress_mix import fake_table, logits, tiny_qwen

GROUP = 8  # tokens per key scale in these tests: windows of 32 tokens hold 4 groups
GPU = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def fake_kv2_table():
    w = [f"{x:g}" for x in KV2_WINDOWS]
    speed = {"bf16": 1.0, "int8c": 0.6, "int4c": 0.4, "int8": 0.6, "int4": 0.4}
    decode = {f"{k}/{v}": {x: 2.0 * float(x) * (speed[k] + speed[v]) / 2 for x in w} for k in KEY_KINDS for v in VALUE_KINDS}
    reads = {kind: {x: 3.0 * float(x) for x in w} for kind in KEY_KINDS[1:] + VALUE_KINDS[1:]}
    return {"keys": list(KEY_KINDS), "values": list(VALUE_KINDS), "windows": list(KV2_WINDOWS), "context": 64, "group": GROUP,
            "decode": {"batch": 4, "attention_us": decode}, "prompt": {"batch": 64, "attention_us": {x: 10.0 * float(x) for x in w},
                                                                     "read_us": reads}}


@pytest.fixture
def setup():
    model = tiny_qwen()
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32))
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(layout.n_layer)], "mlp": [(i, 0.0) for i in range(layout.n_layer)]}
    compressor = KV2Compressor(model, layout, fake_table(layout, 4, 0.5), fake_table(layout, 64, 4.0), fake_kv2_table(), act, order,
                               context=64, bands=2, max_skip=1, new_tokens=100, gen_tokens=10, vram_gb=1.0, mlp_keep=MLP_KEEP)
    return model, compressor, tokens


def genes(c, fmt="bf16", **kw):
    return c.uniform_genes(fmt, **kw)


# --- formats

def test_bytes():
    assert key_bytes("bf16", 2, 64) == value_bytes("bf16", 2, 64) == 2 * 64 * 2
    assert key_bytes("int8c", 2, 64) == 2 * (64 + 2 * 64 / 64)
    assert key_bytes("int4c", 2, 64) == 2 * (32 + 2)
    assert value_bytes("int4", 2, 64) == 2 * (32 + 2)
    assert kept_tokens(1.0, 1000) == 1000 and kept_tokens(0.25, 1000) == 250 + SINKS


def test_keys_get_one_scale_per_channel_and_group():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 20, 8)
    k[..., 3] *= 100  # an outlier channel: per-channel scales keep the others fine
    for kind, levels in (("int8c", 127), ("int4c", 7)):
        codes, scale = quantize_keys(k, kind, GROUP)
        assert scale.shape == (1, 2, 3, 8)  # ceil(20 / 8) groups
        assert codes.shape[-1] == (8 if kind == "int8c" else 4)
        back = dequantize_keys(codes, scale, kind, GROUP)
        assert back.shape == k.shape
        bound = scale.float().repeat_interleave(GROUP, dim=2)[:, :, :20] * 0.51
        assert ((back - k).abs() <= bound + 1e-6).all()
        assert (back - k)[..., [0, 1, 2, 4]].abs().max() < 4 / levels  # the outlier does not spoil the other channels
    assert torch.equal(cached_keys(k, "bf16"), k)


def test_values_get_one_scale_per_token_and_int4_packs_halves():
    v = torch.zeros(1, 1, 2, 4)
    v[0, 0, 0] = torch.tensor([7.0, -7.0, 1.0, 0.0])  # scale 1: codes 7, -7, 1, 0
    codes, scale = quantize_values(v, "int4")
    assert scale[0, 0, 0].item() == 1.0
    assert codes.dtype == torch.uint8 and codes.shape == (1, 1, 2, 2)
    assert codes[0, 0, 0].tolist() == [(7 + 8) | ((1 + 8) << 4), (-7 + 8) | ((0 + 8) << 4)]  # channel j with j + D/2
    assert torch.equal(dequantize_values(codes, scale, "int4"), v)
    assert (cached_values(v, "int8") - v).abs().max() < 7 / 127  # within a step (the scale is rounded to bf16)


def test_cache_mask():
    plain = cache_mask(10, 3, False, GROUP, "cpu")
    assert plain[9].tolist() == [True] + [False] * 6 + [True] * 3  # the first token and the last 3
    assert plain[1].tolist() == [True, True] + [False] * 8
    split = cache_mask(20, 20, True, GROUP, "cpu")
    assert split.shape == (20, 40)
    assert torch.equal(split[:, :20] | split[:, 20:], cache_mask(20, 20, False, GROUP, "cpu"))
    assert not (split[:, :20] & split[:, 20:]).any()
    assert split[17, :20].tolist() == [True] * 16 + [False] * 4  # groups 0 and 1 quantized
    assert split[17, 20:].tolist() == [False] * 16 + [True] * 2 + [False] * 2  # its own group in bf16, up to itself
    assert not split[:GROUP, :20].any()


# --- the cache in the model

def test_uncompressed_cache_changes_nothing(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    c.apply(c.config(genes(c)))
    assert model.config._attn_implementation != "nanochat_kv2"
    assert torch.equal(logits(model, tokens), before)
    c.restore()


def test_bf16_through_the_cache_attention_is_the_same(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    c.apply(c.config(genes(c)))
    c.model.config._attn_implementation, c._attn_impl = "nanochat_kv2", "sdpa"
    for i in range(c.layout.n_layer):
        getattr(c.layout.module(c.layout.layers[i]), c.layout.attn_attr).kv2 = ("bf16", "bf16", 1.0, GROUP)
    assert torch.allclose(logits(model, tokens), before, atol=1e-4)
    c.restore()
    assert model.config._attn_implementation == "sdpa"


def test_precisions_change_the_model_and_restore_undoes_it(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    errors = {}
    for kw in ({"key": "int8c"}, {"key": "int4c"}, {"value": "int8"}, {"value": "int4"}):
        c.apply(c.config(genes(c, **kw)))
        errors[tuple(kw.values())[0]] = (logits(model, tokens) - before).abs().mean().item()
        c.restore()
    assert 0 < errors["int8c"] < errors["int4c"] and 0 < errors["int8"] < errors["int4"]
    assert torch.equal(logits(model, tokens), before) and model.config._attn_implementation == "sdpa"
    assert not hasattr(getattr(c.layout.module(c.layout.layers[0]), c.layout.attn_attr), "kv2")


def test_quantized_keys_wait_for_their_group(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    c.apply(c.config(genes(c, key="int4c")))
    after = logits(model, tokens)
    c.restore()
    assert torch.allclose(after[:, :GROUP], before[:, :GROUP], atol=1e-4)  # the first group reads its own keys in bf16
    assert not torch.allclose(after[:, GROUP:], before[:, GROUP:], atol=1e-3)


def test_window_keeps_the_first_token_only(setup):
    model, c, tokens = setup
    c.apply(c.config(genes(c, window=0.25)))  # 8 of 32 tokens and the first: through 4 layers, position 31 reads back to 3
    base = logits(model, tokens)
    moved = {}
    for pos in (0, 1):
        changed = tokens.clone()
        changed[:, pos] = (changed[:, pos] + 1) % 512
        moved[pos] = (logits(model, changed)[:, -1] - base[:, -1]).abs().max().item()
    c.restore()
    assert moved[0] > 1e-3  # the first token stays visible to every layer
    assert moved[1] < 1e-4  # token 1 has scrolled out (1 + 4 * 7 < 31)


# --- recall

class Echo(torch.nn.Module):
    """A 'model' that predicts each next token right (shift 1) or repeats the current one (shift 0)."""

    def __init__(self, shift=1, vocab=512):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.shift, self.vocab = shift, vocab

    def forward(self, input_ids, use_cache=False, logits_to_keep=0):
        nxt = torch.roll(input_ids, -self.shift, dims=1)
        out = torch.nn.functional.one_hot(nxt, self.vocab).float() * 50
        return type("Out", (), {"logits": out[:, -logits_to_keep:]})()


def test_recall_set_and_score():
    tokenizer = lambda s, add_special_tokens=False: {"input_ids": [ord(ch) % 512 for ch in s]}
    tokens, answers = recall_set(tokenizer, list(range(400)) * 5, 6, 300, seed=1)
    assert tokens.shape == (6, 300) and (answers == 6).all()  # " 12345" is 6 characters
    ce, acc = recall_score(Echo(), tokens, answers, batch_size=4)
    assert acc == 1.0 and ce < 1e-6
    ce, acc = recall_score(Echo(shift=0), tokens, answers)
    assert acc == 0.0 and ce > 10


# --- cost and genes

def test_cost_of_the_cache(setup):
    _, c, _ = setup
    L = c.layout
    bf16 = c.cost(c.config(genes(c)))
    assert bf16["kv_mb"] == pytest.approx(4 * 64 * L.n_layer * L.kv_bytes() / 1e6)
    assert bf16["kv_pct"] == pytest.approx(100.0)
    small = c.cost(c.config(genes(c, key="int4c", value="int4")))
    half = c.cost(c.config(genes(c, window=0.5)))
    for cheaper in (small, half):
        assert cheaper["kv_mb"] < bf16["kv_mb"] and cheaper["agents_fit"] >= bf16["agents_fit"]
        assert cheaper["decode_us"] < bf16["decode_us"]  # read in place: a smaller cache is a faster read
        assert cheaper["weights_mb"] == bf16["weights_mb"]
    per_token = L.n_layer * (key_bytes("int4c", L.n_kv_head, L.head_dim, GROUP) + value_bytes("int4", L.n_kv_head, L.head_dim))
    assert small["kv_mb"] == pytest.approx(4 * 64 * per_token / 1e6)
    assert small["prompt_us"] > bf16["prompt_us"]  # a prompt chunk expands the cache first
    assert half["kv_mb"] == pytest.approx(bf16["kv_mb"] * (32 + SINKS) / 64)
    assert small["latency_ms"] == pytest.approx((100 * small["prompt_us"] + 10 * small["decode_us"]) / 1e3)


def test_genes_and_seeds(setup):
    _, c, _ = setup
    names = [n for n, _ in c.genes]
    assert sum(n.startswith(("key_", "val_", "window_")) for n in names) == 3 * c.bands
    seeds = c.extra_seeds()
    assert len(set(seeds)) == len(seeds) == 13
    for g in seeds:
        assert c.decode(g) == g
        c.describe(c.config(g))
    assert "key:44 val:44 win:1/1" in c.describe(c.config(genes(c, key="int4c", value="int4")))


# --- the kernel (CUDA and Triton)

@pytest.mark.skipif(not GPU, reason="the kernel runs on a CUDA GPU with Triton")
@pytest.mark.parametrize("key", KEY_KINDS)
@pytest.mark.parametrize("value", VALUE_KINDS)
def test_kernel_matches_the_reference(key, value):
    from nanochat.compress_kv2 import check_kernel
    for D, heads in ((64, (14, 2)), (128, (40, 8))):  # Qwen2.5-0.5B and -32B heads
        layout = type("L", (), {"n_head": heads[0], "n_kv_head": heads[1], "head_dim": D})()
        for tokens in (64, 1000, 4100):
            assert check_kernel(layout, key, value, torch.device("cuda"), tokens=tokens) < 0.02


@pytest.mark.skipif(not GPU, reason="timed on a CUDA GPU with Triton")
def test_table_is_measured():
    from nanochat.compress_kv2 import build_kv2_table
    layout = Layout(tiny_qwen().to("cuda"))
    table = build_kv2_table(layout, KEY_KINDS, VALUE_KINDS, (1.0, 0.5), 2, 8, 512, torch.device("cuda"), log=lambda m: None)
    assert set(table["decode"]["attention_us"]) == {f"{k}/{v}" for k in KEY_KINDS for v in VALUE_KINDS}
    assert all(t > 0 for times in table["decode"]["attention_us"].values() for t in times.values())
    assert set(table["prompt"]["read_us"]) == {"int8c", "int4c", "int8", "int4"}
