"""
Test the KV-cache genes of scripts/compress_search_kv.py on a tiny random HF Qwen2 with made-up tables. Timing the reads needs
a CUDA GPU; the rest runs on CPU.

python -m pytest tests/test_compress_kv.py -v
"""

import pytest
import torch

from nanochat import compress_hw
from nanochat.compress_hw import MLP_KEEP, Layout
from nanochat.compress_kv import (KV_PRECISIONS, KV_WINDOWS, KVCompressor, build_kv_table, fake_quantize, kv_tensor_bytes, window_mask,
                                  window_tokens)
from tests.test_compress_mix import fake_table, logits, tiny_qwen


def fake_kv_table():
    keys = [f"{w:g}" for w in KV_WINDOWS]
    part = lambda scale: {"attention_us": {k: scale * float(k) for k in keys},
                          "read_us": {p: {k: scale * float(k) / 2 for k in keys} for p in KV_PRECISIONS[1:]}}
    return {"precisions": list(KV_PRECISIONS), "windows": list(KV_WINDOWS), "context": 64, "decode": part(2.0), "prompt": part(10.0)}


@pytest.fixture
def setup():
    model = tiny_qwen()
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32))
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(layout.n_layer)], "mlp": [(i, 0.0) for i in range(layout.n_layer)]}
    compressor = KVCompressor(model, layout, fake_table(layout, 4, 0.5), fake_table(layout, 64, 4.0), fake_kv_table(), act, order, context=64,
                              bands=2, max_skip=1, new_tokens=100, gen_tokens=10, vram_gb=1.0, mlp_keep=MLP_KEEP)
    return model, compressor, tokens


def test_tensor_bytes_of_bf16_add_up_to_the_layout(setup):
    _, c, _ = setup
    L = c.layout
    assert 2 * kv_tensor_bytes("bf16", L.n_kv_head, L.head_dim) == L.kv_bytes()
    assert kv_tensor_bytes("int8", 2, 64) == 2 * 64 + 2 * 2
    assert kv_tensor_bytes("int4", 2, 64) == 2 * 32 + 2 * 2


def test_fake_quantize_scales_each_token_and_head():
    x = torch.tensor([[[1.0, -0.5, 100.0, 50.0], [0.01, 0.02, -0.03, 0.0]]])  # one sequence, 2 tokens, 2 heads of 2
    assert fake_quantize(x, "bf16", 2) is x
    y = fake_quantize(x, "int8", 2)
    assert y.shape == x.shape
    assert (y - x).abs().view(1, 2, 2, 2).amax(-1).lt(x.abs().view(1, 2, 2, 2).amax(-1) / 127).all()
    coarse = fake_quantize(x, "int4", 2)
    assert (coarse - x).abs().sum() > (y - x).abs().sum()


def test_window_mask():
    mask = window_mask(None, 5, 2, "cpu")[0, 0]
    assert mask[4].tolist() == [False, False, False, True, True]
    assert mask[1].tolist() == [True, True, False, False, False]
    padded = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    padded[..., 4] = False
    assert window_mask(padded, 5, 2, "cpu")[0, 0][4].tolist() == [False, False, False, True, False]
    additive = torch.zeros(1, 1, 5, 5)
    assert (window_mask(additive, 5, 2, "cpu")[0, 0][4, :3] < -1e30).all()
    assert window_tokens(0.25, 2) == 1


def test_uncompressed_cache_changes_nothing(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    c.apply(c.config(c.uniform_genes("bf16")))
    assert not c._hooks
    assert (logits(model, tokens) == before).all()
    c.restore()


def test_precision_changes_the_model_and_restore_undoes_it(setup):
    model, c, tokens = setup
    before = logits(model, tokens)
    errors = []
    for precision in KV_PRECISIONS[1:]:
        c.apply(c.config(c.uniform_genes("bf16", precision=precision)))
        assert c._hooks
        errors.append((logits(model, tokens) - before).abs().mean().item())
        c.restore()
    assert 0 < errors[0] < errors[1]  # int4 further from the original than int8
    assert not c._hooks and (logits(model, tokens) == before).all()


def test_window_hides_older_tokens(setup):
    model, c, tokens = setup
    changed = tokens.clone()
    changed[:, 0] = (changed[:, 0] + 1) % 512  # only the first token differs
    assert (logits(model, changed)[:, -1] != logits(model, tokens)[:, -1]).any()
    c.apply(c.config(c.uniform_genes("bf16", window=0.25)))  # 8 of 32 tokens, through 4 layers: position 31 reads back to 3 at most
    near, far = logits(model, tokens), logits(model, changed)
    c.restore()
    # token 0 cannot reach the last position through 4 layers of 8-token windows (4 * 7 = 28 < 31)
    assert torch.allclose(near[:, -1], far[:, -1], atol=1e-3)
    assert (near[:, 1] != far[:, 1]).any()


def test_cost_of_the_cache(setup):
    _, c, _ = setup
    L = c.layout
    bf16 = c.cost(c.config(c.uniform_genes("bf16")))
    assert bf16["kv_mb"] == pytest.approx(4 * 64 * L.n_layer * L.kv_bytes() / 1e6)
    assert bf16["kv_pct"] == pytest.approx(100.0) and bf16["window_pct"] == pytest.approx(100.0)
    int8 = c.cost(c.config(c.uniform_genes("bf16", precision="int8")))
    half = c.cost(c.config(c.uniform_genes("bf16", window=0.5)))
    for cheaper in (int8, half):
        assert cheaper["kv_mb"] < bf16["kv_mb"] and cheaper["memory_mb"] < bf16["memory_mb"]
        assert cheaper["weights_mb"] == bf16["weights_mb"] and cheaper["agents_fit"] >= bf16["agents_fit"]
    assert int8["decode_us"] > bf16["decode_us"]  # each read expands the cache first
    assert half["decode_us"] < bf16["decode_us"] and half["prompt_us"] < bf16["prompt_us"]
    assert half["kv_mb"] == pytest.approx(bf16["kv_mb"] / 2) and half["window_pct"] == pytest.approx(50.0)
    assert int8["latency_ms"] == pytest.approx((100 * int8["prompt_us"] + 10 * int8["decode_us"]) / 1e3)


def test_removed_attention_keeps_no_cache(setup):
    _, c, _ = setup
    g = dict(zip((n for n, _ in c.genes), c.uniform_genes("bf16")))
    full = c.cost(c.config(tuple(g.values())))
    g["skip_attn"] = 1
    assert c.cost(c.config(tuple(g.values())))["kv_mb"] == pytest.approx(full["kv_mb"] * 3 / 4)


def test_genes_and_seeds(setup):
    _, c, _ = setup
    names = [n for n, _ in c.genes]
    assert not any(n.startswith("share_") for n in names)
    assert "shared" not in c.formats
    assert sum(n.startswith(("key_", "val_", "window_")) for n in names) == 3 * c.bands
    seeds = c.extra_seeds()
    assert len(set(seeds)) == len(seeds)
    for genes in seeds:
        assert c.decode(genes) == genes
        c.describe(c.config(genes))
    assert "key:88 val:88 win:1/1" in c.describe(c.config(c.uniform_genes("bf16", precision="int8")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="timed on a CUDA GPU")
def test_table_is_measured():
    layout = Layout(tiny_qwen().to("cuda"))
    table = build_kv_table(layout, KV_PRECISIONS, (1.0, 0.5), 2, 8, 256, "grouped", torch.device("cuda"), log=lambda m: None)
    for part in ("decode", "prompt"):
        assert table[part]["attention_us"]["1"] > 0 and table[part]["attention_us"]["0.5"] > 0
        assert set(table[part]["read_us"]) == {"int8", "int4"}
        assert all(t > 0 for times in table[part]["read_us"].values() for t in times.values())
