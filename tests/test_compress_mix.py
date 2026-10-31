"""
Test the mixed compression used by scripts/compress_search_mix.py, on a tiny random HF Qwen2 with made-up
latency tables. The deployed model with int4 needs a CUDA GPU (sm80+); the rest runs on CPU.

python -m pytest tests/test_compress_mix.py -v
"""

from dataclasses import replace

import pytest
import torch

from nanochat import compress_hw
from nanochat.compress_hw import FORMATS, MLP_KEEP, Layout, shape_key
from nanochat.compress_mix import SHARED, MixCompressor, entropy_bits


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=128, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, tie_word_embeddings=True)
    return Qwen2ForCausalLM(config).to(dtype=torch.float32).eval()


def fake_table(layout, batch, int4_speed):
    gemm = {shape_key(*s): {f: s[0] * s[1] / 1e4 * (1.0 if f == "bf16" else int4_speed) for f in FORMATS}
            for s in layout.linear_shapes(MLP_KEEP)}
    return {"batch": batch, "context": 16, "formats": list(FORMATS), "gemm_us": gemm, "attn_us": 2.0,
            "attn_profiled_us": 1.0, "misc_attn_us": 3.0, "misc_mlp_us": 2.0, "fixed_us": 5.0}


@pytest.fixture
def setup():
    model = tiny_qwen()
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32))
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(layout.n_layer)], "mlp": [(i, 0.0) for i in range(layout.n_layer)]}
    # int4 is faster when decoding and slower on prompts, as measured
    compressor = MixCompressor(model, layout, fake_table(layout, 4, 0.5), fake_table(layout, 64, 4.0), act, order, bands=2, max_skip=1,
                               new_tokens=100, gen_tokens=10, vram_gb=1.0)
    return model, compressor, tokens


def logits(model, tokens):
    with torch.no_grad():
        return model(input_ids=tokens, use_cache=False).logits.float()


def shared_genes(c, **share):
    g = dict(zip((n for n, _ in c.genes), c.uniform_genes(SHARED)))
    g.update(share)
    return tuple(g[n] for n, _ in c.genes)


def test_decode_ignores_sharing_genes_when_nothing_is_shared(setup):
    _, c, _ = setup
    names = [n for n, _ in c.genes]
    a = list(c.uniform_genes("int4_g64"))
    b = list(a)
    b[names.index("share_k")] = 3
    b[names.index("share_rho")] = 2
    assert c.decode(a) == c.decode(b)
    a = list(c.uniform_genes(SHARED))
    b = list(a)
    b[names.index("share_k")] = 3
    assert c.decode(a) != c.decode(b)


def test_shared_costs_time_and_memory_of_bf16_but_stores_smaller(setup):
    model, c, tokens = setup
    bf16, shared = c.config(c.uniform_genes("bf16")), c.config(c.decode(shared_genes(c, share_k=1, share_prune=2, share_rho=2)))
    assert c.cost(shared) == c.cost(bf16)
    before = logits(model, tokens)
    base = c.apply(bf16)
    assert base["k_merged"] == 0 and base["shared_pct"] == 0
    assert (logits(model, tokens) == before).all()
    c.restore()
    stats = c.apply(shared)
    changed = logits(model, tokens)
    c.restore()
    assert stats["shared_pct"] == pytest.approx(100.0)
    assert 0 < stats["k_merged"] <= 33
    assert stats["size_mb"] < base["size_mb"] / 2
    assert not (changed == before).all()
    assert (logits(model, tokens) == before).all()  # restore() brings the original back


def test_more_merging_means_fewer_levels_and_smaller_size(setup):
    _, c, _ = setup
    sizes, levels = [], []
    for rho in range(4):
        stats = c.apply(c.config(c.decode(shared_genes(c, share_k=3, share_rho=rho))))
        c.restore()
        sizes.append(stats["size_mb"])
        levels.append(stats["k_merged"])
    assert levels == sorted(levels, reverse=True) and levels[0] > levels[-1]
    assert sizes[-1] < sizes[0]


def test_turn_time_comes_from_both_tables(setup):
    _, c, _ = setup
    bf16, int4 = c.cost(c.config(c.uniform_genes("bf16"))), c.cost(c.config(c.uniform_genes("int4_g64")))
    assert int4["decode_us"] < bf16["decode_us"] and int4["prompt_us"] > bf16["prompt_us"]
    for cost in (bf16, int4):
        assert cost["latency_ms"] == pytest.approx((100 * cost["prompt_us"] + 10 * cost["decode_us"]) / 1e3)
    assert int4["memory_mb"] < bf16["memory_mb"] and int4["agents_fit"] > bf16["agents_fit"]


def test_prompt_time_leaves_out_the_head(setup):
    _, c, _ = setup
    cfg = c.config(c.uniform_genes("bf16"))
    head = c.prompt_table["gemm_us"][shape_key(*c.shapes[c.layout.lm_head])]["bf16"]
    c.table, decode_table = c.prompt_table, c.table
    whole = compress_hw.HWCompressor.cost(c, cfg)
    c.table = decode_table
    assert c.cost(cfg)["prompt_us"] == pytest.approx((1e3 * whole["latency_ms"] - head) / 64)


def test_entropy_bits():
    assert entropy_bits([4, 4]) == pytest.approx(8.0)
    assert entropy_bits([8, 0]) == pytest.approx(0.0)


def test_pruned_neurons_and_shared_format_in_the_size(setup):
    _, c, _ = setup
    genes = dict(zip((n for n, _ in c.genes), shared_genes(c, share_k=1)))
    full = c.apply(c.config(c.decode(tuple(genes.values()))))
    c.restore()
    genes.update(keep_b0=len(c.mlp_keep) - 1, keep_b1=len(c.mlp_keep) - 1)
    cut = c.apply(c.config(c.decode(tuple(genes.values()))))
    c.restore()
    assert cut["size_mb"] < full["size_mb"] and cut["shared_pct"] < full["shared_pct"]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                    reason="the int4 tinygemm kernel needs a CUDA GPU with sm80+")
def test_materialized_model_matches_the_written_weights():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=128, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32), device="cuda")
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(4)], "mlp": [(i, 0.0) for i in range(4)]}
    c = MixCompressor(model, layout, fake_table(layout, 4, 0.5), fake_table(layout, 64, 4.0), act, order, bands=2, max_skip=1)
    genes = dict(zip((n for n, _ in c.genes), c.uniform_genes(SHARED)))
    int4 = c.formats.index("int4_g64")
    genes.update(qkv_b0=int4, down_b1=int4, keep_b0=2, keep_b1=1, share_k=1, share_prune=1, skip_mlp=1)
    cfg = c.config(c.decode(tuple(genes.values())))
    assert cfg.lm_head == SHARED and replace(cfg, lm_head="bf16") != cfg
    c.apply(cfg)
    written = logits(model, tokens)
    c.restore()
    c.materialize(cfg)
    assert isinstance(layout.module(layout.blocks[0]["qkv"][0]), compress_hw.Int4Linear)
    assert (logits(model, tokens) - written).abs().max() < 2e-2 * written.abs().max()
