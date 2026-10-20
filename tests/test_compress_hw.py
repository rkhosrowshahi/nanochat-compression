"""
Test the hardware-aware compression used by scripts/compress_search_hw.py, on tiny random models
(an HF Qwen2 and a legacy nanochat GPT). The int4 kernel, the latency table and the deployed
models need a CUDA GPU (sm80+); the rest runs on CPU.

python -m pytest tests/test_compress_hw.py -v
"""

import pytest
import torch

from nanochat import compress_hw
from nanochat.compress_hw import (FORMATS, MLP_KEEP, HWCompressor, Layout, int4_dequantize, int4_quantize, n_kept,
                                  shape_key, weight_bytes)

cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                          reason="the int4 tinygemm kernel needs a CUDA GPU with sm80+")


def tiny_qwen(device="cpu", dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=128, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, tie_word_embeddings=True)
    return Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()


def tiny_nanochat(device="cpu"):
    from nanochat import gpt_legacy
    from nanochat.compress_models import NanochatLM
    torch.manual_seed(0)
    gpt = gpt_legacy.GPT(gpt_legacy.GPTConfig(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=128))
    gpt = gpt.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    gpt.init_rotary()
    return NanochatLM(gpt, autocast=device == "cuda").eval()


def fake_table(layout, batch=1, context=16):
    """A latency table with made-up times: int4 a third of bf16, proportional to the matrix size."""
    gemm = {shape_key(*s): {f: s[0] * s[1] / 1e4 * (1.0 if f == "bf16" else 0.3) for f in FORMATS}
            for s in layout.linear_shapes(MLP_KEEP)}
    return {"batch": batch, "context": context, "formats": list(FORMATS), "gemm_us": gemm, "attn_us": 2.0,
            "attn_profiled_us": 1.0, "misc_attn_us": 3.0, "misc_mlp_us": 2.0, "fixed_us": 5.0}


def make_compressor(model, table=None, tokens=None, **kw):
    layout = Layout(model)
    device = next(model.parameters()).device
    tokens = torch.randint(0, 512, (2, 32), device=device) if tokens is None else tokens
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(layout.n_layer)], "mlp": [(i, 0.0) for i in reversed(range(layout.n_layer))]}
    return HWCompressor(model, layout, table or fake_table(layout), act, order, bands=2, max_skip=1, **kw), tokens


def logits(model, tokens):
    with torch.no_grad():
        return model(input_ids=tokens, use_cache=False).logits.float()


def test_int4_levels_error_and_clipping():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w[0, 0] = 0.5  # an outlier: clipping its group pays off
    q, scale, zero, idx = int4_quantize(w, 64, ratios=(1.0,))
    assert q.min() >= 0 and q.max() <= 15 and scale.dtype == torch.bfloat16
    err = (int4_dequantize(q, scale, zero, 64) - w).abs().view(64, 4, 64).amax(-1)
    assert (err <= 0.5 * scale.float() * 1.02 + 1e-6).all()  # within half a step (bf16 rounding of scale/zero)
    act = torch.rand(256)
    q2, s2, z2, idx2 = int4_quantize(w, 64, act)
    weighted = lambda q, s, z: ((int4_dequantize(q, s, z, 64) - w).square() * act).sum()
    assert weighted(q2, s2, z2) <= weighted(q, scale, zero) + 1e-9
    assert idx2.max() > 0  # some group chose to clip
    q3, s3, z3, _ = int4_quantize(w, 64, act, ratio_idx=idx2)  # a cached choice gives the same codes
    assert torch.equal(q3, q2) and torch.equal(s3, s2) and torch.equal(z3, z2)


def test_n_kept_and_weight_bytes():
    assert n_kept(1.0, 4864) == 4864 and n_kept(0.75, 4864) == 3584 and n_kept(0.5, 4864) % 128 == 0
    assert n_kept(0.01, 512) == 128
    assert weight_bytes(10, 256, "bf16") == 5120
    assert weight_bytes(10, 256, "int4_g128") == 10 * 128 + 4 * 10 * 2


@pytest.mark.parametrize("make", [tiny_qwen, tiny_nanochat])
def test_layout(make):
    layout = Layout(make())
    assert layout.n_layer in (2, 4) and len(layout.blocks[0]["qkv"]) == 3
    if layout.pair:  # HF Qwen: SwiGLU, tied head
        assert len(layout.blocks[0]["up"]) == 2 and layout.tied and layout.embeds == []
    else:
        assert len(layout.blocks[0]["up"]) == 1 and not layout.tied and len(layout.embeds) == 1


@pytest.mark.parametrize("make", [tiny_qwen, tiny_nanochat])
def test_apply_restore_is_exact(make):
    model = make()
    comp, tokens = make_compressor(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref = logits(model, tokens)
    genes = list(comp.uniform_genes("int4_g32"))
    names = [n for n, _ in comp.genes]
    genes[names.index("keep_b0")], genes[names.index("skip_mlp")] = 2, 1
    comp.apply(comp.config(tuple(genes)))
    assert not torch.allclose(logits(model, tokens), ref)
    comp.restore()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
    comp.apply(comp.config(comp.uniform_genes("bf16")))  # the original model: nothing written
    assert comp._touched == []


@pytest.mark.parametrize("make", [tiny_qwen, tiny_nanochat])
def test_fake_pruning_and_removal_equal_the_deployed_model(make):
    """With bf16 formats, apply() (zeroed neurons and projections) and materialize() (sliced MLPs, Removed
    sublayers) are the same model."""
    model = make()
    comp, tokens = make_compressor(model)
    names = [n for n, _ in comp.genes]
    genes = list(comp.uniform_genes("bf16"))
    genes[names.index("keep_b0")], genes[names.index("keep_b1")] = 2, 4
    genes[names.index("skip_attn")], genes[names.index("skip_mlp")] = 1, 1
    cfg = comp.config(tuple(genes))
    comp.apply(cfg)
    fake = logits(model, tokens)
    comp.restore()
    comp.materialize(cfg)
    torch.testing.assert_close(logits(model, tokens), fake, rtol=1e-4, atol=1e-4)
    mlp = model.get_submodule(comp.layout.layers[comp.order["mlp"][0]]).mlp
    assert isinstance(mlp, compress_hw.Removed)
    up = comp.layout.module(comp.layout.blocks[0]["up"][0])  # layer 0: band 0, MLP kept
    assert up.out_features == n_kept(comp.mlp_keep[2], comp.layout.intermediate)


def test_cost_accounting():
    model = tiny_qwen()
    comp, _ = make_compressor(model)
    table = comp.table
    base = comp.cost(comp.config(comp.uniform_genes("bf16")))
    n_params = sum(p.numel() for p in model.parameters())  # tied: counted once
    kv = table["batch"] * table["context"] * comp.layout.n_layer * comp.layout.kv_bytes()
    assert base["memory_mb"] == pytest.approx((2 * n_params + kv) / 1e6)
    assert base["bits_per_weight"] == pytest.approx(16)
    int4 = comp.cost(comp.config(comp.uniform_genes("int4_g128")))
    assert int4["latency_ms"] < base["latency_ms"] and int4["memory_mb"] < base["memory_mb"]
    assert int4["bits_per_weight"] == pytest.approx(4 + 32 / 128)  # codes + bf16 scale and zero per 128
    names = [n for n, _ in comp.genes]
    genes = list(comp.uniform_genes("bf16"))
    genes[names.index("skip_attn")] = 1
    removed = comp.cost(comp.config(tuple(genes)))
    assert removed["kv_mb"] == pytest.approx(kv / 1e6 * 3 / 4)  # one of 4 layers has no KV cache
    assert removed["latency_ms"] < base["latency_ms"] and removed["n_removed_attn"] == 1


@cuda
def test_int4_kernel_matches_dequantized_weights():
    torch.manual_seed(0)
    w = torch.randn(256, 384, device="cuda") * 0.02
    q, scale, zero, _ = int4_quantize(w, 128)
    layer = compress_hw.Int4Linear(q, scale, zero, 128)
    x = torch.randn(5, 384, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ int4_dequantize(q, scale, zero, 128).bfloat16().float().t()
    y = layer(x).float()
    assert ((y - ref).abs().max() / ref.abs().max()).item() < 2e-2


@cuda
@pytest.mark.parametrize("make", [tiny_qwen, tiny_nanochat])
def test_materialized_int4_model_matches_fake_quantization(make):
    model = make("cuda") if make is tiny_nanochat else make("cuda", torch.bfloat16)
    comp, tokens = make_compressor(model)
    names = [n for n, _ in comp.genes]
    genes = list(comp.uniform_genes("int4_g32"))
    genes[names.index("qkv_b1")] = comp.formats.index("int4_g128")
    genes[names.index("keep_b1")], genes[names.index("skip_attn")] = 2, 1
    cfg = comp.config(tuple(genes))
    comp.apply(cfg)
    fake = logits(model, tokens)
    comp.restore()
    comp.materialize(cfg)
    assert any(isinstance(m, compress_hw.Int4Linear) for m in model.modules())
    real = logits(model, tokens)
    assert ((real - fake).abs().max() / fake.abs().max()).item() < 3e-2  # bf16 rounding inside the kernel


@cuda
def test_latency_table_reproduces_the_profiled_model():
    """Needs an otherwise idle GPU (see build_latency_table)."""
    model = tiny_qwen("cuda", torch.bfloat16)
    layout = Layout(model)
    table = compress_hw.build_latency_table(model, layout, FORMATS, MLP_KEEP, batch=1, context=64, log=lambda *a: None)
    assert table["attn_impl"] in compress_hw.ATTENTION and table["attn_profiled_us"] > 0 and table["formats"]
    assert table["attn_us"] == min(table["attn_impl_us"].values())
    for entry in table["gemm_us"].values():
        assert entry["bf16"] > 0
    comp, _ = make_compressor(model, table=table)
    predicted = comp.cost(comp.config(comp.uniform_genes("bf16")), as_profiled=True)["latency_ms"] * 1000
    # the latency model is calibrated on the original model: its prediction of the profiled forward is that profile
    assert predicted == pytest.approx(table["profile_us"]["full"], rel=0.05)


@cuda
def test_grouped_attention_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 6, 1, 32, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(2, 2, 50, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    ref = compress_hw.ATTENTION["sdpa"](q, k, v)
    torch.testing.assert_close(compress_hw.grouped_attention(q, k, v), ref, rtol=2e-2, atol=2e-2)
