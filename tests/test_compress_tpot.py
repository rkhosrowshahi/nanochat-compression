"""
Test the decode-speed measurement (nanochat/compress_tpot.py) on a tiny random HF Qwen2. Needs a CUDA GPU
(sm80+ for the int4 model).

python -m pytest tests/test_compress_tpot.py -v
"""

import pytest
import torch

from nanochat import compress_tpot
from nanochat.compress_hw import ROLES, Config
from tests.test_compress_hw import cuda, make_compressor, tiny_qwen


@cuda
@pytest.mark.parametrize("attention", ["grouped", "model"])
@pytest.mark.parametrize("batch", [1, 3])
def test_decode_step_matches_a_forward_without_cache(attention, batch):
    """A decode step into the static cache gives the logits of the whole sequence run at once, with the grouped
    attention (tiny_qwen has 4 query heads on 2 KV heads) as with the model's own."""
    from transformers import StaticCache
    model = tiny_qwen("cuda", torch.bfloat16)
    seq = torch.randint(0, 512, (batch, 17), device="cuda")
    impl = compress_tpot.GROUPED_IMPL if attention == "grouped" else None
    with torch.no_grad(), compress_tpot.attention_implementation(model, impl):
        cache = StaticCache(config=model.config, max_cache_len=40)
        model(input_ids=seq[:, :16], past_key_values=cache, cache_position=torch.arange(16, device="cuda"), use_cache=True)
        step = model(input_ids=seq[:, 16:], past_key_values=cache, cache_position=torch.tensor([16], device="cuda"),
                     use_cache=True).logits[:, -1].float()
    assert model.config._attn_implementation != compress_tpot.GROUPED_IMPL  # restored
    with torch.no_grad():
        full = model(input_ids=seq, use_cache=False).logits[:, -1].float()
    assert torch.allclose(step, full, atol=0.02 * full.abs().max().item())


@cuda
@pytest.mark.parametrize("fmt", ["bf16", "int4_g32"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("attention", ["grouped", "model"])
def test_graph_decoding_matches_eager(fmt, batch, attention):
    model = tiny_qwen("cuda", torch.bfloat16)
    if fmt != "bf16":
        comp, _ = make_compressor(model)
        comp.materialize(Config(tuple((fmt, fmt) for _ in ROLES), "bf16", "bf16", (1.0, 1.0), 0, 0))
    r = compress_tpot.decode_timing(model, batch=batch, prompt_tokens=16, gen_tokens=24, repeats=2, check_tokens=20,
                                    attention=attention)
    assert r["graph_matches_eager"] == "21/21" and r["attention"] == attention
    assert r["tpot_ms"] > 0 and r["ttft_ms"] > 0 and len(r["tpot_ms_runs"]) == 2
    assert r["tok_s_total"] == pytest.approx(batch * r["tok_s"])
    assert r["e2e_ms"] == pytest.approx(r["ttft_ms"] + 23 * r["tpot_ms"])


@cuda
def test_measuring_twice_gives_the_same_tokens():
    """Each run rewinds the cache: a second measurement starts from the same state."""
    model = tiny_qwen("cuda", torch.bfloat16)
    a = compress_tpot.decode_timing(model, prompt_tokens=8, gen_tokens=12, repeats=1, check_tokens=11)
    b = compress_tpot.decode_timing(model, prompt_tokens=8, gen_tokens=12, repeats=1, check_tokens=11)
    assert a["graph_matches_eager"] == b["graph_matches_eager"] == "12/12"


@cuda
def test_decoding_runs_the_grouped_attention():
    from transformers import AttentionInterface
    calls = {"decode": 0, "prompt": 0}

    def spy(module, query, *args, **kwargs):
        calls["decode" if query.shape[2] == 1 else "prompt"] += 1
        return compress_tpot.grouped_decode_attention(module, query, *args, **kwargs)

    AttentionInterface.register(compress_tpot.GROUPED_IMPL, spy)
    try:
        model = tiny_qwen("cuda", torch.bfloat16)  # 4 layers
        compress_tpot.decode_timing(model, prompt_tokens=8, gen_tokens=6, repeats=1, check_tokens=2)
    finally:
        AttentionInterface.register(compress_tpot.GROUPED_IMPL, compress_tpot.grouped_decode_attention)
    assert calls["decode"] > 0 and calls["prompt"] > 0 and calls["decode"] % 4 == 0


def tpot_compressor(model, bands=4):
    from nanochat.compress_hw import Layout, activation_stats
    from tests.test_compress_hw import fake_table
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32), device=next(model.parameters()).device)
    order = {k: [(i, 0.0) for i in range(layout.n_layer)] for k in ("attn", "mlp")}
    return compress_tpot.TPOTCompressor(model, layout, fake_table(layout), activation_stats(model, layout, tokens), order,
                                        bands=bands, mlp_keep=(1.0,), max_skip=0)


def test_per_layer_genes_without_fixed_ones():
    comp = tpot_compressor(tiny_qwen(), bands=4)  # 4 layers: one band each
    names = [n for n, _ in comp.genes]
    assert names == [f"{r}_b{b}" for r in ROLES for b in range(4)] + ["lm_head"]  # tied: no embed gene
    genes = list(comp.uniform_genes("int4_g64"))
    genes[names.index("qkv_b2")] = 0
    cfg = comp.config(genes)
    assert cfg.fmt[0] == ("int4_g64", "int4_g64", "bf16", "int4_g64") and cfg.lm_head == "int4_g64"
    assert cfg.keep == (1.0,) * 4 and cfg.skip_attn == cfg.skip_mlp == 0


@cuda
@pytest.mark.parametrize("head", ["bf16", "int4_g32"])
def test_predicted_memory_is_the_deployed_models(head):
    """With a tied head in int4 the bf16 embedding stays: the cost counts both, as the deployed model holds both."""
    model = tiny_qwen("cuda", torch.bfloat16)
    comp = tpot_compressor(model)
    cfg = Config(tuple(("int4_g32", "bf16", "int4_g64", "int4_g128") for _ in ROLES), head, "bf16", (1.0,) * 4, 0, 0)
    predicted = comp.cost(cfg)["memory_mb"] * 1e6
    comp.materialize(cfg)
    buffers = sum(b.numel() * b.element_size() for n, b in model.named_buffers() if "inv_freq" in n)  # rotary: not weights
    assert compress_tpot.deployed_bytes(model)["weights_bytes"] - buffers == predicted


def test_deployed_bytes_counts_a_tied_matrix_once():
    model = tiny_qwen()
    size = compress_tpot.deployed_bytes(model)
    params = sum(p.numel() * p.element_size() for p in model.parameters())  # parameters() yields a tied one once
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    assert size["weights_bytes"] == params + buffers
    assert size["embedding_bytes"] == 512 * 128 * 4
