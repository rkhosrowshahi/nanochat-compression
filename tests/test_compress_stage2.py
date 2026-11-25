"""
Test stage 2 of the two-stage search (nanochat/compress_stage2.py: weight sharing + pruning on a hardware-aware front) on a tiny
random HF Qwen2 (tied embedding, 4 layers) with a made-up latency table. The fake-quant parts run on CPU; the check against the
deployed stage-1 model needs a CUDA GPU with sm80+.

python -m pytest tests/test_compress_stage2.py -v
"""

import random

import pytest
import torch

from nanochat import compress_hw, compress_stage2, compress_tpot
from nanochat.compress_hw import FORMATS, ROLES, Config, HWCompressor, Layout
from nanochat.compress_mix import SHARED, entropy_bits
from tests.test_compress_hw import cuda, fake_table, logits, tiny_qwen

RUN_ARGS = {"formats": ",".join(FORMATS), "bands": 2, "mlp_keep": "1.0", "max_skip": 0, "no_embeddings": False,
            "clip_ratios": ",".join(map(str, compress_hw.CLIP_RATIOS))}


def stage1_row(genes, kld, latency, memory):
    return {**genes, "kld": kld, "latency_ms": latency, "tpot_ms": latency, "memory_mb": memory, "bits_per_weight": 0.0,
            "int4_pct": 0.0}


def stage1_genes(fmt_by_role, head):
    """Stage-1 gene dict of 2 bands: fmt_by_role[role] = (band 0 format, band 1 format)."""
    g = {f"{role}_b{b}": FORMATS.index(fmt_by_role[role][b]) for role in ROLES for b in range(2)}
    g["lm_head"] = FORMATS.index(head)
    return g


BF16 = stage1_genes({r: ("bf16", "bf16") for r in ROLES}, "bf16")
MIXED = stage1_genes({"qkv": ("int4_g32", "bf16"), "o": ("bf16", "bf16"), "up": ("int4_g64", "int4_g64"),
                      "down": ("bf16", "int4_g128")}, "bf16")
INT4_HEAD = stage1_genes({"qkv": ("bf16", "int4_g32"), "o": ("int4_g32", "int4_g32"), "up": ("bf16", "int4_g64"),
                          "down": ("int4_g128", "int4_g128")}, "int4_g64")
ALL_INT4 = stage1_genes({r: ("int4_g32", "int4_g32") for r in ROLES}, "int4_g32")


def untied_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=128, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, tie_word_embeddings=False)
    return Qwen2ForCausalLM(config).to(dtype=torch.float32).eval()


def make(model=None, bases=None, **kw):
    model = tiny_qwen() if model is None else model
    layout = Layout(model)
    device = next(model.parameters()).device
    tokens = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(1)).to(device)
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {k: [(i, 0.0) for i in range(layout.n_layer)] for k in ("attn", "mlp")}
    bases = bases or [stage1_row(BF16, 0.0, 10.0, 4.0), stage1_row(MIXED, 0.01, 8.0, 3.0),
                      stage1_row(INT4_HEAD, 0.02, 6.0, 2.5), stage1_row(ALL_INT4, 0.05, 5.0, 1.5)]
    table = fake_table(layout)
    c = compress_stage2.Stage2Compressor(model, layout, table, act, order, RUN_ARGS, bases, **kw)
    return model, c, tokens, (table, act, order)


def genes_of(c, base, **set_):
    g = dict(zip(c.names, c.no_sharing_genes(base)))
    g.update(set_)
    return c.decode([g[n] for n in c.names])


# --- stage 1: genes and front


def test_stage1_config_matches_the_stage1_compressors():
    model = tiny_qwen()
    _, c, _, (table, act, order) = make(model)
    hw = HWCompressor(model, c.layout, table, act, order, bands=2, mlp_keep=(1.0, 0.5), max_skip=1)
    tpot = compress_tpot.TPOTCompressor(model, c.layout, table, act, order, bands=2, mlp_keep=(1.0,), max_skip=0)
    rng = random.Random(0)
    for _ in range(20):
        for comp, keep in ((hw, (1.0, 0.5)), (tpot, (1.0,))):
            genes = [rng.randrange(n) for _, n in comp.genes]
            d = dict(zip((n for n, _ in comp.genes), genes))
            assert compress_stage2.stage1_config(d, FORMATS, keep, 2, comp.compress_embeddings) == comp.config(genes)


def test_per_layer_config_is_the_same_model():
    """A 2-band stage-1 configuration expanded to one band per layer writes the same weights."""
    model = tiny_qwen()
    _, c, tokens, (table, act, order) = make(model)
    hw = HWCompressor(model, c.layout, table, act, order, bands=2, mlp_keep=(1.0,), max_skip=0)
    per = HWCompressor(model, c.layout, table, act, order, bands=4, mlp_keep=(1.0,), max_skip=0)
    cfg = compress_stage2.stage1_config(MIXED, FORMATS, (1.0,), 2, False)
    expanded = compress_stage2.per_layer(cfg, 4, 2)
    assert expanded.fmt[0] == ("int4_g32", "int4_g32", "bf16", "bf16") and expanded.keep == (1.0,) * 4
    hw.apply(cfg)
    a = logits(model, tokens)
    hw.restore()
    per.apply(expanded)
    b = logits(model, tokens)
    per.restore()
    assert torch.equal(a, b)


def test_pareto_front_and_base_selection():
    rows = [{"key": "u", "kld": 0.0, "latency_ms": 10.0, "memory_mb": 9.0},
            {"key": "a", "kld": 0.1, "latency_ms": 5.0, "memory_mb": 4.0},
            {"key": "b", "kld": 0.2, "latency_ms": 6.0, "memory_mb": 5.0},  # dominated by a
            {"key": "c", "kld": 0.3, "latency_ms": 4.0, "memory_mb": 4.5},
            {"key": "d", "kld": 0.05, "latency_ms": 8.0, "memory_mb": 3.0},
            {"key": "d2", "kld": 0.05, "latency_ms": 8.0, "memory_mb": 3.0},  # duplicate of d
            {"key": "x", "kld": float("nan"), "latency_ms": 1.0, "memory_mb": 1.0}]
    front = compress_stage2.select_bases(rows, "kld", ["latency_ms", "memory_mb"], is_uncompressed=lambda r: r["key"] == "u")
    assert [r["key"] for r in front] == ["c", "a", "d", "u"]  # by TPOT; b dominated, d2 a duplicate, x broken
    limited = compress_stage2.select_bases(rows, "kld", ["latency_ms", "memory_mb"], max_quality=0.2,
                                           is_uncompressed=lambda r: r["key"] == "u")
    assert [r["key"] for r in limited] == ["a", "d", "u"]
    two = compress_stage2.select_bases(rows, "kld", ["latency_ms", "memory_mb"], max_bases=2, is_uncompressed=lambda r: r["key"] == "u")
    assert [r["key"] for r in two] == ["c", "u"]  # the ends of the TPOT range; the uncompressed model always stays
    with pytest.raises(AssertionError):
        compress_stage2.select_bases(rows[1:], "kld", ["latency_ms"], is_uncompressed=lambda r: r["key"] == "u")


# --- genes


def test_genes():
    _, c, _, _ = make()
    assert c.names == (["base"] + [f"share_{r}_b{g}" for r in ROLES for g in range(4)] + ["share_head"]
                       + list(compress_stage2.CANDIDATE_GENES))  # tied: the embedding is the head
    assert dict(c.genes)["base"] == 4 and c.uncompressed_base == 0
    _, c2, _, _ = make(untied_qwen(), bases=[stage1_row({**BF16, "embed": 0}, 0.0, 10.0, 4.0),
                                             stage1_row({**ALL_INT4, "embed": 1}, 0.05, 5.0, 1.5)])
    assert c2.names[-5] == "share_embed"
    assert dict(zip(c2.names, c2.all_shared_genes(0)))["share_embed"] == 1
    assert dict(zip(c2.names, c2.all_shared_genes(1)))["share_embed"] == 0  # its embedding is in int4


def test_untied_embedding_is_shared_and_stored_smaller():
    model = untied_qwen()
    _, c, tokens, _ = make(model, bases=[stage1_row({**BF16, "embed": 0}, 0.0, 10.0, 4.0)])
    emb = model.get_input_embeddings().weight
    orig = emb.detach().clone()
    plain = c.apply(c.config(c.no_sharing_genes(0)))
    c.restore()
    g = dict(zip(c.names, c.no_sharing_genes(0)))
    g.update(share_embed=1, share_k=0)
    stats = c.apply(c.config(c.decode([g[n] for n in c.names])))
    changed = not torch.equal(emb, orig)
    head_kept = torch.equal(model.get_output_embeddings().weight, c.engine._orig(c.layout.lm_head))
    c.restore()
    assert changed and head_kept and torch.equal(emb, orig)
    assert stats["size_mb"] < plain["size_mb"] and stats["k_merged"] <= 9


def test_decode_zeroes_genes_that_change_nothing():
    _, c, _, _ = make()
    # nothing shared: the candidate does not matter
    assert genes_of(c, 1, share_k=3, share_rho=2) == c.no_sharing_genes(1)
    # base 1 has bf16 qkv in layers 2-3 only (share bands 2-3), o everywhere, up nowhere, down in layers 0-1
    on = {n: 1 for n in c.names if n.startswith("share_") and n not in compress_stage2.CANDIDATE_GENES}
    g = dict(zip(c.names, genes_of(c, 1, share_k=2, **on)))
    assert [g[f"share_qkv_b{b}"] for b in range(4)] == [0, 0, 1, 1]
    assert [g[f"share_o_b{b}"] for b in range(4)] == [1, 1, 1, 1]
    assert [g[f"share_up_b{b}"] for b in range(4)] == [0, 0, 0, 0]
    assert [g[f"share_down_b{b}"] for b in range(4)] == [1, 1, 0, 0]
    assert g["share_head"] == 1 and g["share_k"] == 2
    # base 3 is all int4: nothing to share, so nothing changes (the bf16 lookup table of its tied int4 head holds the head's
    # int4 values, rebuilt from its codes)
    assert genes_of(c, 3, share_k=2, **on) == c.no_sharing_genes(3)
    assert not c.shared_any(c.config(c.all_shared_genes(3)))
    assert c.uniform_genes("bf16") == c.no_sharing_genes(0)


def test_config_shares_only_bf16_matrices():
    _, c, _, _ = make()
    cfg = c.config(c.all_shared_genes(2))
    assert cfg.base == 2
    assert cfg.fmt[0] == (SHARED, SHARED, "int4_g32", "int4_g32") and cfg.fmt[3] == ("int4_g128",) * 4
    assert cfg.lm_head == "int4_g64" and cfg.fmt[2] == (SHARED, SHARED, "int4_g64", "int4_g64")
    plain = c.config(c.no_sharing_genes(2))
    assert plain.fmt == compress_stage2.per_layer(compress_stage2.stage1_config(INT4_HEAD, FORMATS, (1.0,), 2, False), 4, 2).fmt
    assert "share k" in c.describe(cfg) and "share k" not in c.describe(plain) and c.describe(cfg).startswith("base 2: qkv:SS33")


# --- writing configurations


def test_nothing_shared_is_the_stage1_model_and_restore_is_exact():
    model = tiny_qwen()
    _, c, tokens, (table, act, order) = make(model)
    before = logits(model, tokens)
    hw = HWCompressor(model, c.layout, table, act, order, bands=2, mlp_keep=(1.0,), max_skip=0)
    hw.apply(compress_stage2.stage1_config(MIXED, FORMATS, (1.0,), 2, False))
    stage1 = logits(model, tokens)
    hw.restore()
    c.apply(c.config(c.no_sharing_genes(1)))
    assert torch.equal(logits(model, tokens), stage1)
    c.restore()
    assert torch.equal(logits(model, tokens), before)
    for genes in (c.all_shared_genes(1, share_k=0, share_prune=4, share_rho=4), c.all_shared_genes(2, share_k=1)):
        c.apply(c.config(genes))
        assert not torch.equal(logits(model, tokens), before)
        c.restore()
        assert torch.equal(logits(model, tokens), before)  # weights back, embedding hook removed


def test_uncompressed_size_and_cost():
    model, c, tokens, _ = make()
    cfg = c.config(c.uniform_genes("bf16"))
    stats = c.apply(cfg)
    c.restore()
    n_params = sum(p.numel() for p in model.parameters())  # the tied matrix once
    assert stats["size_mb"] == pytest.approx(2 * n_params / 1e6) and stats["shared_pct"] == 0 and stats["k_merged"] == 0
    assert stats["stored_bits_per_weight"] == pytest.approx(16.0, rel=0.01)  # 16, plus the q/k/v biases of the matrices
    cost = c.cost(cfg)
    assert cost["latency_ms"] == cost["tpot_ms"] == 10.0 and cost["memory_mb"] == 4.0 and cost["tok_s"] == 100.0
    assert cost["base"] == 0 and cost["base_quality"] == 0.0


def test_sharing_keeps_the_base_cost_and_int4_weights_and_shrinks_the_file():
    model, c, tokens, _ = make()
    plain_cfg = c.config(c.no_sharing_genes(1))
    plain = c.apply(plain_cfg)
    int4_w = {n: c.layout.module(n).weight.clone() for n in c.layout.blocks[0]["qkv"] + c.layout.blocks[0]["up"]}
    c.restore()
    shared_cfg = c.config(c.all_shared_genes(1, share_k=1, share_prune=3, share_rho=2))
    stats = c.apply(shared_cfg)
    for n, w in int4_w.items():  # int4 matrices are written exactly as without sharing
        assert torch.equal(c.layout.module(n).weight, w)
    o = c.layout.module(c.layout.blocks[0]["o"][0]).weight
    orig = c.engine._orig(c.layout.blocks[0]["o"][0])
    assert not torch.equal(o, orig)
    assert (o == 0).float().mean() > (orig == 0).float().mean() + 0.05  # the pruning interval zeroes weights
    c.restore()
    assert c.cost(shared_cfg) == c.cost(plain_cfg)
    assert stats["size_mb"] < plain["size_mb"] and 0 < stats["shared_pct"] < 100 and 0 < stats["k_merged"] <= 17


def test_shared_values_come_from_one_codebook():
    """Every shared weight is a row scale times one of the k_merged codebook values (zero included)."""
    _, c, _, _ = make()
    cfg = c.config(c.all_shared_genes(0, share_k=1, share_c=2, share_rho=2))  # k 17, c 0.9, half the levels
    stats = c.apply(cfg)
    name = c.layout.blocks[1]["o"][0]
    w = c.layout.module(name).weight.float()
    w0 = c.engine._orig(name).float()
    scale = (0.9 * w0.abs().amax(1, keepdim=True)) / 8
    units = torch.unique(torch.round((w / scale) * 1e3) / 1e3)
    c.restore()
    assert len(units) <= stats["k_merged"] <= 9


def test_more_levels_are_closer_and_larger():
    model, c, tokens, _ = make()
    ref = logits(model, tokens)
    errs, sizes = [], []
    for k in range(len(compress_stage2.SHARE_K)):
        stats = c.apply(c.config(c.all_shared_genes(0, share_k=k)))
        errs.append((logits(model, tokens) - ref).pow(2).mean().item())
        sizes.append(stats["size_mb"])
        c.restore()
    assert sizes == sorted(sizes) and errs[0] > errs[-1]


def test_stored_size_of_int4_and_shared_parts():
    """size = int4 codes at their entropy + 32 bits per group, bf16 at 16 bits, the rest at 16 bits."""
    model, c, _, _ = make()
    stats = c.apply(c.config(c.no_sharing_genes(3)))  # all int4, the tied head in int4 g32
    c.restore()
    e, bits = c.engine, 16.0 * c.rest_params
    for blk in c.layout.blocks:
        for role in ROLES:
            for name in blk[role]:
                q, scale, _ = e.int4(name, "int4_g32")
                bits += entropy_bits(torch.bincount(q.flatten(), minlength=16).numpy()) + 32 * scale.numel()
    q, scale, _ = e.int4(c.layout.lm_head, "int4_g32")
    bits += entropy_bits(torch.bincount(q.flatten(), minlength=16).numpy()) + 32 * scale.numel()
    bits += 16 * sum(e.layout.module(n).bias.numel() for blk in c.layout.blocks for n in blk["qkv"])  # q/k/v biases
    # nothing for the embedding: its lookup table is rebuilt from the head's int4 codes
    assert stats["size_mb"] == pytest.approx(bits / 8e6, rel=1e-9)


def test_int4_head_is_also_the_lookup_table():
    """A tied head in int4: the embedding lookup reads the head's int4 values, as in the deployed model (materialize)."""
    model, c, tokens, _ = make()
    emb = model.get_input_embeddings()
    c.apply(c.config(c.all_shared_genes(2)))
    head = model.get_output_embeddings().weight.detach().clone()
    q, scale, zero = c.engine.int4(c.layout.lm_head, "int4_g64")
    expected = compress_hw.int4_dequantize(q, scale, zero, 64).to(head.dtype)
    looked_up = emb(tokens)
    c.restore()
    assert torch.equal(head, expected)
    assert torch.equal(looked_up, torch.nn.functional.embedding(tokens, expected))


@cuda
def test_no_sharing_matches_the_deployed_stage1_model():
    """Nothing shared, the stage-2 model is the deployed stage-1 model (int4 on tinygemm, the bf16 embedding looked up)."""
    tokens = torch.randint(0, 512, (2, 32), device="cuda")
    model = tiny_qwen("cuda", torch.bfloat16)
    _, c, _, (table, act, order) = make(model)
    c.apply(c.config(c.no_sharing_genes(2)))
    fake = logits(model, tokens)
    c.restore()
    deployed = tiny_qwen("cuda", torch.bfloat16)
    hw = HWCompressor(deployed, Layout(deployed), table, act, order, bands=2, mlp_keep=(1.0,), max_skip=0)
    hw.materialize(compress_stage2.stage1_config(INT4_HEAD, FORMATS, (1.0,), 2, False))
    real = logits(deployed, tokens)
    assert ((real - fake).abs().max() / fake.abs().max()).item() < 3e-2  # bf16 rounding inside the kernel, as test_compress_hw
