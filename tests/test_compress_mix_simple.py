"""
Test the simplified mix of scripts/compress_search_mix_simple.py and scripts/compress_storage.py on a tiny random HF Qwen2 with
made-up latency tables.

python -m pytest tests/test_compress_mix_simple.py -v
"""

import pytest
import torch

from nanochat import compress_hw
from nanochat.compress_hw import Layout
from nanochat.compress_mix import SHARED, MixCompressor
from nanochat.compress_mix_simple import has_bf16, shared_storage, storage_front
from tests.test_compress_mix import fake_table, logits, tiny_qwen


@pytest.fixture
def setup():
    model = tiny_qwen()
    layout = Layout(model)
    tokens = torch.randint(0, 512, (2, 32))
    act = compress_hw.activation_stats(model, layout, tokens)
    order = {"attn": [(i, 0.0) for i in range(layout.n_layer)], "mlp": [(i, 0.0) for i in range(layout.n_layer)]}
    compressor = MixCompressor(model, layout, fake_table(layout, 4, 0.5), fake_table(layout, 64, 4.0), act, order, bands=2, max_skip=1, sharing=False,
                               new_tokens=100, gen_tokens=10, vram_gb=1.0)
    return model, compressor, tokens


def test_no_sharing_in_the_search(setup):
    _, c, _ = setup
    assert SHARED not in c.formats
    assert not any(n.startswith("share_") for n, _ in c.genes)
    for genes in (c.uniform_genes("bf16"), c.uniform_genes("int4_g64")):
        assert c.decode(genes) == genes
        cfg = c.config(genes)
        assert cfg.share is None and "share:" not in c.describe(cfg)


def test_genes_are_those_of_the_hardware_search(setup):
    model, c, _ = setup
    hw = compress_hw.HWCompressor(model, c.layout, c.table, c.act_sq, {k: [(i, 0.0) for i in v] for k, v in c.order.items()},
                                  bands=2, max_skip=1)
    assert c.genes == hw.genes  # so compress_storage also reads the fronts of compress_search_hw


def test_storage_keeps_time_and_memory_and_shrinks_the_file(setup):
    model, c, tokens = setup
    genes = dict(zip((n for n, _ in c.genes), c.uniform_genes("bf16")))
    genes.update(qkv_b0=c.formats.index("int4_g64"))
    source = c.config(c.decode(tuple(genes.values())))
    before = logits(model, tokens)
    plain = c.apply(source)
    plain_logits = logits(model, tokens)
    c.restore()
    shared = shared_storage(source, 33)
    assert c.cost(shared) == c.cost(source)
    assert shared.fmt[0][0] == "int4_g64" and shared.fmt[0][1] == SHARED and shared.lm_head == SHARED
    stats = c.apply(shared)
    shared_logits = logits(model, tokens)
    c.restore()
    assert stats["size_mb"] < plain["size_mb"] / 2 and stats["shared_pct"] > 0
    assert (shared_logits != plain_logits).any()
    assert (logits(model, tokens) == before).all()
    assert "share:k33 c1 p0 rho1" in c.describe(shared)


def test_more_levels_are_closer_and_larger(setup):
    model, c, tokens = setup
    source = c.config(c.uniform_genes("bf16"))
    before = logits(model, tokens)
    sizes, errors = [], []
    for k in (17, 257):
        stats = c.apply(shared_storage(source, k))
        errors.append((logits(model, tokens) - before).abs().mean().item())
        c.restore()
        sizes.append(stats["size_mb"])
    assert sizes[0] < sizes[1] and errors[0] > errors[1]


def test_all_int4_has_nothing_to_share(setup):
    _, c, _ = setup
    assert has_bf16(c.config(c.uniform_genes("bf16")))
    assert not has_bf16(c.config(c.uniform_genes("int4_g128")))


def test_storage_front():
    rows = [dict(delta_loss=d, size_mb=s) for d, s in ((0.0, 100), (0.1, 50), (0.2, 60), (0.2, 40), (0.3, 40), (-0.1, 200))]
    assert [(r["delta_loss"], r["size_mb"]) for r in storage_front(rows)] == [(-0.1, 200), (0.0, 100), (0.1, 50), (0.2, 40)]
