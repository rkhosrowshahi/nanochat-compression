"""
Test the global quantization + pruning used by scripts/compress_search.py.
Runs on CPU.

python -m pytest tests/test_compress.py -v
"""

import math

import pytest
import torch
import torch.nn as nn

from nanochat.compress import Candidate, GlobalCompressor, centroid_levels, compress_weight, encoded_bits, find_target_linears, format_bits, huffman_bits


def make_weight(rows=64, cols=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=gen) * 0.02


class TinyLM(nn.Module):
    """Two linears, an embedding and a tied output head, like a small HF model."""

    def __init__(self, vocab=50, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_output_embeddings(self):
        return self.lm_head


@pytest.mark.parametrize("k", [3, 4, 16, 17, 255])
def test_grid_has_at_most_k_levels(k):
    w = make_weight()
    q, _, _ = compress_weight(w, w.std(dim=1, keepdim=True), Candidate(k, 1.0, 0.0, 0.0))
    assert q.unique().numel() <= k
    assert q.min() >= -(k // 2) and q.max() <= (k - 1) // 2


def test_pruning_interval_is_zeroed_and_relative_to_row_std():
    w = make_weight()
    w[0] *= 100  # a row with a very different scale must be pruned the same fraction
    std = w.std(dim=1, keepdim=True)
    q, _, mask = compress_weight(w, std, Candidate(255, 1.0, -0.5, 0.25))
    pruned = (w >= -0.5 * std) & (w <= 0.25 * std)
    assert (q[pruned] == 0).all()
    assert torch.equal(mask, pruned)
    frac = pruned.float().mean(dim=1)
    assert abs(frac[0] - frac[1:].mean()) < 0.1


def test_clipping_limits_the_range():
    w = make_weight()
    k = 15
    q, scale, _ = compress_weight(w, w.std(dim=1, keepdim=True), Candidate(k, 0.5, 0.0, 0.0))
    w_hat = q * scale
    assert torch.allclose(w_hat.abs().amax(dim=1), 0.5 * w.abs().amax(dim=1), rtol=1e-5)


def test_fine_grid_without_pruning_is_accurate():
    w = make_weight()
    # alpha = beta = 0 prunes only exact zeros, which randn does not produce
    q, scale, _ = compress_weight(w, w.std(dim=1, keepdim=True), Candidate(255, 1.0, 0.0, 0.0))
    assert (q * scale - w).abs().max() <= scale.max() / 2 + 1e-7


def test_encoded_bits_fixed_and_entropy():
    rows, cols, k = 8, 32, 16
    lut = math.ceil(math.log2(k))  # bits per lookup-table entry (the grid level)
    q = torch.zeros(rows, cols)
    cand = Candidate(k, 1.0, 0.0, 0.0)
    # all zeros: a single used level needs no code bits, only the scales and a 1-entry table
    assert encoded_bits(q, cand, "fixed") == (16 * rows + lut, "dense", 1)
    assert encoded_bits(q, cand, "entropy")[0] == pytest.approx(16 * rows + lut)
    # uniform over all k levels: fixed = entropy = log2(k) bits/weight, full table
    q = torch.arange(rows * cols).remainder(k).view(rows, cols).float() - k // 2
    assert encoded_bits(q, cand, "fixed")[0] == rows * cols * math.ceil(math.log2(k)) + 16 * rows + k * lut
    assert encoded_bits(q, cand, "entropy")[0] == pytest.approx(rows * cols * math.log2(k) + 16 * rows + k * lut)


def test_codes_address_only_used_levels():
    rows, cols, k = 8, 32, 255  # an 8-bit grid ...
    cand = Candidate(k, 1.0, 0.0, 0.0)
    q = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0]).repeat(rows * cols // 5 + 1)[:rows * cols].view(rows, cols)
    n, nnz = rows * cols, int(torch.count_nonzero(q))
    bits, fmt, k_used = encoded_bits(q, cand, "fixed", formats=("dense",))
    assert k_used == 5  # ... of which 5 levels are used: 3-bit codes + a 5-entry table of 8-bit levels
    assert bits == n * 3 + 16 * rows + 5 * 8
    # bitmap: the 4 used nonzero levels need 2-bit codes
    assert encoded_bits(q, cand, "fixed", formats=("bitmap",))[0] == n + nnz * 2 + 16 * rows + 5 * 8
    # centroid: the table holds 16-bit values instead of 8-bit grid levels
    assert encoded_bits(q, cand, "fixed", formats=("dense",), reconstruction="centroid")[0] == n * 3 + 16 * rows + 5 * 16


def test_targets_skip_tied_output_head():
    model = TinyLM()
    assert [name for name, _ in find_target_linears(model)] == ["fc1", "fc2"]
    assert [name for name, _ in find_target_linears(model, include_lm_head=True)] == ["fc1", "fc2", "lm_head"]


def test_targets_exclude_and_embeddings():
    model = TinyLM()
    assert [name for name, _ in find_target_linears(model, exclude=("fc2",))] == ["fc1"]
    # the embedding is tied to the output head, which stays uncompressed
    assert [name for name, _ in find_target_linears(model, include_embeddings=True)] == ["fc1", "fc2"]
    model.lm_head.weight = nn.Parameter(model.embed.weight.detach().clone())  # untie
    assert [name for name, _ in find_target_linears(model, include_embeddings=True)] == ["embed", "fc1", "fc2"]
    comp = GlobalCompressor(model, include_embeddings=True)
    before = model.embed.weight.detach().clone()
    comp.apply(Candidate(5, 0.8, -0.5, 0.5))
    assert not torch.equal(model.embed.weight, before)
    comp.restore()
    assert torch.equal(model.embed.weight, before)


def test_apply_changes_weights_and_restore_is_exact():
    torch.manual_seed(0)
    model = TinyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    comp = GlobalCompressor(model)
    stats = comp.apply(Candidate(5, 0.8, -0.5, 0.5))
    assert not torch.equal(model.fc1.weight, before["fc1.weight"])
    assert torch.equal(model.embed.weight, before["embed.weight"])  # not a target
    assert 0 < stats["sparsity_pct"] < 100
    assert 1 <= stats["k_used_min"] <= stats["k_used_mean"] <= stats["k_used_max"] <= 5
    assert stats["k_used_row_mean"] <= stats["k_used_max"]
    assert stats["n_pruned"] + stats["n_rounded_zero"] + stats["n_nonzero"] == comp.n_target_params
    assert stats["size_ratio"] < 1
    comp.restore()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n


def test_no_pruning_zeros_only_where_quantization_rounds_to_zero():
    w = make_weight()
    std = w.std(dim=1, keepdim=True)
    cand = Candidate(255, 1.0, -0.5, 0.5)  # alpha/beta must be ignored
    q, scale, pruned = compress_weight(w, std, cand, prune=False)
    assert not pruned.any()
    assert torch.equal(q == 0, torch.round(w / scale) == 0)
    torch.manual_seed(0)
    model = TinyLM()
    comp = GlobalCompressor(model, prune=False)
    stats = comp.apply(cand)
    comp.restore()
    assert stats["n_pruned"] == 0 and stats["pruned_pct"] == 0


def test_size_accounting_counts_tied_params_once():
    model = TinyLM()
    comp = GlobalCompressor(model)
    n_params = sum(p.numel() for p in model.parameters())
    assert comp.n_target_params + comp.rest_params == n_params
    assert comp.baseline_bits == 16 * n_params


@pytest.mark.parametrize("k", [3, 9, 17])
def test_centroids_reduce_error_and_keep_zeros(k):
    w = make_weight()
    std = w.std(dim=1, keepdim=True)
    cand = Candidate(k, 0.8, -0.3, 0.3)
    q, scale, _ = compress_weight(w, std, cand)
    levels = centroid_levels(w, q, scale, k)
    assert levels.numel() == k and levels[k // 2] == 0
    w_grid = q * scale
    w_cent = levels[(q + k // 2).long()] * scale
    assert torch.equal(w_cent == 0, w_grid == 0)  # same sparsity pattern
    assert ((w_cent - w) ** 2).sum() <= ((w_grid - w) ** 2).sum()


def test_centroid_compressor_charges_codebook_and_restores():
    torch.manual_seed(0)
    model = TinyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cand = Candidate(5, 0.8, -0.5, 0.5)
    grid_comp = GlobalCompressor(model)
    grid = grid_comp.apply(cand)
    grid_comp.restore()
    comp = GlobalCompressor(model, reconstruction="centroid")
    cent = comp.apply(cand)
    # centroid table entries are 16-bit values instead of ceil(log2 5) = 3-bit grid levels
    n_levels = cent["k_used_mean"] * len(comp.targets)
    assert cent["size_mb"] * 8e6 == pytest.approx(grid["size_mb"] * 8e6 + (16 - 3) * n_levels)
    assert cent["sparsity_pct"] == grid["sparsity_pct"]
    comp.restore()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n


def test_csr_wins_only_when_very_sparse():
    rows, cols, k = 16, 1024, 17  # 10 column bits per nonzero
    cand = Candidate(k, 1.0, 0.0, 0.0)
    q = torch.zeros(rows, cols)
    q[:, :8] = torch.tensor([1.0, 2, 3, 4, -1, -2, -3, -4])  # 8 nonzero levels used: 3 value bits; 0.8% dense
    bits = format_bits(q, cand)
    nnz = rows * 8
    assert bits["csr"] == nnz * (3 + 10) + (rows + 1) * math.ceil(math.log2(nnz + 1))
    assert bits["csr"] < bits["bitmap"] < bits["dense"]
    assert encoded_bits(q, cand, "fixed", ("dense", "bitmap", "csr"))[1] == "csr"
    assert encoded_bits(q, cand, "fixed")[1] == "bitmap"  # csr is off by default
    q[:, :cols // 2] = q[:, :8].repeat(1, cols // 16)  # half nonzero: csr loses to the bitmap
    assert encoded_bits(q, cand, "fixed", ("dense", "bitmap", "csr"))[1] == "bitmap"


def test_huffman_bits():
    assert huffman_bits([5]) == 5
    assert huffman_bits([1, 1, 1, 1]) == 8  # 4 equal symbols: 2 bits each
    assert huffman_bits([8, 4, 2, 2]) == 8 * 1 + 4 * 2 + 2 * 3 + 2 * 3
    assert huffman_bits([3, 0, 3]) == 6  # unused levels get no code


def test_huffman_size_between_entropy_and_fixed():
    torch.manual_seed(0)
    model = TinyLM()
    comp = GlobalCompressor(model)
    stats = comp.apply(Candidate(17, 0.9, -0.2, 0.2), huffman=True)
    comp.restore()
    assert 0 < stats["huffman_bits_per_weight"] < stats["target_bits_per_weight"] + 1
