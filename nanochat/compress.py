"""
Global weight-only compression of a causal LM: uniform quantization with K
levels combined with pruning of a weight-value interval. Works on any torch
model whose weight matrices are nn.Linear (HuggingFace models, nanochat GPT).

A candidate is 4 numbers shared by every compressed matrix ("global"):
    k      number of uniform quantization levels (integer >= 3)
    c      clipping: each output row's range is c * max|w_row|, c in (0, 1]
    alpha  lower pruning bound, in units of the row's std (<= 0)
    beta   upper pruning bound, in units of the row's std (>= 0)

Per matrix W (rows = output channels), for a weight w in row r:
    1) prune:    w -> 0 if alpha * std_r <= w <= beta * std_r
    2) quantize: symmetric, zero-preserving integer grid q in [-(k//2), (k-1)//2]
                 with step scale_r = c * max|w_row| / ((k-1)//2), values outside clamp
    3) re-mask:  pruned weights stay exactly 0 (they are index 0 on the grid)
    4) reconstruct each index q as
         grid:     scale_r * q                      (the center of the bin)
         centroid: scale_r * m_q, where m_q is the mean of w/scale over all weights
                   of the matrix that landed in bin q (bin edges stay uniform).
                   The zero bin keeps m_0 = 0 so pruned weights stay exactly zero.
std_r and max|w_row| are always taken from the ORIGINAL weights, so a gene value
means the same thing for every candidate. With prune=False step 1 is skipped (alpha and
beta are ignored): the only zeros are weights that quantize to 0.

Stored codes address only the levels a matrix actually uses: k is the size of the grid
(k_init), k_used <= k the number of its levels that at least one weight of the matrix
(any row) lands on; a per-matrix lookup table maps each code back to its level.
Encoded size (bits) of a matrix with n weights, R rows, C columns, nnz nonzeros,
v = ceil(log2(number of used nonzero levels)):
    fixed:   smallest of the enabled formats, + 16*R scales + lookup table
               dense   n * ceil(log2 k_used)
               bitmap  n + nnz*v                        (1 bit per weight: zero or not)
               csr     nnz*(v + ceil(log2 C)) + (R+1)*ceil(log2(nnz+1))   (column index
                       per nonzero + row pointers, minimal integer widths)
    entropy: n * H(q) + 16*R scales + lookup table, H = empirical entropy of the codes
    lookup table: k_used entries of ceil(log2 k) bits (the grid level) with grid
             reconstruction, or of 16 bits (the centroid value) with centroid
Optionally (apply(huffman=True), for reporting only) the size with the codes Huffman
coded: sum(count_i * codelength_i) + k_used * (8-bit code length + table entry) + 16*R.
Every parameter that is not compressed counts 16 bits.
"""

import heapq
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Candidate:
    k: int
    c: float
    alpha: float
    beta: float


def find_target_linears(model, include_lm_head=False, exclude=(), include_embeddings=False):
    """Return [(name, module)] of the nn.Linear layers to compress, each weight once.
    exclude: skip modules whose name contains any of these strings (e.g. tiny gates).
    include_embeddings: also compress nn.Embedding tables (one row per token), except the
    output projection's weight when it is tied to one and include_lm_head is off."""
    skip = set()
    if not include_lm_head and hasattr(model, "get_output_embeddings"):
        out = model.get_output_embeddings()
        if out is not None:
            skip.add(out.weight.data_ptr())
    kinds = (nn.Linear, nn.Embedding) if include_embeddings else (nn.Linear,)
    targets, seen = [], set()
    for name, module in model.named_modules():
        if not isinstance(module, kinds) or any(e in name for e in exclude):
            continue
        ptr = module.weight.data_ptr()
        if ptr in skip or ptr in seen:
            continue
        seen.add(ptr)
        targets.append((name, module))
    return targets


@torch.no_grad()
def compress_weight(w, row_std, cand, prune=True):
    """Prune + quantize one matrix. Returns (integer grid q as float32, per-row scale, pruned mask)."""
    assert cand.k >= 3, "k must be >= 3 so the grid has a positive level"
    w = w.float()
    qmax = (cand.k - 1) // 2
    qmin = -(cand.k // 2)
    if prune:
        pruned = (w >= cand.alpha * row_std) & (w <= cand.beta * row_std)
    else:
        pruned = torch.zeros_like(w, dtype=torch.bool)
    clip = cand.c * w.abs().amax(dim=1, keepdim=True)
    scale = (clip / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    q.masked_fill_(pruned, 0.0)
    return q, scale, pruned


@torch.no_grad()
def centroid_levels(w, q, scale, k):
    """Mean of w/scale in each bin of the matrix (in grid units), zero bin pinned to 0."""
    qmin = -(k // 2)
    idx = (q - qmin).long().flatten()
    u = (w.float() / scale).flatten()
    sums = torch.bincount(idx, weights=u, minlength=k)
    counts = torch.bincount(idx, minlength=k)
    grid = torch.arange(qmin, qmin + k, device=q.device, dtype=sums.dtype)
    levels = torch.where(counts > 0, sums / counts.clamp_min(1), grid)  # empty bin: never used, keep grid value
    levels[-qmin] = 0.0
    return levels.float()


FORMATS = ("dense", "bitmap", "csr")


def ceil_log2(x):
    return math.ceil(math.log2(x)) if x > 1 else 0


@torch.no_grad()
def level_counts(q, k):
    """How many weights of the matrix land on each of the k grid levels (index 0 = level -(k//2))."""
    return torch.bincount((q.flatten() + k // 2).long(), minlength=k)


def lut_entry_bits(k, reconstruction):
    """One lookup-table entry: the grid level it stands for, or the centroid value."""
    return 16 if reconstruction == "centroid" else ceil_log2(k)


@torch.no_grad()
def format_bits(q, cand, counts=None):
    """Code bits of q in each fixed-width storage format, with codes over the used levels only
    (scales and the lookup table not included)."""
    n, (rows, cols) = q.numel(), q.shape
    counts = level_counts(q, cand.k) if counts is None else counts
    n_zero = int(counts[cand.k // 2])
    nnz = n - n_zero
    k_used = int((counts > 0).sum())
    value_bits = ceil_log2(k_used - (n_zero > 0))  # a nonzero is one of the used nonzero levels
    return {
        "dense": n * ceil_log2(k_used),
        "bitmap": n + nnz * value_bits,
        "csr": nnz * (value_bits + ceil_log2(cols)) + (rows + 1) * ceil_log2(nnz + 1),
    }


def huffman_bits(counts):
    """Total bits of the Huffman-coded symbols with these frequencies (code table not included)."""
    heap = [int(c) for c in counts if c > 0]
    if len(heap) == 1:
        return heap[0]  # a single symbol still costs 1 bit per occurrence
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:  # every merge adds one bit to the code of every symbol below it
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        total += merged
        heapq.heappush(heap, merged)
    return total


@torch.no_grad()
def encoded_bits(q, cand, size_mode="fixed", formats=("dense", "bitmap"), reconstruction="grid"):
    """Bits needed to store q with its fp16 per-row scales and lookup table.
    Returns (bits, format, number of used levels)."""
    n, rows = q.numel(), q.shape[0]
    counts = level_counts(q, cand.k)
    k_used = int((counts > 0).sum())
    side_bits = 16 * rows + k_used * lut_entry_bits(cand.k, reconstruction)
    if size_mode == "fixed":
        bits = format_bits(q, cand, counts)
        fmt = min(formats, key=bits.get)
        return bits[fmt] + side_bits, fmt, k_used
    if size_mode == "entropy":
        p = counts[counts > 0].double() / n
        return float(-(p * torch.log2(p)).sum()) * n + side_bits, "entropy", k_used
    raise ValueError(f"unknown size_mode {size_mode}")


class GlobalCompressor:
    """
    Holds the original weights of the target matrices and writes compressed
    versions of them into the model in place, one candidate at a time.
    Originals live on the GPU when they fit (fast) or in pinned CPU memory.
    """

    def __init__(self, model, include_lm_head=False, size_mode="fixed", offload="auto", reconstruction="grid",
                 formats=("dense", "bitmap"), prune=True, exclude=(), include_embeddings=False):
        assert reconstruction in ("grid", "centroid")
        assert formats and set(formats) <= set(FORMATS), f"formats must be a subset of {FORMATS}"
        self.model = model
        self.size_mode = size_mode
        self.formats = tuple(formats)
        self.reconstruction = reconstruction
        self.prune = prune
        self.targets = find_target_linears(model, include_lm_head, exclude, include_embeddings)
        assert self.targets, "no layers found to compress"
        device = self.targets[0][1].weight.device
        target_bytes = sum(m.weight.numel() * m.weight.element_size() for _, m in self.targets)
        if offload == "auto":
            offload = "no"
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                # originals + fp32 working copies of the largest matrix + activations headroom
                offload = "yes" if target_bytes > 0.6 * free else "no"
        self.offloaded = offload == "yes"
        self.originals, self.row_stds = [], []
        for _, m in self.targets:
            w = m.weight.detach()
            self.row_stds.append(w.float().std(dim=1, keepdim=True))
            orig = w.clone()
            if self.offloaded:
                orig = orig.cpu().pin_memory() if device.type == "cuda" else orig.cpu()
            self.originals.append(orig)
        target_ptrs = {m.weight.data_ptr() for _, m in self.targets}
        self.n_target_params = sum(m.weight.numel() for _, m in self.targets)
        # model.parameters() already yields tied parameters only once
        self.rest_params = sum(p.numel() for p in model.parameters() if p.data_ptr() not in target_ptrs)
        self.baseline_bits = 16 * (self.n_target_params + self.rest_params)

    @torch.no_grad()
    def apply(self, cand, huffman=False):
        """Write the compressed weights for cand into the model. Returns size stats."""
        bits, zeros, n_pruned, huff_bits = 0.0, 0, 0, 0
        n_format = {f: 0 for f in FORMATS}
        k_used, row_levels, n_rows = [], 0, 0
        for (_, m), orig, row_std in zip(self.targets, self.originals, self.row_stds):
            w0 = orig.to(m.weight.device, non_blocking=True)
            q, scale, pruned = compress_weight(w0, row_std, cand, self.prune)
            matrix_bits, fmt, matrix_k_used = encoded_bits(q, cand, self.size_mode, self.formats, self.reconstruction)
            bits += matrix_bits
            k_used.append(matrix_k_used)
            # levels used within each row (a per-matrix table cannot drop a level that any row uses)
            idx = (q + cand.k // 2).long()
            used = torch.zeros(q.shape[0], cand.k, dtype=torch.bool, device=q.device).scatter_(1, idx, True)
            row_levels += int(used.sum())
            n_rows += q.shape[0]
            if fmt in n_format:
                n_format[fmt] += 1
            zeros += q.numel() - int(torch.count_nonzero(q))
            n_pruned += int(pruned.sum())
            if huffman:
                counts = level_counts(q, cand.k)
                huff_bits += (huffman_bits(counts.tolist()) + 16 * q.shape[0]
                              + matrix_k_used * (8 + lut_entry_bits(cand.k, self.reconstruction)))
            if self.reconstruction == "centroid":
                levels = centroid_levels(w0, q, scale, cand.k)
                q = levels[idx]
            m.weight.copy_((q * scale).to(m.weight.dtype))
        total_bits = bits + 16 * self.rest_params
        n = self.n_target_params
        stats = {
            "size_mb": total_bits / 8 / 1e6,
            "size_ratio": total_bits / self.baseline_bits,
            "target_bits_per_weight": bits / n,
            "n_pruned": n_pruned,               # zeroed by the pruning interval
            "n_rounded_zero": zeros - n_pruned,  # outside the interval, but quantized to 0
            "n_nonzero": n - zeros,
            "pruned_pct": 100 * n_pruned / n,
            "sparsity_pct": 100 * zeros / n,     # all zeros, of the compressed weights
            **{f"n_{f}_matrices": c for f, c in n_format.items()},  # storage format chosen per matrix
            "k_used_mean": sum(k_used) / len(k_used),  # levels stored per matrix (lookup table size)
            "k_used_min": min(k_used),
            "k_used_max": max(k_used),
            "k_used_row_mean": row_levels / n_rows,     # levels used within a single row, on average
        }
        if huffman:
            stats["huffman_size_mb"] = (huff_bits + 16 * self.rest_params) / 8 / 1e6
            stats["huffman_bits_per_weight"] = huff_bits / n
        return stats

    @torch.no_grad()
    def restore(self):
        for (_, m), orig in zip(self.targets, self.originals):
            m.weight.copy_(orig.to(m.weight.device, non_blocking=True))
