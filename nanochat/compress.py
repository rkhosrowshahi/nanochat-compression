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

UBQ-ECSQ (method="ubq-ecsq"; ported from moea-compression, utils/compression/ecsq.py and
moea/ecsq.py): a fifth gene rho merges the grid's levels after steps 1-3:
    4) each non-empty level gets its global centroid: the mean of w/scale over all weights of all
       compressed matrices on that level (one codebook for the whole model, in row-scale units)
    5) entropy-constrained merge: contiguous levels are grouped to minimize J = D + lambda*R
       (D = added squared error, R = entropy of level occupancy), exactly by dynamic programming;
       sweeping lambda gives one partition per reachable size (the "ladder"), and the size nearest
       rho * (non-empty levels) is used (rho = 1: no merge). The zero level (pruned weights and
       weights rounded to 0) is never merged and stays exactly 0, so the negative and positive
       levels are merged separately with the same lambda (the same as the full DP with zero
       forced to stay its own group).
    Storage: per matrix, codes over its used merged levels and a lookup table of k_used entries
    of ceil(log2 K') bits pointing into the global codebook, which costs K' fp16 values once.
"""

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Candidate:
    k: int
    c: float
    alpha: float
    beta: float
    rho: float = 1.0  # UBQ-ECSQ only: fraction of the non-empty levels kept by the merge (1 = no merge)


METHODS = ("ubq", "ubq-ecsq")


def dp_merge(counts, centers, lam, n_total=None):
    """Optimal contiguous grouping of bins under J = D + lam*R (moea-compression's dp_merge;
    n_total = the population the rate is measured against, default the bins' own).
    D = sum_g m_g (c_g - c_merged)^2 (between-group squared error; the within-bin error does not
    change with the grouping), R = sum over merged bins of -m log2(m / N).
    Returns (group_of_bin, merged_centers)."""
    m = np.asarray(counts, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    K = len(m)
    if K == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    N = m.sum() if n_total is None else float(n_total)
    M = np.concatenate([[0.0], np.cumsum(m)])
    MC = np.concatenate([[0.0], np.cumsum(m * c)])
    MC2 = np.concatenate([[0.0], np.cumsum(m * c * c)])
    dp = np.full(K + 1, np.inf)
    dp[0] = 0.0
    parent = np.zeros(K + 1, dtype=np.int64)
    for j in range(1, K + 1):
        i = np.arange(j)
        mm = M[j] - M[i]
        ss = MC[j] - MC[i]
        dD = np.maximum((MC2[j] - MC2[i]) - ss * ss / mm, 0.0)  # clip fp noise
        dR = -mm * np.log2(mm / N)
        tot = dp[i] + dD + lam * dR
        k = int(np.argmin(tot))
        dp[j], parent[j] = tot[k], k
    groups, j = [], K
    while j > 0:
        groups.append((parent[j], j))
        j = parent[j]
    groups.reverse()
    group_of = np.empty(K, dtype=np.int64)
    merged = np.empty(len(groups), dtype=np.float64)
    for g, (a, b) in enumerate(groups):
        group_of[a:b] = g
        merged[g] = (MC[b] - MC[a]) / (M[b] - M[a])
    return group_of, merged


def ecsq_ladder(counts, sums, k, n_lambdas=160):
    """{K': plan} for every merged size reachable by the Lagrangian, from the global per-level counts
    and sums of w/scale (index 0 = level -(k//2)). A plan is (level -> code, code -> value) as numpy
    arrays: codes are signed (0 = the zero level, -1, -2, ... below it, 1, 2, ... above it), values
    in row-scale units. The lambda grid is scaled to the data: from 0 (no merge) up to where each
    side collapses into one group."""
    counts = np.asarray(counts, dtype=np.float64)
    sums = np.asarray(sums, dtype=np.float64)
    zero = k // 2
    neg = np.flatnonzero(counts[:zero] > 0)
    pos = zero + 1 + np.flatnonzero(counts[zero + 1:] > 0)
    centers = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    n_total = counts.sum() - counts[zero]
    has_zero = counts[zero] > 0
    # lambda at which merging a whole side into one group beats keeping its levels apart
    d_all = r_gain = 0.0
    for side in (neg, pos):
        if len(side):
            m, c = counts[side], centers[side]
            d_all += (m * c * c).sum() - (m * c).sum() ** 2 / m.sum()
            r_gain += (-m * np.log2(m / n_total)).sum() + m.sum() * np.log2(m.sum() / n_total)
    lam_hi = 10.0 * max(d_all, 1e-30) / max(r_gain, 1e-12)
    ladder = {}
    for lam in np.concatenate([[0.0], np.geomspace(lam_hi * 1e-10, lam_hi, n_lambdas)]):
        g_neg, v_neg = dp_merge(counts[neg], centers[neg], lam, n_total)
        g_pos, v_pos = dp_merge(counts[pos], centers[pos], lam, n_total)
        size = len(v_neg) + len(v_pos) + int(has_zero)
        if size in ladder:
            continue
        code = np.zeros(k, dtype=np.int64)
        code[neg] = g_neg - len(v_neg)  # most negative group -> -len(v_neg), the one next to zero -> -1
        code[pos] = g_pos + 1
        value = np.concatenate([v_neg, [0.0], v_pos])  # indexed by code + len(v_neg)
        ladder[size] = (code, value, len(v_neg))
    return dict(sorted(ladder.items()))


def nearest_rung(ladder, k_target):
    """Rung of the ladder closest to k_target (ties -> the larger, i.e. less merged)."""
    sizes = np.fromiter(ladder.keys(), dtype=np.int64)
    best = int(sizes[np.argmin(np.abs(sizes - int(k_target)) - 1e-9 * sizes)])
    return best, ladder[best]


INCLUDE_GROUPS = ("lm_head", "embeddings")


def find_target_linears(model, include=(), exclude=()):
    """Return [(name, module)] of the layers to compress, each weight once.
    By default: every nn.Linear except the output projection (lm_head). include adds groups:
        "lm_head"     the output projection
        "embeddings"  every embedding table (nn.Embedding, one row per token): the token
                      embedding, and in nanochat also the value embeddings
    A token embedding tied to lm_head (e.g. Qwen2.5) is one matrix, compressed once when either
    group is included. exclude: skip modules whose name contains any of these strings (tiny gates)."""
    unknown = set(include) - set(INCLUDE_GROUPS)
    assert not unknown, f"unknown include group(s) {sorted(unknown)}, choose from {INCLUDE_GROUPS}"
    skip = set()
    if "lm_head" not in include and hasattr(model, "get_output_embeddings"):
        out = model.get_output_embeddings()
        if out is not None:
            skip.add(out.weight.data_ptr())
    kinds = (nn.Linear, nn.Embedding) if "embeddings" in include else (nn.Linear,)
    targets, seen = [], set()
    for name, module in model.named_modules():
        if not isinstance(module, kinds) or any(e in name for e in exclude):
            continue
        ptr = module.weight.data_ptr()
        # the output projection is skipped as a Linear; a token embedding tied to it is still an embedding
        if ptr in seen or (ptr in skip and not isinstance(module, nn.Embedding)):
            continue
        seen.add(ptr)
        targets.append((name, module))
    return targets


def include_from_config(run_args):
    """The --include groups of a run's saved arguments (runs from before --include stored two flags)."""
    if "include" in run_args:
        return tuple(run_args["include"])
    return tuple(g for g, flag in (("lm_head", "include_lm_head"), ("embeddings", "include_embeddings"))
                 if run_args.get(flag))


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

    def __init__(self, model, include=(), size_mode="fixed", offload="auto", reconstruction="grid",
                 formats=("dense", "bitmap"), prune=True, exclude=(), method="ubq"):
        assert reconstruction in ("grid", "centroid")
        assert method in METHODS, f"method must be one of {METHODS}"
        self.method = method
        self._level_stats = {}            # UBQ-ECSQ: (k, c, alpha, beta) -> global per-level counts and sums
        self._ladders = OrderedDict()     # and their merge ladders (a few recent ones; each is O(k^2))
        self._rungs = {}                  # and the merged sizes each ladder reaches (small, all kept)
        assert formats and set(formats) <= set(FORMATS), f"formats must be a subset of {FORMATS}"
        self.model = model
        self.size_mode = size_mode
        self.formats = tuple(formats)
        self.reconstruction = reconstruction
        self.prune = prune
        self.targets = find_target_linears(model, include, exclude)
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
    def level_stats(self, cand):
        """UBQ-ECSQ: per grid level, over all compressed weights: (counts, sums of w/scale). Cached."""
        key = (cand.k, round(cand.c, 9), round(cand.alpha, 9), round(cand.beta, 9))
        if key not in self._level_stats:
            counts = sums = 0
            for (_, m), orig, row_std in zip(self.targets, self.originals, self.row_stds):
                w0 = orig.to(m.weight.device, non_blocking=True)
                q, scale, _ = compress_weight(w0, row_std, cand, self.prune)
                idx = (q + cand.k // 2).long().flatten()
                counts = counts + torch.bincount(idx, minlength=cand.k).double()
                sums = sums + torch.bincount(idx, weights=(w0.float() / scale).flatten().double(), minlength=cand.k)
            self._level_stats[key] = (counts.cpu().numpy(), sums.cpu().numpy())
        return self._level_stats[key]

    def merge_plan(self, cand):
        """UBQ-ECSQ: (K' reached, non-empty levels, (level -> code, code -> value, code offset)) for cand."""
        key = (cand.k, round(cand.c, 9), round(cand.alpha, 9), round(cand.beta, 9))
        counts, sums = self.level_stats(cand)
        if key not in self._ladders:
            self._ladders[key] = ecsq_ladder(counts, sums, cand.k)
            self._rungs[key] = {size: None for size in self._ladders[key]}
            if len(self._ladders) > 32:
                self._ladders.popitem(last=False)
        self._ladders.move_to_end(key)
        k_realized = int((counts > 0).sum())
        k_merged, plan = nearest_rung(self._ladders[key], max(1, round(cand.rho * k_realized)))
        return k_merged, k_realized, plan

    def canonical_rho(self, cand):
        """The rho of the merge actually reached (K' / non-empty levels): every rho that leads to the
        same K' gives the same model, so this identifies duplicates (moea-compression's ECSQRepair)."""
        if self.method != "ubq-ecsq":
            return 1.0
        key = (cand.k, round(cand.c, 9), round(cand.alpha, 9), round(cand.beta, 9))
        if key not in self._rungs:  # NSGA-II's duplicate check calls this pairwise: keep it a lookup
            self.merge_plan(cand)
        k_realized = int((self.level_stats(cand)[0] > 0).sum())
        k_merged, _ = nearest_rung(self._rungs[key], max(1, round(cand.rho * k_realized)))
        return k_merged / k_realized

    @torch.no_grad()
    def apply(self, cand, huffman=False):
        """Write the compressed weights for cand into the model. Returns size stats."""
        if self.method == "ubq-ecsq":
            return self._apply_merged(cand, huffman)
        bits, zeros, n_pruned, huff_bits = 0.0, 0, 0, 0
        n_format = {f: 0 for f in FORMATS}
        k_used, row_levels, n_rows = [], 0, 0
        levels_used = torch.zeros(cand.k, dtype=torch.bool)
        for (_, m), orig, row_std in zip(self.targets, self.originals, self.row_stds):
            w0 = orig.to(m.weight.device, non_blocking=True)
            q, scale, pruned = compress_weight(w0, row_std, cand, self.prune)
            levels_used |= (level_counts(q, cand.k) > 0).cpu()
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
            "k_realized": int(levels_used.sum()),       # levels used anywhere in the model
            "k_merged": int(levels_used.sum()),         # no merging in plain UBQ
        }
        if huffman:
            stats["huffman_size_mb"] = (huff_bits + 16 * self.rest_params) / 8 / 1e6
            stats["huffman_bits_per_weight"] = huff_bits / n
        return stats

    @torch.no_grad()
    def _apply_merged(self, cand, huffman=False):
        """UBQ-ECSQ: prune, clip, quantize to the K-level grid, then replace each level by its merged
        group's global centroid (see the module docstring)."""
        k_merged, k_realized, (level_code, code_value, offset) = self.merge_plan(cand)
        device = self.targets[0][1].weight.device
        level_code_t = torch.as_tensor(level_code, device=device)
        level_value_t = torch.as_tensor(code_value[level_code + offset], device=device, dtype=torch.float32)
        k_codes = 2 * max(offset, len(code_value) - offset - 1, 1) + 1  # codes fit in [-(k_codes//2), k_codes//2]
        code_cand = Candidate(k_codes, cand.c, cand.alpha, cand.beta)
        table_bits = ceil_log2(k_merged)  # a lookup-table entry points into the global codebook
        bits, zeros, n_pruned, huff_bits = 16.0 * k_merged, 0, 0, 16 * k_merged  # the codebook, once
        n_format = {f: 0 for f in FORMATS}
        k_used, row_levels, n_rows = [], 0, 0
        for (_, m), orig, row_std in zip(self.targets, self.originals, self.row_stds):
            w0 = orig.to(m.weight.device, non_blocking=True)
            q, scale, pruned = compress_weight(w0, row_std, cand, self.prune)
            idx = (q + cand.k // 2).long()
            codes = level_code_t[idx].float()
            counts = level_counts(codes, k_codes)
            matrix_k_used = int((counts > 0).sum())
            side_bits = 16 * q.shape[0] + matrix_k_used * table_bits
            if self.size_mode == "fixed":
                fbits = format_bits(codes, code_cand, counts)
                fmt = min(self.formats, key=fbits.get)
                bits += fbits[fmt] + side_bits
                n_format[fmt] += 1
            else:
                p = counts[counts > 0].double() / q.numel()
                bits += float(-(p * torch.log2(p)).sum()) * q.numel() + side_bits
            k_used.append(matrix_k_used)
            used = torch.zeros(q.shape[0], k_codes, dtype=torch.bool, device=q.device)
            row_levels += int(used.scatter_(1, (codes + k_codes // 2).long(), True).sum())
            n_rows += q.shape[0]
            zeros += q.numel() - int(torch.count_nonzero(codes))
            n_pruned += int(pruned.sum())
            if huffman:
                huff_bits += huffman_bits(counts.tolist()) + side_bits + 8 * matrix_k_used
            m.weight.copy_((level_value_t[idx] * scale).to(m.weight.dtype))
        total_bits = bits + 16 * self.rest_params
        n = self.n_target_params
        stats = {
            "size_mb": total_bits / 8 / 1e6,
            "size_ratio": total_bits / self.baseline_bits,
            "target_bits_per_weight": bits / n,
            "n_pruned": n_pruned,
            "n_rounded_zero": zeros - n_pruned,
            "n_nonzero": n - zeros,
            "pruned_pct": 100 * n_pruned / n,
            "sparsity_pct": 100 * zeros / n,
            **{f"n_{f}_matrices": c for f, c in n_format.items()},
            "k_used_mean": sum(k_used) / len(k_used),
            "k_used_min": min(k_used),
            "k_used_max": max(k_used),
            "k_used_row_mean": row_levels / n_rows,
            "k_realized": k_realized,  # non-empty grid levels before merging
            "k_merged": k_merged,      # after merging (the global codebook size)
        }
        if huffman:
            stats["huffman_size_mb"] = (huff_bits + 16 * self.rest_params) / 8 / 1e6
            stats["huffman_bits_per_weight"] = huff_bits / n
        return stats

    @torch.no_grad()
    def restore(self):
        for (_, m), orig in zip(self.targets, self.originals):
            m.weight.copy_(orig.to(m.weight.device, non_blocking=True))
