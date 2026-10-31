"""
Weight sharing + pruning and hardware-aware formats in one search (scripts/compress_search_mix.py).

Stacking them does not work: UBQ-ECSQ codes pushed through the int4 grid lose their sharing and add loss,
and pruning on top of int4 adds loss without shrinking the codes. So a matrix group is in ONE format:
    bf16, int4_g32/g64/g128   as in compress_hw: the kernels of the deployed model
    shared                    UBQ-ECSQ (compress.py) with one codebook for all shared matrices. No GPU kernel
                              reads these codes, so a shared matrix runs and sits in GPU memory as bf16: sharing
                              buys stored size only, and costs accuracy, never time
The order within a configuration is fixed: neurons and sublayers are removed first, each kept matrix is then
put in its format, and the codes are entropy coded for storage last.

Besides the genes of compress_hw, 4 genes hold the sharing candidate: grid levels k, clipping c, the symmetric
pruning interval beta and the merge rho. They mean nothing when no group is shared: decode() then zeroes them.

Time is the workload of an agent, not one decode step: the GPU time of a turn
    new_tokens * prompt_us + gen_tokens * decode_us
from two latency tables, since the best format depends on the batch (int4 wins at small batches and loses by a
wide margin on prompts). Prompt tokens read one shared cache together (time_prompt_attention), each decoding
sequence reads its own, and only decoding runs lm_head on every token.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import torch

from nanochat.compress import Candidate, ceil_log2, compress_weight, ecsq_ladder, nearest_rung
from nanochat.compress_hw import (ROLES, Config, HWCompressor, build_latency_table, graph_time_us, shape_key)

SHARED = "shared"
SHARE_K = (17, 33, 65, 129, 257)
SHARE_C = (1.0, 0.9, 0.8, 0.7)
SHARE_PRUNE = (0.0, 0.1, 0.2, 0.3, 0.5)  # weights within +-beta row stds become zero
SHARE_RHO = (1.0, 0.75, 0.5, 0.25)
SHARE_GENES = ("share_k", "share_c", "share_prune", "share_rho")


def entropy_bits(counts):
    """Bits of the symbols with these frequencies under an ideal entropy coder."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    return float(-(counts * np.log2(counts / counts.sum())).sum())


@torch.no_grad()
def time_prompt_attention(chunk, n_head, n_kv_head, head_dim, context, device):
    """GPU time in us per token, one layer, of attention during prompt processing: all tokens of a chunk read
    the same cache together. The attention weights of a chunk take n_head * chunk * context floats."""
    q = torch.randn(1, n_kv_head, n_head // n_kv_head * chunk, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, n_kv_head, context, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, n_kv_head, context, head_dim, device=device, dtype=torch.bfloat16)

    def call():
        a = torch.softmax((q @ k.transpose(-1, -2)).float() * head_dim ** -0.5, dim=-1).to(q.dtype)
        return a @ v

    return graph_time_us([call]) / chunk


@torch.no_grad()
def build_prompt_table(model, layout, formats, mlp_keep, batch, context, log=print):
    """Kernel times of `batch` prompt tokens processed together, in the layout of build_latency_table. Its
    attention over a context of `batch` separate caches would not even fit: that entry is the shared-cache time."""
    table = build_latency_table(model, layout, formats, mlp_keep, batch, 1, log=log)
    us = time_prompt_attention(batch, layout.n_head, layout.n_kv_head, layout.head_dim, context, next(model.parameters()).device)
    log(f"  prompt attention per layer at context {context}: {us:.1f} us per token")
    table.update(attn_us=round(us * batch, 3), attn_impl="shared cache", context=context)
    return table


@dataclass(frozen=True)
class MixConfig(Config):
    share: Candidate = None


class MixCompressor(HWCompressor):
    """HWCompressor with the `shared` format, the cost of an agent's turn and the stored size. table is the
    decode table (its batch: the agents served together), prompt_table the one from build_prompt_table."""

    def __init__(self, model, layout, table, prompt_table, act_sq, removal_order, new_tokens=2000, gen_tokens=300,
                 vram_gb=80.0, reserve=0.1, **kw):
        super().__init__(model, layout, table, act_sq, removal_order, **kw)
        self.prompt_table = prompt_table
        self.new_tokens, self.gen_tokens = new_tokens, gen_tokens
        self.vram_mb, self.reserve = 1e3 * vram_gb, reserve
        self.formats = (*self.formats, SHARED)
        self.genes = [(n, len(self.formats) if not n.startswith(("keep_", "skip_")) else k) for n, k in self.genes]
        self.genes += [(n, len(o)) for n, o in zip(SHARE_GENES, (SHARE_K, SHARE_C, SHARE_PRUNE, SHARE_RHO))]
        self._format_genes = [i for i, (n, _) in enumerate(self.genes) if not n.startswith(("keep_", "skip_", "share_"))]
        self._share_genes = [i for i, (n, _) in enumerate(self.genes) if n.startswith("share_")]
        self._level_stats, self._ladders = {}, OrderedDict()
        self._codebook, self._bits, self._shared_params = None, 0.0, 0

    # --- genes <-> configurations

    def decode(self, x):
        """Integer genes; the sharing genes are zeroed when no group is shared, so those configurations are
        recognized as the same model."""
        genes = list(super().decode(x))
        shared = self.formats.index(SHARED)
        if not any(genes[i] == shared for i in self._format_genes):
            for i in self._share_genes:
                genes[i] = 0
        return tuple(genes)

    def config(self, genes):
        g = dict(zip((name for name, _ in self.genes), genes))
        beta = SHARE_PRUNE[g["share_prune"]]
        share = Candidate(SHARE_K[g["share_k"]], SHARE_C[g["share_c"]], -beta, beta, SHARE_RHO[g["share_rho"]])
        base = super().config(genes)
        return MixConfig(base.fmt, base.lm_head, base.embed, base.keep, base.skip_attn, base.skip_mlp, share)

    def uniform_genes(self, fmt):
        """All matrices in fmt, nothing pruned or removed, the mildest sharing."""
        f = self.formats.index(fmt)
        return tuple(0 if name.startswith(("keep_", "skip_", "share_")) else f for name, _ in self.genes)

    def describe(self, cfg):
        short = {"bf16": "B", "int4_g32": "3", "int4_g64": "6", "int4_g128": "8", SHARED: "S"}
        fmt = " ".join(f"{role}:{''.join(short[f] for f in cfg.fmt[r])}" for r, role in enumerate(ROLES))
        s = cfg.share
        return (f"{fmt} head:{short[cfg.lm_head]}" + (f" emb:{short[cfg.embed]}" if self.compress_embeddings else "")
                + f" keep:{'/'.join(f'{k:g}' for k in cfg.keep)} -attn {cfg.skip_attn} -mlp {cfg.skip_mlp}"
                + (f" share:k{s.k} c{s.c:g} p{s.beta:g} rho{s.rho:g}" if self.shared_matrices(cfg) else ""))

    def deployed(self, cfg):
        """cfg as the GPU runs it: a shared matrix is expanded to bf16."""
        run = lambda f: "bf16" if f == SHARED else f
        return replace(cfg, fmt=tuple(tuple(map(run, band)) for band in cfg.fmt), lm_head=run(cfg.lm_head), embed=run(cfg.embed))

    # --- weight sharing

    def shared_matrices(self, cfg):
        """[(name, kept rows, kept cols)] of the matrices cfg keeps and shares."""
        L, out = self.layout, []
        attn_off, mlp_off = self.removed(cfg)
        for i, blk in enumerate(L.blocks):
            b = self.band[i]
            if i not in attn_off:
                out += [(n, None, None) for r, role in ((0, "qkv"), (1, "o")) if cfg.fmt[r][b] == SHARED for n in blk[role]]
            if i not in mlp_off:
                kept = self.kept(i, cfg.keep[b])
                if cfg.fmt[2][b] == SHARED:
                    out += [(n, kept, None) for n in blk["up"]]
                if cfg.fmt[3][b] == SHARED:
                    out.append((blk["down"][0], None, kept))
        if cfg.lm_head == SHARED:
            out.append((L.lm_head, None, None))
        if self.compress_embeddings and cfg.embed == SHARED:
            out += [(n, None, None) for n in L.embeds]
        return out

    def _quantized(self, name, rows, cols, cand):
        """The grid of the kept part of the original weight: (kept weight, q, per-row scale)."""
        w = self._orig(name)
        if rows is not None:
            w = w[rows]
        if cols is not None:
            w = w[:, cols]
        row_std = w.float().std(dim=1, keepdim=True)
        return (w, *compress_weight(w, row_std, cand)[:2])

    def _matrix_levels(self, name, rows, cols, cand):
        """Per grid level of one kept matrix: (counts, sums of w/scale), cached: the search revisits them."""
        size = (None if rows is None else len(rows), None if cols is None else len(cols))
        key = (name, size, cand.k, cand.c, cand.beta)
        if key not in self._level_stats:
            w, q, scale = self._quantized(name, rows, cols, cand)
            idx = (q + cand.k // 2).long().flatten()
            self._level_stats[key] = (torch.bincount(idx, minlength=cand.k).double().cpu().numpy(),
                                      torch.bincount(idx, weights=(w.float() / scale).flatten().double(), minlength=cand.k).cpu().numpy())
        return self._level_stats[key]

    def _make_codebook(self, cfg):
        """The merge of the shared levels over exactly the matrices cfg shares: the codebook, or None."""
        matrices = self.shared_matrices(cfg)
        if not matrices:
            return None
        cand = cfg.share
        stats = [self._matrix_levels(n, r, c, cand) for n, r, c in matrices]
        counts, sums = sum(s[0] for s in stats), sum(s[1] for s in stats)
        key = (cand.k, cand.c, cand.beta, frozenset((n, None if r is None else len(r), None if c is None else len(c)) for n, r, c in matrices))
        if key not in self._ladders:
            self._ladders[key] = ecsq_ladder(counts, sums, cand.k, n_lambdas=48)
            if len(self._ladders) > 32:
                self._ladders.popitem(last=False)
        self._ladders.move_to_end(key)
        k_merged, (level_code, code_value, offset) = nearest_rung(self._ladders[key], max(1, round(cand.rho * int((counts > 0).sum()))))
        return {"cand": cand, "k_merged": k_merged, "code_of_level": level_code + offset,
                "value_of_level": torch.as_tensor(code_value[level_code + offset], device=self.device, dtype=torch.float32)}

    def _shared_weight(self, name, rows, cols):
        book = self._codebook
        cand = book["cand"]
        w, q, scale = self._quantized(name, rows, cols, cand)
        part = book["value_of_level"][(q + cand.k // 2).long()] * scale
        counts = np.bincount(book["code_of_level"], weights=self._matrix_levels(name, rows, cols, cand)[0])
        self._bits += entropy_bits(counts) + 16 * w.shape[0] + int((counts > 0).sum()) * ceil_log2(book["k_merged"])
        self._shared_params += w.numel()
        if rows is None and cols is None:
            return part
        out = torch.zeros(self._orig(name).shape, dtype=part.dtype, device=part.device)
        if rows is not None:
            out[rows] = part
        else:
            out[:, cols] = part
        return out

    # --- writing a configuration; the stored size is added up on the way

    def int4(self, name, fmt, rows=None, cols=None):
        q, scale, zero = super().int4(name, fmt, rows, cols)
        self._bits += entropy_bits(torch.bincount(q.flatten(), minlength=16).cpu().numpy()) + 32 * scale.numel()
        return q, scale, zero

    def weight(self, name, fmt, rows=None, cols=None):
        n_rows, n_cols, bias = self._shape(name, rows, cols)
        self._bits += 16 * n_rows if bias else 0
        if fmt == SHARED:
            return self._shared_weight(name, rows, cols)
        if self.effective(name, fmt, rows, cols) == "bf16":
            self._bits += 16 * n_rows * n_cols
        return super().weight(name, fmt, rows, cols)

    @torch.no_grad()
    def apply(self, cfg):
        """Write cfg's weights into the model (undo with restore()). Returns its stored size."""
        self._bits, self._shared_params = 16.0 * self.rest_params, 0
        self._codebook = self._make_codebook(cfg)
        if self._codebook is not None:
            self._bits += 16 * self._codebook["k_merged"]
        super().apply(cfg)
        return {"size_mb": self._bits / 8e6, "shared_pct": 100 * self._shared_params / self.n_target_params,
                "k_merged": 0 if self._codebook is None else self._codebook["k_merged"]}

    @torch.no_grad()
    def materialize(self, cfg):
        """Rebuild the model as deployed (see HWCompressor.materialize): shared matrices in bf16, with the
        codebook's values. Cannot be undone."""
        self._codebook = self._make_codebook(cfg)
        for name, rows, cols in self.shared_matrices(cfg):
            w = self._shared_weight(name, rows, cols)
            self.originals[name] = w.to(self.originals[name].device, dtype=self.originals[name].dtype)
            self.layout.module(name).weight.copy_(w.to(self.layout.module(name).weight.dtype))
        super().materialize(self.deployed(cfg))

    # --- cost model

    def cost(self, cfg):
        """GPU time of an agent's turn (latency_ms) and its parts, deployed memory with the KV cache of the
        agents served together, and the agents of the same context that fit on the GPU."""
        run = self.deployed(cfg)
        decode_table = self.table
        decode = super().cost(run)
        self.table = self.prompt_table
        try:
            prompt = super().cost(run)
            head = self.prompt_table["gemm_us"][shape_key(*self.shapes[self.layout.lm_head])][self.effective(self.layout.lm_head, run.lm_head)]
        finally:
            self.table = decode_table
        decode_us = 1e3 * decode["latency_ms"] / decode_table["batch"]
        prompt_us = (1e3 * prompt["latency_ms"] - head) / self.prompt_table["batch"]  # only the last token needs logits
        kv_per_agent = decode["kv_mb"] / decode_table["batch"]
        room = self.vram_mb * (1 - self.reserve) - decode["weights_mb"]
        return {**decode, "latency_ms": (self.new_tokens * prompt_us + self.gen_tokens * decode_us) / 1e3,
                "decode_us": decode_us, "prompt_us": prompt_us,
                "agents_fit": math.floor(max(0.0, room) / kv_per_agent) if kv_per_agent > 0 else math.inf}
