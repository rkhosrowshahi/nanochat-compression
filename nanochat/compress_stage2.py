"""
Stage 2 of the two-stage search (scripts/compress_search_stage2.py): weight sharing + pruning (UBQ-ECSQ, compress.py) on the
Pareto front of a finished hardware-aware search (stage 1: scripts/compress_search_tpot.py, or compress_search_hw.py).

Stage 1 chose the deployed kernel of every matrix (bf16 or int4 tinygemm), which fixes the decode TPOT and the GPU memory.
Stage 2 keeps them: each solution is a stage-1 solution (the `base` gene) with some of its bf16 matrices STORED as UBQ-ECSQ
codes on one shared codebook. A shared matrix still runs as bf16 (no kernel reads the codes), holding the codebook's values,
so it changes the quality and the stored size, never the time or the memory. int4 matrices are left as they are: sharing
pushed through the int4 grid loses its sharing and adds loss (measured on Qwen2.5-0.5B, 2026-10-30). Their codes are stored
entropy coded, as every code here.

Genes (integers):
    base                     index into the stage-1 front, sorted by TPOT (neighbours are similar solutions)
    share_<role>_b<g>        0/1: share the bf16 matrices of this role (qkv, o, up, down) in share band g (the layers split
                             into --share-bands equal runs). A matrix that the base deploys in int4 is never shared
    share_head               0/1: share the output head when the base keeps it in bf16 (with a tied embedding: the one matrix)
    share_embed              0/1 (untied embeddings that stage 1 searched, e.g. nanochat): share them when the base keeps them
                             in bf16
    share_k, share_c,        the UBQ-ECSQ candidate, one for all shared matrices (one global codebook, as in compress.py):
    share_prune, share_rho   grid levels, clipping, the symmetric pruning interval (+-beta row stds -> 0) and the merge
Genes that change nothing are zeroed by decode(): sharing genes of groups without bf16 matrices, and the candidate when
nothing is shared. So equal models have equal keys.

Tied embedding with the head in int4: the deployed model (HWCompressor.materialize) looks up a bf16 table that holds the int4
head's dequantized values, so it is rebuilt from the int4 codes when the model is loaded: it takes GPU memory (stage 1 counts it)
but no stored bits, and there is nothing to share in it. With nothing shared a configuration is exactly its base, so its KLD is
the stage-1 value (logged as base_quality).

Stored size (size_mb, MB = 1e6 bytes) under an ideal entropy coder: int4 codes at their empirical entropy plus a bf16 scale and
zero per group; shared matrices at the entropy of their codes plus a 16-bit scale per row, a lookup table per matrix and the
fp16 codebook once; other bf16 matrices and the remaining parameters (norms, biases) at 16 bits.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from nanochat.compress import Candidate
from nanochat.compress_hw import FORMATS, MLP_KEEP, ROLES, Config
from nanochat.compress_mix import SHARED, MixCompressor, MixConfig

SHARE_K = (9, 17, 33, 65, 129, 257)
SHARE_C = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7)
SHARE_PRUNE = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
SHARE_RHO = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15)
CANDIDATE_GENES = ("share_k", "share_c", "share_prune", "share_rho")
SHORT = {"bf16": "B", "int4_g32": "3", "int4_g64": "6", "int4_g128": "8", SHARED: "S"}


# -----------------------------------------------------------------------------
# stage 1: its genes and its front

def stage1_formats(run_args):
    """The format options of a stage-1 run, in gene order (HWCompressor keeps FORMATS' order)."""
    chosen = [f for f in run_args["formats"].split(",") if f]
    return tuple(f for f in FORMATS if f in chosen)


def stage1_mlp_keep(run_args):
    return tuple(sorted({float(k) for k in str(run_args.get("mlp_keep", ",".join(map(str, MLP_KEEP)))).split(",")} | {1.0},
                        reverse=True))


def stage1_config(genes, formats, mlp_keep, bands, compress_embeddings):
    """Config of a stage-1 gene dict {name: int}, as HWCompressor.config (and TPOTCompressor's, which leaves the genes with one
    option out: a missing gene takes option 0)."""
    g = lambda name: int(genes.get(name, 0))
    fmt = tuple(tuple(formats[g(f"{role}_b{b}")] for b in range(bands)) for role in ROLES)
    return Config(fmt, formats[g("lm_head")], formats[g("embed")] if compress_embeddings else "bf16",
                  tuple(mlp_keep[g(f"keep_b{b}")] for b in range(bands)), g("skip_attn"), g("skip_mlp"))


def per_layer(cfg, n_layer, bands):
    """cfg of `bands` depth bands as a configuration with one band per layer (the same model)."""
    band = [i * bands // n_layer for i in range(n_layer)]
    return replace(cfg, fmt=tuple(tuple(cfg.fmt[r][band[i]] for i in range(n_layer)) for r in range(len(ROLES))),
                   keep=tuple(cfg.keep[band[i]] for i in range(n_layer)))


def pareto_front(rows, objectives):
    """The rows that no other row dominates on the objective columns (all minimized), duplicates (same values) once."""
    pts = np.array([[r[o] for o in objectives] for r in rows], dtype=float)
    keep = []
    for i, p in enumerate(pts):
        dominated = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        same_earlier = any(np.array_equal(pts[j], p) for j in keep)
        if not dominated and not same_earlier:
            keep.append(i)
    return [rows[i] for i in keep]


def select_bases(rows, quality, objectives, max_quality=None, max_bases=None, is_uncompressed=None):
    """The stage-1 solutions stage 2 starts from: the front of all evaluations on (quality, *objectives) with quality up to
    max_quality, sorted by the first objective, at most max_bases of them spread evenly along it. The uncompressed model
    (is_uncompressed(row)) is always kept: it is the baseline, and sharing it is plain UBQ-ECSQ."""
    ok = [r for r in rows if math.isfinite(r[quality]) and (max_quality is None or r[quality] <= max_quality)]
    front = sorted(pareto_front(ok, [quality, *objectives]), key=lambda r: (r[objectives[0]], r[quality]))
    if max_bases is not None and len(front) > max_bases:
        idx = set(np.linspace(0, len(front) - 1, max_bases).round().astype(int).tolist())
        front = [r for i, r in enumerate(front) if i in idx or (is_uncompressed and is_uncompressed(r))]
    if is_uncompressed is not None:
        base = [r for r in rows if is_uncompressed(r)]
        assert base, "the stage-1 evaluations do not contain the uncompressed model (its first seed)"
        if not any(is_uncompressed(r) for r in front):
            front = sorted(front + base[:1], key=lambda r: (r[objectives[0]], r[quality]))
    return front


# -----------------------------------------------------------------------------
# the compressor

@dataclass(frozen=True)
class Stage2Config(MixConfig):
    base: int = 0


class _SharingEngine(MixCompressor):
    """MixCompressor with one band per layer and the stored bits per weight."""

    @torch.no_grad()
    def apply(self, cfg):
        stats = super().apply(cfg)
        return {**stats, "stored_bits_per_weight": (self._bits - 16 * self.rest_params) / self.n_target_params}

    def materialize(self, cfg):
        raise NotImplementedError("stage 2 deploys the stage-1 model with the shared values: materialize the base instead")


class Stage2Compressor:
    """Writes stage-2 configurations into the model (apply/restore), with their stored size; their cost is their base's
    (stage-1 row). The interface of the compressors of scripts/compress_search_common.Search.

    bases: the stage-1 rows (dicts with the stage-1 gene names, latency_ms, memory_mb and the quality), in base-gene order.
    run_args: the stage-1 run's arguments (config.json "args"); table: its latency table (which kernels run which shapes)."""

    def __init__(self, model, layout, table, act_sq, removal_order, run_args, bases, quality="kld", share_bands=4,
                 share_k=SHARE_K, share_c=SHARE_C, share_prune=SHARE_PRUNE, share_rho=SHARE_RHO, offload="auto"):
        self.layout, self.bases, self.quality = layout, list(bases), quality
        self.options = {"share_k": tuple(share_k), "share_c": tuple(share_c), "share_prune": tuple(share_prune),
                        "share_rho": tuple(share_rho)}
        formats, mlp_keep = stage1_formats(run_args), stage1_mlp_keep(run_args)
        bands1 = min(int(run_args["bands"]), layout.n_layer)
        compress_embeddings = not run_args.get("no_embeddings", False) and bool(layout.embeds)
        n = layout.n_layer
        self.engine = _SharingEngine(model, layout, table, table, act_sq, removal_order, sharing=False, formats=formats,
                                     bands=n, mlp_keep=mlp_keep, max_skip=int(run_args.get("max_skip", 0)),
                                     compress_embeddings=compress_embeddings,
                                     clip_ratios=[float(r) for r in run_args["clip_ratios"].split(",")], offload=offload)
        e = self.engine
        self.base_cfgs = [per_layer(stage1_config(r, formats, mlp_keep, bands1, compress_embeddings), n, bands1) for r in self.bases]
        self.share_bands = min(share_bands, n)
        self.share_band = [i * self.share_bands // n for i in range(n)]
        has_embed_gene = e.compress_embeddings and any(c.embed == "bf16" for c in self.base_cfgs)
        self.genes = [("base", len(self.bases))]
        self.genes += [(f"share_{role}_b{g}", 2) for role in ROLES for g in range(self.share_bands)]
        self.genes.append(("share_head", 2))
        if has_embed_gene:
            self.genes.append(("share_embed", 2))
        self.genes += [(name, len(self.options[name])) for name in CANDIDATE_GENES]
        self.names = [name for name, _ in self.genes]
        uncompressed = [i for i, c in enumerate(self.base_cfgs) if self._all_bf16(c)]
        assert uncompressed, "the bases do not contain the uncompressed model"
        self.uncompressed_base = uncompressed[0]
        # Search's interface
        self.formats, self.offloaded = ("bf16",), e.offloaded
        self.n_target_params, self.rest_params = e.n_target_params, e.rest_params

    @staticmethod
    def _all_bf16(cfg):
        return (all(f == "bf16" for band in cfg.fmt for f in band) and cfg.lm_head == "bf16" and cfg.embed == "bf16"
                and all(k == 1.0 for k in cfg.keep) and cfg.skip_attn == cfg.skip_mlp == 0)

    # --- which matrices a base lets stage 2 share

    def shareable(self, base):
        """{gene name: True} for the sharing genes that change something on this base."""
        cfg, e, out = self.base_cfgs[base], self.engine, {}
        attn_off, mlp_off = e.removed(cfg)
        for r, role in enumerate(ROLES):
            for i in range(self.layout.n_layer):
                off = i in (attn_off if r < 2 else mlp_off)
                if not off and cfg.fmt[r][i] == "bf16":
                    out[f"share_{role}_b{self.share_band[i]}"] = True
        if cfg.lm_head == "bf16":
            out["share_head"] = True
        if "share_embed" in self.names and cfg.embed == "bf16":
            out["share_embed"] = True
        return out

    # --- genes <-> configurations

    def decode(self, x):
        g = [int(min(max(round(float(v)), 0), n - 1)) for v, (_, n) in zip(x, self.genes)]
        ok = self.shareable(g[0])
        any_shared = False
        for j, name in enumerate(self.names):
            if name.startswith("share_") and name not in CANDIDATE_GENES:
                if not ok.get(name):
                    g[j] = 0
                any_shared |= g[j] == 1
        if not any_shared:
            for name in CANDIDATE_GENES:
                g[self.names.index(name)] = 0
        return tuple(g)

    def config(self, genes):
        g = dict(zip(self.names, genes))
        base = self.base_cfgs[g["base"]]
        share = lambda f, on: SHARED if on and f == "bf16" else f
        fmt = tuple(tuple(share(base.fmt[r][i], g[f"share_{role}_b{self.share_band[i]}"]) for i in range(self.layout.n_layer))
                    for r, role in enumerate(ROLES))
        o = self.options
        beta = o["share_prune"][g["share_prune"]]
        cand = Candidate(o["share_k"][g["share_k"]], o["share_c"][g["share_c"]], -beta, beta, o["share_rho"][g["share_rho"]])
        return Stage2Config(fmt, share(base.lm_head, g["share_head"]), share(base.embed, g.get("share_embed", 0)),
                            base.keep, base.skip_attn, base.skip_mlp, cand, g["base"])

    def uniform_genes(self, fmt):
        """bf16: the uncompressed model, nothing shared (the baseline). Other formats are stage 1's business."""
        assert fmt == "bf16", "stage 2 has one uniform configuration: the uncompressed model"
        return self.decode([self.uncompressed_base] + [0] * (len(self.genes) - 1))

    def no_sharing_genes(self, base):
        return self.decode([base] + [0] * (len(self.genes) - 1))

    def all_shared_genes(self, base, **candidate):
        """Every bf16 matrix of the base shared, with the candidate genes given by name (default: the mildest)."""
        g = [base] + [1 if n.startswith("share_") and n not in CANDIDATE_GENES else 0 for n in self.names[1:]]
        for name, v in candidate.items():
            g[self.names.index(name)] = v
        return self.decode(g)

    def shared_any(self, cfg):
        return bool(self.engine.shared_matrices(cfg))

    def describe(self, cfg):
        fmt = " ".join(f"{role}:{''.join(SHORT[f] for f in cfg.fmt[r])}" for r, role in enumerate(ROLES))
        text = f"base {cfg.base}: {fmt} head:{SHORT[cfg.lm_head]}"
        if self.engine.compress_embeddings:
            text += f" emb:{SHORT[cfg.embed]}"
        if any(k < 1 for k in cfg.keep) or cfg.skip_attn or cfg.skip_mlp:
            text += f" keep:{'/'.join(f'{k:g}' for k in cfg.keep)} -attn {cfg.skip_attn} -mlp {cfg.skip_mlp}"
        if self.shared_any(cfg):
            s = cfg.share
            text += f" | share k{s.k} c{s.c:g} p{s.beta:g} rho{s.rho:g}"
        return text

    # --- writing a configuration, and its cost

    def apply(self, cfg):
        return self.engine.apply(cfg)

    def restore(self):
        self.engine.restore()

    def cost(self, cfg):
        """The base's stage-1 time and memory: sharing changes neither."""
        r = self.bases[cfg.base]
        latency = float(r["latency_ms"])
        return {"base": cfg.base, "base_quality": float(r[self.quality]), "tpot_ms": float(r.get("tpot_ms", latency)),
                "latency_ms": latency, "tok_s": 1000 / latency, "memory_mb": float(r["memory_mb"]),
                "weights_gib": float(r["memory_mb"]) * 1e6 / (1 << 30),
                "deployed_bits_per_weight": float(r.get("bits_per_weight", math.nan)),
                "int4_pct": float(r.get("int4_pct", math.nan))}
