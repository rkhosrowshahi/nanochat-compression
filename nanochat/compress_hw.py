"""
Hardware-aware compression of a causal LM, for a search that trades accuracy against the decode
latency and the memory of the DEPLOYED model (scripts/compress_search_hw.py), not an idealized storage
size: every configuration maps onto kernels that run on the target GPU, its latency is predicted
from a table of those kernels' times measured on that GPU, and the same configuration can be built
for real (materialize) to check the prediction.

Model structure (HF Llama/Qwen2-style models, nanochat GPT): n_layer blocks, each with
    attention  qkv: q, k, v (one format: they are fused in deployment)    o: output projection
    MLP        up: gate + up (SwiGLU) or c_fc                              down: down_proj or c_proj
plus the output head (lm_head) and the input embeddings (tied to lm_head in Qwen2.5: one matrix).

A Config holds the decision variables:
    fmt       format of each role in each depth band (the layers split into `bands` equal runs)
    lm_head   format of the output head (a full matrix-vector product every token)
    embed     format of the untied embeddings: memory only, a lookup reads one row per token
    keep      fraction of MLP neurons kept in each band, rounded to multiples of 128 neurons; the
              neurons kept are those with the largest ||W_down[:, j]|| * sqrt(E[h_j^2]) (Wanda-style)
    skip_attn, skip_mlp  number of attention / MLP sublayers removed, in the order of a one-time
              ranking (removal_ranking): the sublayer whose removal alone raises the loss least first
Formats: bf16, and int4 with asymmetric per-group scales in the layout of PyTorch's built-in tinygemm
kernel (torch.ops.aten._weight_int4pack_mm, CUDA sm80+): w = (q - 8) * scale + zero, q in 0..15,
scale and zero in bf16, one pair per group of 32/64/128 consecutive input weights. The clipping of
each group (its min and max shrunk by r) minimizes the activation-weighted error
sum_j E[x_j^2] (w_j - w^_j)^2: it changes accuracy, not cost, so it is solved per group, not searched.
A format a kernel cannot run for some shape (see the latency table) falls back to bf16 there.

apply()/restore() write the deployed weights into the model in place (dequantized int4, zeroed
pruned neurons, zeroed output projections of removed sublayers), so the loss is the deployed
model's: the int4 kernel matches the dequantized bf16 weights to bf16 rounding. materialize()
builds the deployed model itself.

Latency (build_latency_table): the GPU time of one decode step for `batch` sequences, as a runtime
without launch overhead (CUDA graphs) would see it:
    fixed + sum over kept attention sublayers (misc_attn + attention over the context + q,k,v,o)
          + sum over kept MLP sublayers (misc_mlp + up, down at the kept width) + lm_head
Each matrix product is timed alone for its shape and format, inside a CUDA graph and cycling
through copies of the weight larger than the L2 cache (as in the model, where every layer reads
its own weights from memory). Attention over the context is the faster of PyTorch's SDPA and a
plain grouped-query matmul (builds without FlashAttention, e.g. on Windows, run SDPA's slow math
path for grouped-query attention). misc_attn, misc_mlp (norms, rotary, activation, residual adds)
and fixed (embedding, final norm, logits) come from the real model's summed kernel times
(profiler) for one token, with all attention or all MLP sublayers removed and without.
"""

import contextlib
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.compress_eval import eval_loss

FORMATS = ("bf16", "int4_g32", "int4_g64", "int4_g128")  # most accurate (and largest) first
ROLES = ("qkv", "o", "up", "down")
MLP_KEEP = (1.0, 0.875, 0.75, 0.625, 0.5)
CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
NEURON_MULTIPLE = 128  # kept MLP widths are multiples of this (tensor-core tiles, int4 groups)
TABLE_VERSION = 2  # of build_latency_table's output: cached tables of another version are measured again


def group_size(fmt):
    return None if fmt == "bf16" else int(fmt.split("_g")[1])


def weight_bytes(rows, cols, fmt):
    """Deployed bytes of a rows x cols matrix: bf16, or 4-bit codes plus a bf16 scale and zero per group."""
    if fmt == "bf16":
        return 2 * rows * cols
    return rows * cols // 2 + 4 * rows * (cols // group_size(fmt))


def n_kept(keep, width):
    """MLP neurons kept for the fraction `keep` of `width`: a multiple of NEURON_MULTIPLE (all if keep >= 1)."""
    if keep >= 1:
        return width
    return min(width, max(NEURON_MULTIPLE, round(keep * width / NEURON_MULTIPLE) * NEURON_MULTIPLE))


def shape_key(rows, cols, bias):
    return f"{rows}x{cols}" + ("+bias" if bias else "")


# -----------------------------------------------------------------------------
# int4 in the layout of PyTorch's tinygemm kernel

@torch.no_grad()
def int4_quantize(w, group, act_sq=None, ratios=CLIP_RATIOS, ratio_idx=None, chunk=1 << 24):
    """Asymmetric int4 of w [N, K] in groups of `group` consecutive columns: w^ = (q - 8) * scale + zero.
    Each group's range [min, max] is shrunk by the ratio from `ratios` that minimizes
    sum_j act_sq_j (w_j - w^_j)^2 (the plain squared error without act_sq), unless ratio_idx gives the
    choice. Returns q (uint8 [N, K], 0..15), scale and zero (bf16 [N, K/group]), ratio_idx (uint8 [N, K/group])."""
    N, K = w.shape
    assert K % group == 0, f"{K} columns are not a multiple of the int4 group size {group}"
    G = K // group
    r_all = torch.tensor(ratios, device=w.device)
    weight = None if act_sq is None else act_sq.to(w.device).float().view(1, G, group)
    qs, scales, zeros, idxs = [], [], [], []
    step = max(1, chunk // K)
    for r0 in range(0, N, step):
        wg = w[r0:r0 + step].float().view(-1, G, group)
        hi, lo = wg.amax(-1), wg.amin(-1)

        def quantize(r):
            scale = ((hi - lo) * r / 15).clamp_min(1e-8).bfloat16().float()
            zero = (lo * r + 8 * scale).bfloat16().float()  # q = 0 -> lo * r, q = 15 -> hi * r
            q = torch.clamp(torch.round((wg - zero[..., None]) / scale[..., None]) + 8, 0, 15)
            return q, scale, zero

        if ratio_idx is None:
            best_err = idx = None
            for i in range(len(ratios)):
                q, scale, zero = quantize(r_all[i])
                d2 = ((q - 8) * scale[..., None] + zero[..., None] - wg).square()
                err = (d2 if weight is None else d2 * weight).sum(-1)
                if best_err is None:
                    best_err, idx = err, torch.zeros_like(err, dtype=torch.uint8)
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    idx[better] = i
        else:
            idx = ratio_idx[r0:r0 + step]
        q, scale, zero = quantize(r_all[idx.long()])
        qs.append(q.view(-1, K).to(torch.uint8))
        scales.append(scale.bfloat16())
        zeros.append(zero.bfloat16())
        idxs.append(idx)
    return torch.cat(qs), torch.cat(scales), torch.cat(zeros), torch.cat(idxs)


def int4_dequantize(q, scale, zero, group):
    N, K = q.shape
    w = (q.float().view(N, K // group, group) - 8) * scale.float()[..., None] + zero.float()[..., None]
    return w.view(N, K)


def int4_pack(q, scale, zero):
    """tinygemm operands: the packed codes and the [K/group, N, 2] scale-and-zero tensor."""
    K = q.shape[1]
    inner_k_tiles = next(t for t in (8, 4, 2) if K % (16 * t) == 0)
    packed = torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).contiguous(), inner_k_tiles)
    return packed, torch.stack([scale, zero], -1).transpose(0, 1).contiguous()


class Int4Linear(nn.Module):
    """A linear layer on PyTorch's int4 tinygemm kernel (bf16 activations)."""

    def __init__(self, q, scale, zero, group, bias=None):
        super().__init__()
        self.out_features, self.in_features = q.shape
        self.group = group
        packed, scale_zero = int4_pack(q, scale, zero)
        self.register_buffer("packed", packed)
        self.register_buffer("scale_zero", scale_zero)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        y = torch.ops.aten._weight_int4pack_mm(x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous(),
                                               self.packed, self.group, self.scale_zero)
        y = y.view(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


class Removed(nn.Module):
    """A removed sublayer: adds nothing to the residual stream (HF attention returns a pair)."""

    def __init__(self, pair=False):
        super().__init__()
        self.pair = pair

    def forward(self, *args, **kwargs):
        z = torch.zeros_like(kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])
        return (z, None) if self.pair else z


# -----------------------------------------------------------------------------
# model structure

class Layout:
    """Where the compressible parts of a model are, by module name: an HF Llama/Qwen2-style model,
    or a nanochat GPT behind compress_models.NanochatLM (both architectures)."""

    def __init__(self, model):
        self.model = model
        gpt = getattr(model, "gpt", None)
        if gpt is not None:
            cfg, prefix, self.attn_attr, self.pair = gpt.config, "gpt.transformer.h", "attn", False
            roles = {"qkv": ("attn.c_q", "attn.c_k", "attn.c_v"), "o": ("attn.c_proj",),
                     "up": ("mlp.c_fc",), "down": ("mlp.c_proj",)}
            n_layer, self.n_head, self.n_kv_head = cfg.n_layer, cfg.n_head, cfg.n_kv_head
            self.head_dim = cfg.n_embd // cfg.n_head
            lm_head = gpt.lm_head
            embeds = [gpt.transformer.wte, *getattr(gpt, "value_embeds", {}).values()]
        else:
            cfg, prefix, self.attn_attr, self.pair = model.config, "model.layers", "self_attn", True
            roles = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
                     "up": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}
            n_layer, self.n_head, self.n_kv_head = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
            self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            lm_head, embeds = model.get_output_embeddings(), [model.get_input_embeddings()]
        self.mlp_attr = "mlp"
        self.layers = [f"{prefix}.{i}" for i in range(n_layer)]
        self.blocks = [{role: [f"{layer}.{n}" for n in names] for role, names in roles.items()} for layer in self.layers]
        name_of = {id(m): n for n, m in model.named_modules()}
        self.lm_head = name_of[id(lm_head)]
        self.tied = any(e.weight is lm_head.weight for e in embeds)
        self.embeds = [name_of[id(e)] for e in embeds if e.weight is not lm_head.weight]
        self.intermediate = self.module(self.blocks[0]["up"][0]).out_features
        for blk in self.blocks:  # removal zeroes these weights: a bias would survive it
            for name in blk["o"] + blk["down"]:
                assert self.module(name).bias is None, f"{name} has a bias: removing its sublayer is not supported"

    @property
    def n_layer(self):
        return len(self.layers)

    def module(self, name):
        return self.model.get_submodule(name)

    def linear_names(self):
        return [n for blk in self.blocks for role in ROLES for n in blk[role]] + [self.lm_head]

    def kv_bytes(self):
        """KV cache bytes per layer per token (keys and values, bf16)."""
        return 2 * self.n_kv_head * self.head_dim * 2

    def linear_shapes(self, mlp_keep):
        """(rows, cols, bias) of every matrix product the deployed model can contain."""
        shapes, widths = set(), {n_kept(k, self.intermediate) for k in (1.0, *mlp_keep)}
        for blk in self.blocks:
            for role in ROLES:
                for name in blk[role]:
                    m = self.module(name)
                    N, K, bias = m.out_features, m.in_features, m.bias is not None
                    if role == "up":
                        shapes |= {(n, K, bias) for n in widths}
                    elif role == "down":
                        shapes |= {(N, n, bias) for n in widths}
                    else:
                        shapes.add((N, K, bias))
        m = self.module(self.lm_head)
        shapes.add((m.out_features, m.in_features, m.bias is not None))
        return sorted(shapes)


@contextlib.contextmanager
def removed_sublayers(layout, attn=(), mlp=()):
    """Temporarily replace the attention / MLP sublayers of these layers by Removed."""
    saved = []
    for layers, attr, pair in ((attn, layout.attn_attr, layout.pair), (mlp, layout.mlp_attr, False)):
        for i in layers:
            parent = layout.module(layout.layers[i])
            saved.append((parent, attr, getattr(parent, attr)))
            setattr(parent, attr, Removed(pair))
    try:
        yield
    finally:
        for parent, attr, old in reversed(saved):
            setattr(parent, attr, old)


# -----------------------------------------------------------------------------
# one-time statistics of the uncompressed model

@torch.no_grad()
def activation_stats(model, layout, tokens, batch_size=1):
    """E[x_j^2] of every input channel of every block linear and of lm_head, over all tokens: {name: [in_features]}."""
    sums, counts, hooks = {}, {}, []

    def hook(name):
        def accumulate(module, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            sums[name] = sums.get(name, 0) + x.square().sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return accumulate

    for name in layout.linear_names():
        hooks.append(layout.module(name).register_forward_pre_hook(hook(name)))
    try:
        eval_loss(model, tokens, batch_size)
    finally:
        for h in hooks:
            h.remove()
    return {name: (sums[name] / counts[name]).cpu() for name in layout.linear_names()}


@torch.no_grad()
def removal_ranking(model, layout, tokens, batch_size=1, log=print):
    """Loss increase on tokens when one sublayer alone is removed (its output projection zeroed), for every
    attention and MLP sublayer. Returns (baseline loss, {"attn": [(layer, delta)], "mlp": [...]}), ascending."""
    base = eval_loss(model, tokens, batch_size)
    out = {"attn": [], "mlp": []}
    for i, blk in enumerate(layout.blocks):
        for kind, role in (("attn", "o"), ("mlp", "down")):
            w = layout.module(blk[role][0]).weight
            saved = w.detach().clone()
            w.zero_()
            try:
                delta = eval_loss(model, tokens, batch_size) - base
            finally:
                w.copy_(saved)
            out[kind].append((i, delta))
        log(f"  layer {i:2d}: removing attention {out['attn'][-1][1]:+.4f}, MLP {out['mlp'][-1][1]:+.4f}")
    return base, {kind: sorted(v, key=lambda t: t[1]) for kind, v in out.items()}


# -----------------------------------------------------------------------------
# latency table

def graph_time_us(fns, replays=15, replay_us=2000):
    """GPU time of one call, cycling through fns inside a CUDA graph (no launch overhead), with enough calls
    per replay (~replay_us of work) that the GPU stays at full clocks. The minimum over replays: other
    work on the GPU (a desktop, another process) can only make a replay slower."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up (cuBLAS handles, workspaces) outside the capture
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(stream)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure(inner, n):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(inner):
                fns[i % len(fns)]()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / inner)
        return min(times)

    rough = measure(max(16, 2 * len(fns)), 3)
    inner = min(2048, max(16, 2 * len(fns), math.ceil(replay_us / max(rough, 1e-3))))
    return measure(inner, replays)


def warm_up(device, seconds=1.0):
    """Busy the GPU until its clocks are up (an idle GPU runs its memory clock far below the maximum)."""
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()


def n_copies(nbytes, device):
    """Copies of an operand to cycle through so that they do not fit in the L2 cache together."""
    l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 4 << 20)
    return min(16, max(1, math.ceil(4 * l2 / nbytes)))


@torch.no_grad()
def time_linear(rows, cols, bias, fmt, batch, device):
    """GPU time (us) of y = x W^T (+ b) for x [batch, cols] with W [rows, cols] in fmt."""
    x = torch.randn(batch, cols, device=device, dtype=torch.bfloat16)
    b = torch.randn(rows, device=device, dtype=torch.bfloat16) if bias else None
    fns = []
    for _ in range(n_copies(weight_bytes(rows, cols, fmt), device)):
        if fmt == "bf16":
            w = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
            fns.append(lambda w=w: F.linear(x, w, b))
        else:
            g = group_size(fmt)
            q = torch.randint(0, 16, (rows, cols), device=device, dtype=torch.uint8)
            scale = torch.rand(rows, cols // g, device=device).bfloat16() * 0.01
            packed, scale_zero = int4_pack(q, scale, torch.zeros_like(scale))
            mm = torch.ops.aten._weight_int4pack_mm
            fns.append(lambda p=packed, sz=scale_zero: mm(x, p, g, sz) if b is None else mm(x, p, g, sz) + b)
    return graph_time_us(fns)


def grouped_attention(q, k, v):
    """Decode attention (q: [B, H, 1, d]) with each query head reading its KV head's cache in place: two
    batched matmuls, no copy of the cache (SDPA's math fallback expands it for grouped-query attention)."""
    B, H, _, d = q.shape
    qg = q.view(B, k.shape[1], H // k.shape[1], d)
    a = torch.softmax((qg @ k.transpose(-1, -2)).float() * d ** -0.5, dim=-1).to(q.dtype)
    return (a @ v).view(B, H, 1, d)


ATTENTION = {"sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=q.shape[1] != k.shape[1]),
             "grouped": grouped_attention}


@torch.no_grad()
def time_attention(batch, n_head, n_kv_head, head_dim, context, device, impl):
    """GPU time (us) of one decode step of attention: 1 query per sequence over `context` cached keys."""
    q = torch.randn(batch, n_head, 1, head_dim, device=device, dtype=torch.bfloat16)
    kv_bytes = 2 * batch * n_kv_head * context * head_dim * 2
    caches = [(torch.randn(batch, n_kv_head, context, head_dim, device=device, dtype=torch.bfloat16),
               torch.randn(batch, n_kv_head, context, head_dim, device=device, dtype=torch.bfloat16))
              for _ in range(n_copies(kv_bytes, device))]
    return graph_time_us([lambda k=k, v=v: ATTENTION[impl](q, k, v) for k, v in caches])


@torch.no_grad()
def kernel_us(model, input_ids, runs=3):
    """Summed GPU time of all kernels of one forward pass (profiler): the fastest of `runs` forwards, after a
    warm-up to full clocks (an eager forward at batch 1 leaves the GPU idle between kernels, and its clocks
    drift down, e.g. while a model loads)."""
    from torch.profiler import ProfilerActivity, profile
    warm_up(input_ids.device, 0.5)
    for _ in range(2):
        model(input_ids=input_ids, use_cache=False)
    best = math.inf
    for _ in range(runs):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(input_ids=input_ids, use_cache=False)
            torch.cuda.synchronize()
        best = min(best, sum(e.device_time for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA))
    return best


@torch.no_grad()
def build_latency_table(model, layout, formats, mlp_keep, batch, context, log=print):
    """Kernel times on this GPU for the decode latency model (see the module docstring). Measure on an
    otherwise idle GPU: another job slows some measurements and not others, and the parts derived as
    differences (misc_attn, misc_mlp, fixed) then come out wrong."""
    device = next(model.parameters()).device
    assert device.type == "cuda", "the latency table is measured on a CUDA GPU"
    warm_up(device)
    gemm = {}
    for rows, cols, bias in layout.linear_shapes(mlp_keep):
        entry = {}
        for fmt in formats:
            try:
                entry[fmt] = round(time_linear(rows, cols, bias, fmt, batch, device), 3)
            except Exception as e:  # this kernel cannot run this shape: the format falls back to bf16 here
                entry[fmt] = None
                log(f"  {fmt} unavailable for {shape_key(rows, cols, bias)}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        gemm[shape_key(rows, cols, bias)] = entry
        log(f"  {shape_key(rows, cols, bias):>18}: " + ", ".join(
            f"{f} {t:.1f}" if t is not None else f"{f} -" for f, t in entry.items()) + " us")
    heads = (batch, layout.n_head, layout.n_kv_head, layout.head_dim)
    attn = {impl: round(time_attention(*heads, context, device, impl), 3) for impl in ATTENTION}
    attn_impl = min(attn, key=attn.get)  # the decode step uses the fastest implementation
    attn_profiled = round(time_attention(*heads, 1, device, "sdpa"), 3)  # the model's own, in the profiles below
    log(f"  attention per layer at context {context}: " + ", ".join(f"{i} {t:.1f} us" for i, t in attn.items())
        + f" (using {attn_impl}); the model's own at 1 token: {attn_profiled:.1f} us")

    def bf16_us(name, rows=None, cols=None):
        m = layout.module(name)
        return gemm[shape_key(rows or m.out_features, cols or m.in_features, m.bias is not None)]["bf16"]

    # everything that is not a matrix product, from the real model's summed kernel times
    x = torch.zeros(batch, 1, dtype=torch.long, device=device)
    every = range(layout.n_layer)
    full = kernel_us(model, x)
    with removed_sublayers(layout, attn=every):
        no_attn = kernel_us(model, x)
    with removed_sublayers(layout, mlp=every):
        no_mlp = kernel_us(model, x)
    attn_gemm = sum(bf16_us(n) for blk in layout.blocks for n in blk["qkv"] + blk["o"])
    mlp_gemm = sum(bf16_us(n) for blk in layout.blocks for n in blk["up"] + blk["down"])
    L = layout.n_layer
    table = {
        "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__), "batch": batch, "context": context,
        "formats": [f for f in formats if any(e[f] is not None for e in gemm.values())],
        "gemm_us": gemm, "attn_us": attn[attn_impl], "attn_impl": attn_impl, "attn_impl_us": attn,
        "attn_profiled_us": attn_profiled,
        "misc_attn_us": round(max(0.0, (full - no_attn - attn_gemm) / L - attn_profiled), 3),
        "misc_mlp_us": round(max(0.0, (full - no_mlp - mlp_gemm) / L), 3),
        "fixed_us": round(max(0.0, no_attn + no_mlp - full - bf16_us(layout.lm_head)), 3),
        "profile_us": {"full": round(full, 1), "no_attention": round(no_attn, 1), "no_mlp": round(no_mlp, 1)},
    }
    log(f"  real model, one decode-shaped forward: {full:.0f} us of kernels; per layer outside the matrix "
        f"products: attention {table['misc_attn_us']:.1f} us, MLP {table['misc_mlp_us']:.1f} us; "
        f"fixed {table['fixed_us']:.1f} us")
    return table


# -----------------------------------------------------------------------------
# configurations

@dataclass(frozen=True)
class Config:
    fmt: tuple       # fmt[role index in ROLES][band]
    lm_head: str
    embed: str
    keep: tuple      # MLP fraction kept per band
    skip_attn: int
    skip_mlp: int


ZERO = "zero"  # _write marker: the weight becomes all zeros


class HWCompressor:
    """
    Holds the original weights of every matrix a Config can change and writes a configuration into
    the model in place (apply/restore), predicts its latency and memory (cost), or rebuilds the model
    as deployed (materialize). Genes (self.genes, all integers) index the options in this order:
    a format per role and band, lm_head, embed (only with untied embeddings), keep per band,
    skip_attn, skip_mlp.
    """

    def __init__(self, model, layout, table, act_sq, removal_order, formats=FORMATS, bands=4, mlp_keep=MLP_KEEP,
                 max_skip=None, compress_embeddings=True, clip_ratios=CLIP_RATIOS, offload="auto"):
        assert set(formats) <= set(FORMATS), f"formats must be among {FORMATS}"
        self.model, self.layout, self.table = model, layout, table
        self.formats = tuple(f for f in FORMATS if f in formats)
        self.bands = min(bands, layout.n_layer)
        self.band = [i * self.bands // layout.n_layer for i in range(layout.n_layer)]
        self.mlp_keep = tuple(sorted(set(mlp_keep) | {1.0}, reverse=True))  # option 0: nothing pruned
        self.max_skip = layout.n_layer // 4 if max_skip is None else max_skip
        self.act_sq = act_sq
        self.order = {kind: [i for i, _ in v] for kind, v in removal_order.items()}
        self.clip_ratios = tuple(clip_ratios)
        self.compress_embeddings = compress_embeddings and bool(layout.embeds)
        self.genes = [(f"{role}_b{b}", len(self.formats)) for role in ROLES for b in range(self.bands)]
        self.genes.append(("lm_head", len(self.formats)))
        if self.compress_embeddings:
            self.genes.append(("embed", len(self.formats)))
        self.genes += [(f"keep_b{b}", len(self.mlp_keep)) for b in range(self.bands)]
        self.genes += [("skip_attn", self.max_skip + 1), ("skip_mlp", self.max_skip + 1)]

        names = layout.linear_names() + (layout.embeds if self.compress_embeddings else [])
        device = layout.module(names[0]).weight.device
        target_bytes = sum(layout.module(n).weight.numel() * layout.module(n).weight.element_size() for n in names)
        if offload == "auto":
            offload = "no"
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                offload = "yes" if target_bytes > 0.5 * free else "no"
        self.offloaded, self.device = offload == "yes", device
        # (rows, cols, bias) as loaded: the modules themselves change shape and type in materialize()
        self.shapes = {n: (*layout.module(n).weight.shape, getattr(layout.module(n), "bias", None) is not None)
                       for n in names + layout.embeds}
        self.originals = {}
        for n in names:
            w = layout.module(n).weight.detach().clone()
            self.originals[n] = (w.cpu().pin_memory() if device.type == "cuda" else w.cpu()) if self.offloaded else w
        target = {layout.module(n).weight.data_ptr() for n in names} | \
                 {layout.module(n).bias.data_ptr() for n in names if getattr(layout.module(n), "bias", None) is not None}
        self.rest_params = sum(p.numel() for p in model.parameters() if p.data_ptr() not in target)
        self.n_target_params = sum(layout.module(n).weight.numel() for n in names)
        self.neuron_order = []  # per layer, MLP neurons by importance ||W_down[:, j]|| * sqrt(E[h_j^2])
        for blk in layout.blocks:
            down = blk["down"][0]
            score = self._orig(down).float().norm(dim=0) * act_sq[down].to(device).float().sqrt()
            self.neuron_order.append(torch.argsort(score, descending=True))
        self._clip, self._kept, self._touched = {}, {}, []

    # --- genes <-> configurations

    def decode(self, x):
        """Gene vector (any reals) -> the integer genes it stands for (rounded, clipped)."""
        return tuple(int(min(max(round(float(v)), 0), n - 1)) for v, (_, n) in zip(x, self.genes))

    def config(self, genes):
        g = dict(zip((name for name, _ in self.genes), genes))
        fmt = tuple(tuple(self.formats[g[f"{role}_b{b}"]] for b in range(self.bands)) for role in ROLES)
        return Config(fmt, self.formats[g["lm_head"]], self.formats[g["embed"]] if self.compress_embeddings else "bf16",
                      tuple(self.mlp_keep[g[f"keep_b{b}"]] for b in range(self.bands)), g["skip_attn"], g["skip_mlp"])

    def uniform_genes(self, fmt):
        """All matrices in fmt, nothing pruned or removed."""
        f = self.formats.index(fmt)
        return tuple(0 if name.startswith(("keep_", "skip_")) else f for name, _ in self.genes)

    def describe(self, cfg):
        short = {"bf16": "B", "int4_g32": "3", "int4_g64": "6", "int4_g128": "8"}  # 3/6/8: int4 group 32/64/128
        fmt = " ".join(f"{role}:{''.join(short[f] for f in cfg.fmt[r])}" for r, role in enumerate(ROLES))
        return (f"{fmt} head:{short[cfg.lm_head]}" + (f" emb:{short[cfg.embed]}" if self.compress_embeddings else "")
                + f" keep:{'/'.join(f'{k:g}' for k in cfg.keep)} -attn {cfg.skip_attn} -mlp {cfg.skip_mlp}")

    # --- what a configuration keeps

    def removed(self, cfg):
        return set(self.order["attn"][:cfg.skip_attn]), set(self.order["mlp"][:cfg.skip_mlp])

    def kept(self, layer, keep):
        """Indices of the MLP neurons kept (sorted), or None for all."""
        n = n_kept(keep, self.layout.intermediate)
        if n >= self.layout.intermediate:
            return None
        if (layer, n) not in self._kept:
            self._kept[(layer, n)] = self.neuron_order[layer][:n].sort().values
        return self._kept[(layer, n)]

    def _shape(self, name, rows=None, cols=None):
        n_rows, n_cols, bias = self.shapes[name]
        return n_rows if rows is None else len(rows), n_cols if cols is None else len(cols), bias

    def effective(self, name, fmt, rows=None, cols=None):
        """fmt, or bf16 where the kernel cannot run this shape (embeddings are looked up, any format works)."""
        if fmt == "bf16" or name in self.layout.embeds:
            return fmt
        return fmt if self.table["gemm_us"][shape_key(*self._shape(name, rows, cols))].get(fmt) is not None else "bf16"

    def _orig(self, name):
        return self.originals[name].to(self.device, non_blocking=True)

    def int4(self, name, fmt, rows=None, cols=None):
        """(q, scale, zero) of the kept part of the original weight; the clipping choice is cached."""
        group, w = group_size(fmt), self._orig(name)
        act = self.act_sq.get(name)
        if cols is not None:
            w, act = w[:, cols], act[cols.cpu()]
        key = (name, fmt, None if cols is None else len(cols))
        q, scale, zero, idx = int4_quantize(w, group, act, self.clip_ratios, self._clip.get(key))
        self._clip.setdefault(key, idx)
        if rows is not None:  # rows are quantized independently: keep the kept ones
            q, scale, zero = q[rows], scale[rows], zero[rows]
        return q, scale, zero

    def weight(self, name, fmt, rows=None, cols=None):
        """Deployed weight of `name` at full size (pruned rows / columns zero), or None if unchanged."""
        fmt = self.effective(name, fmt, rows, cols)
        if fmt == "bf16" and rows is None and cols is None:
            return None
        w0 = self._orig(name)
        if fmt == "bf16":
            part = w0[rows] if rows is not None else w0[:, cols]
        else:
            part = int4_dequantize(*self.int4(name, fmt, rows, cols), group_size(fmt))
        if rows is None and cols is None:
            return part
        out = torch.zeros(w0.shape, dtype=part.dtype, device=w0.device)
        if rows is not None:
            out[rows] = part
        else:
            out[:, cols] = part
        return out

    # --- fake mode

    def _write(self, name, w):
        if w is None:
            return
        weight = self.layout.module(name).weight
        if w is ZERO:
            weight.zero_()
        else:
            weight.copy_(w.to(weight.dtype))
        self._touched.append(name)

    @torch.no_grad()
    def apply(self, cfg):
        """Write the deployed weights of cfg into the model (undo with restore())."""
        assert not self._touched, "restore() before applying another configuration"
        L = self.layout
        attn_off, mlp_off = self.removed(cfg)
        for i, blk in enumerate(L.blocks):
            b = self.band[i]
            if i in attn_off:
                self._write(blk["o"][0], ZERO)
            else:
                for r, role in ((0, "qkv"), (1, "o")):
                    for name in blk[role]:
                        self._write(name, self.weight(name, cfg.fmt[r][b]))
            if i in mlp_off:
                self._write(blk["down"][0], ZERO)
            else:
                kept = self.kept(i, cfg.keep[b])
                for name in blk["up"]:
                    self._write(name, self.weight(name, cfg.fmt[2][b], rows=kept))
                self._write(blk["down"][0], self.weight(blk["down"][0], cfg.fmt[3][b], cols=kept))
        self._write(L.lm_head, self.weight(L.lm_head, cfg.lm_head))
        if self.compress_embeddings:
            for name in L.embeds:
                self._write(name, self.weight(name, cfg.embed))

    @torch.no_grad()
    def restore(self):
        for name in set(self._touched):
            self.layout.module(name).weight.copy_(self._orig(name))
        self._touched = []

    # --- cost model

    def cost(self, cfg, as_profiled=False):
        """Predicted decode latency (ms, from the table) and deployed memory (MB) of cfg, with details.
        as_profiled: predict instead the forward the table was calibrated on and the validation measures
        (one token, no cache, the model's own attention)."""
        T, L = self.table, self.layout
        attn_us = T["attn_profiled_us"] if as_profiled else T["attn_us"]
        attn_off, mlp_off = self.removed(cfg)
        us, weights, matrix_bytes, int4_params, kept_neurons = T["fixed_us"], 2 * self.rest_params, 0, 0, 0

        def matrix(name, fmt, n_rows=None, n_cols=None):
            nonlocal us, weights, matrix_bytes, int4_params
            rows, cols, bias = self._shape(name, None if n_rows is None else range(n_rows),
                                           None if n_cols is None else range(n_cols))
            fmt = self.effective(name, fmt, range(rows), range(cols))
            nbytes = weight_bytes(rows, cols, fmt)
            if name not in L.embeds:
                us += T["gemm_us"][shape_key(rows, cols, bias)][fmt]
            weights += nbytes + (2 * rows if bias else 0)
            matrix_bytes += nbytes
            int4_params += rows * cols if fmt != "bf16" else 0

        for i, blk in enumerate(L.blocks):
            b = self.band[i]
            if i not in attn_off:
                us += T["misc_attn_us"] + attn_us
                for r, role in ((0, "qkv"), (1, "o")):
                    for name in blk[role]:
                        matrix(name, cfg.fmt[r][b])
            if i not in mlp_off:
                us += T["misc_mlp_us"]
                n = n_kept(cfg.keep[b], L.intermediate)
                kept_neurons += n
                for name in blk["up"]:
                    matrix(name, cfg.fmt[2][b], n_rows=n)
                matrix(blk["down"][0], cfg.fmt[3][b], n_cols=n)
        matrix(L.lm_head, cfg.lm_head)
        if self.compress_embeddings:  # otherwise they are among the rest, in bf16
            for name in L.embeds:
                matrix(name, cfg.embed)
        kv =T["batch"] * T["context"] * (L.n_layer - len(attn_off)) * L.kv_bytes()
        return {"latency_ms": us / 1000, "memory_mb": (weights + kv) / 1e6, "weights_mb": weights / 1e6,
                "kv_mb": kv / 1e6, "bits_per_weight": 8 * matrix_bytes / self.n_target_params,
                "int4_pct": 100 * int4_params / self.n_target_params,
                "mlp_kept_pct": 100 * kept_neurons / (L.n_layer * L.intermediate),
                "n_removed_attn": len(attn_off), "n_removed_mlp": len(mlp_off)}

    # --- the deployed model

    def _deploy(self, name, fmt, rows=None, cols=None):
        m = self.layout.module(name)
        fmt = self.effective(name, fmt, rows, cols)
        if fmt == "bf16" and rows is None and cols is None:
            return
        w, bias = self._orig(name), m.bias
        if rows is not None:
            w, bias = w[rows], None if bias is None else bias[rows]
        if cols is not None:
            w = w[:, cols]
        if fmt == "bf16":
            new = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device=w.device, dtype=m.weight.dtype)
            new.weight.copy_(w)
            if bias is not None:
                new.bias.copy_(bias)
        else:
            new = Int4Linear(*self.int4(name, fmt, rows, cols), group_size(fmt), bias)
        parent, _, attr = name.rpartition(".")
        setattr(self.layout.module(parent), attr, new)

    @torch.no_grad()
    def materialize(self, cfg):
        """Rebuild the model in place as the deployed model of cfg: int4 matrices on the tinygemm kernel,
        MLPs cut to the kept neurons, removed sublayers replaced by Removed. Cannot be undone."""
        assert not self._touched, "restore() first"
        L = self.layout
        attn_off, mlp_off = self.removed(cfg)
        # embeddings take the dequantized values a lookup in their int4 table gives (a tied one: the head's)
        for name in L.embeds:
            w = self.weight(name, cfg.embed if self.compress_embeddings else "bf16")
            if w is not None:
                L.module(name).weight.copy_(w.to(L.module(name).weight.dtype))
        if L.tied:
            w = self.weight(L.lm_head, cfg.lm_head)
            if w is not None:
                L.module(L.lm_head).weight.copy_(w.to(L.module(L.lm_head).weight.dtype))  # the shared parameter
        for i, blk in enumerate(L.blocks):
            b, layer = self.band[i], L.module(L.layers[i])
            if i in attn_off:
                setattr(layer, L.attn_attr, Removed(L.pair))
            else:
                for r, role in ((0, "qkv"), (1, "o")):
                    for name in blk[role]:
                        self._deploy(name, cfg.fmt[r][b])
            if i in mlp_off:
                setattr(layer, L.mlp_attr, Removed())
            else:
                kept = self.kept(i, cfg.keep[b])
                for name in blk["up"]:
                    self._deploy(name, cfg.fmt[2][b], rows=kept)
                self._deploy(blk["down"][0], cfg.fmt[3][b], cols=kept)
        self._deploy(L.lm_head, cfg.lm_head)
