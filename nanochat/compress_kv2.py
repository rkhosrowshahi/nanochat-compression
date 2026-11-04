"""
The KV cache as a decision of the search, second version (scripts/compress_search_kv2.py), from what the KV probe
(scripts/compress_kv_probe.py, compress_runs/kv_probe) found on the A100 about qwen_kv's cache genes:
    - keys with one scale per token cost +0.037 nats at int8 and break at int4 (+1.61, 6% of planted facts recalled);
      one scale per channel and group of 64 tokens (as KIVI) costs +0.0004 at int8 and +0.070 at int4 (100% recalled)
    - values take int4 for +0.008; nothing survives int2
    - a plain window loses the first token, which every layer attends to (an attention sink, as StreamingLLM): +0.87
      at 0.75. Kept visible, the loss is small (+0.004), but the window still forgets what scrolled out of it
      (0.75: 69% recalled, 0.25: 14%), and calibration loss cannot see that
    - the expand-then-attend time model made every quantized cache slower than bf16 (int8: +53% turn time)

Per depth band, besides the genes of compress_hw (no weight sharing):
    key_b<i>      cached keys: bf16, int8c or int4c (symmetric codes, one bf16 scale per channel and group of
                  `group` tokens)
    val_b<i>      cached values: bf16, int8 or int4 (one bf16 scale per token and KV head)
    window_b<i>   fraction of the context the layers of the band keep: the most recent tokens, plus the first one

Loss and recall. apply() switches the model's attention to kv2_attention, which takes the keys after the rotary embedding
(what a cache holds and what the kernel reads) and gives each layer its cache:
    keys    as deployed: a group is quantized once it is complete, and until then its keys stay in bf16. Each query
            reads the quantized keys of the groups before its own and the bf16 keys of its own group (an attention over
            the two copies, 2T keys, with that mask). No scale is computed from tokens a query has not seen.
    values  quantized as written (their scales are per token).
    A layer with a bf16 cache and the whole window takes the plain causal attention (flash), the others an attention
    with a mask (the keys repeated for the query heads, so the memory-efficient kernel takes it, not the math fallback).
    window  the last round(fraction * T) tokens of a T-token window, and the first token. A PROXY of the same fraction of
            the context (calibration windows: 2048 tokens), so the search also scores recall (recall_set): a fact
            planted at a random depth of a window and asked for at its end, as exact-match accuracy. The script makes
            it a constraint.

Time and memory (build_kv2_table):
    decode  one step of attention with the cache read in place, by the Triton kernel of compress_kv2_kernel.py, per key
            precision, value precision and window. For bf16 the faster of that kernel and compress_hw's attention.
    prompt  a chunk reads the cache once: expanded to bf16 (measured), then the attention of compress_mix.
A window is timed at its tokens rounded up to a multiple of 64 (timed_tokens): the first table timed 24001 tokens for 0.75,
and a matrix product of that odd length took 1.6x the time of an aligned one.
The bf16 keys of an unfinished group (at most group-1 tokens per layer) are left out of time and memory, and so are the
writes of new entries.
"""

import math

import torch
import torch.nn.functional as F
from transformers import AttentionInterface

from nanochat.compress_hw import ATTENTION, graph_time_us, n_copies, time_attention, warm_up
from nanochat.compress_kv import KVConfig, window_tokens
from nanochat.compress_mix import MixCompressor, time_prompt_attention

KEY_KINDS = ("bf16", "int8c", "int4c")  # most accurate first
VALUE_KINDS = ("bf16", "int8", "int4")
KV2_WINDOWS = (1.0, 0.75, 0.5, 0.25)
GROUP = 64  # tokens per key scale
SINKS = 1  # first tokens every window keeps
LEVELS = {"int8": 127, "int4": 7}
IMPL = "nanochat_kv2"


def key_bytes(kind, n_kv_head, head_dim, group=GROUP):
    """Bytes of the cached keys per layer and token (scales spread over their group)."""
    payload = {"bf16": 2 * head_dim, "int8c": head_dim, "int4c": head_dim // 2}[kind]
    return n_kv_head * (payload + (0 if kind == "bf16" else 2 * head_dim / group))


def value_bytes(kind, n_kv_head, head_dim):
    payload = {"bf16": 2 * head_dim, "int8": head_dim, "int4": head_dim // 2}[kind]
    return n_kv_head * (payload + (0 if kind == "bf16" else 2))


def kept_tokens(fraction, context):
    """Tokens of a cache at this window: the most recent ones and the first SINKS."""
    return context if fraction >= 1 else min(context, window_tokens(fraction, context) + SINKS)


def timed_tokens(fraction, context, align=64):
    """The tokens a window is timed at: kept_tokens rounded up to a multiple of align (a cache is allocated in pages)."""
    return -(-kept_tokens(fraction, context) // align) * align


# --- the cache formats (tensors [B, Hkv, T, D])

def _pack(codes, bits):
    """Integer codes (float) -> int8, or uint8 with channel j and j + D/2 in one byte."""
    if bits == "int8":
        return codes.to(torch.int8)
    c = (codes + 8).to(torch.uint8)
    half = c.shape[-1] // 2
    return c[..., :half] | (c[..., half:] << 4)


def _unpack(codes, bits):
    if bits == "int8":
        return codes.float()
    return torch.cat(((codes & 15).float() - 8, (codes >> 4).float() - 8), dim=-1)


@torch.no_grad()
def quantize_keys(k, kind, group=GROUP):
    """Keys -> (codes, scales [B, Hkv, ceil(T / group), D] bf16): one scale per channel and group of tokens."""
    B, Hk, T, D = k.shape
    bits = kind[:-1]
    x = F.pad(k.float(), (0, 0, 0, (-T) % group)).view(B, Hk, -1, group, D)
    scale = (x.abs().amax(3) / LEVELS[bits]).clamp_min(1e-8).bfloat16()
    codes = (x / scale.float()[:, :, :, None]).round().clamp(-LEVELS[bits], LEVELS[bits]).view(B, Hk, -1, D)[:, :, :T]
    return _pack(codes, bits), scale


def dequantize_keys(codes, scale, kind, group=GROUP):
    T = codes.shape[2]
    return _unpack(codes, kind[:-1]) * scale.float().repeat_interleave(group, dim=2)[:, :, :T]


@torch.no_grad()
def quantize_values(v, kind):
    """Values -> (codes, scales [B, Hkv, T] bf16): one scale per token and head."""
    scale = (v.float().abs().amax(-1) / LEVELS[kind]).clamp_min(1e-8).bfloat16()
    codes = (v.float() / scale.float()[..., None]).round().clamp(-LEVELS[kind], LEVELS[kind])
    return _pack(codes, kind), scale


def dequantize_values(codes, scale, kind):
    return _unpack(codes, kind) * scale.float()[..., None]


def cached_keys(k, kind, group=GROUP):
    """k as the quantized part of the cache gives it back."""
    return k if kind == "bf16" else dequantize_keys(*quantize_keys(k, kind, group), kind, group).to(k.dtype)


def cached_values(v, kind):
    return v if kind == "bf16" else dequantize_values(*quantize_values(v, kind), kind).to(v.dtype)


# --- the cache in the loss: an attention function of the HF interface

_masks = {}


def cache_mask(length, window, quantized, group, device):
    """[T, T] (bf16 keys) or [T, 2T] (quantized keys, then the same keys in bf16): True where a query reads a key.
    window: tokens kept (the most recent ones; the first SINKS are always visible)."""
    key = (length, window, quantized, group, str(device))
    if key not in _masks:
        if len(_masks) > 32:
            _masks.clear()
        i = torch.arange(length, device=device)[:, None]
        j = torch.arange(length, device=device)[None, :]
        band = (j <= i) & ((i - j < window) | (j < SINKS))
        if quantized:
            done = (j // group) < (i // group)  # the key's group is complete before the query's
            band = torch.cat((band & done, band & ~done), dim=1)
        _masks[key] = band
    return _masks[key]


def kv2_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query [B, H, T, D], key and value [B, Hkv, T, D] (keys after the rotary embedding). module.kv2 holds
    (key kind, value kind, window fraction, group) for its layer."""
    key_kind, value_kind, fraction, group = module.kv2
    B, H, T, D = query.shape
    assert key.shape[2] == T, "kv2_attention scores whole windows (no cache of earlier tokens)"
    scale = scaling if scaling is not None else D ** -0.5
    if key_kind == value_kind == "bf16" and fraction >= 1 and attention_mask is None:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale, enable_gqa=H != key.shape[1])
        return out.transpose(1, 2).contiguous(), None
    quantized = key_kind != "bf16"
    mask = cache_mask(T, window_tokens(fraction, T) if fraction < 1 else T, quantized, group, query.device)[None, None]
    if attention_mask is not None:  # padding: the same keys in both copies
        given = attention_mask[..., :T] if attention_mask.dtype == torch.bool else attention_mask[..., :T] > torch.finfo(attention_mask.dtype).min / 2
        mask = mask & (torch.cat((given, given), dim=-1) if quantized else given)
    v = cached_values(value, value_kind)
    if quantized:
        key, v = torch.cat((cached_keys(key, key_kind, group), key), dim=2), torch.cat((v, v), dim=2)
    repeat = H // key.shape[1]
    out = F.scaled_dot_product_attention(query, key.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
                                         attn_mask=mask, scale=scale)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(IMPL, kv2_attention)


# --- planted-fact recall

def recall_set(tokenizer, filler, n, length, seed=0):
    """n windows of `length` tokens of filler text (a flat list of token ids), each with one fact ("The access code for
    project 7 is 48213. Remember it.") at a random depth in the first 90% and the question about it at the end.
    Returns (tokens [n, length], answer lengths [n])."""
    import random
    rng = random.Random(seed)
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    rows, answers = [], []
    for j in range(n):
        code = str(rng.randint(10000, 99999))
        fact = enc(f" The access code for project {j} is {code}. Remember it.")
        ask = enc(f" What is the access code for project {j}? The access code for project {j} is")
        answer = enc(f" {code}")
        body = length - len(fact) - len(ask) - len(answer)
        start = rng.randint(0, len(filler) - body - 1)
        at = int(rng.uniform(0.0, 0.9) * body)
        text = filler[start:start + body]
        rows.append(text[:at] + fact + text[at:] + ask + answer)
        answers.append(len(answer))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(answers)


@torch.no_grad()
def recall_score(model, tokens, answers, batch_size=8):
    """(mean cross-entropy of the answer tokens, fraction of answers predicted exactly)."""
    device = next(model.parameters()).device
    keep = int(answers.max()) + 1
    ce, hits = 0.0, 0
    for s in range(0, tokens.size(0), batch_size):
        x = tokens[s:s + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False, logits_to_keep=keep).logits.float()
        for r, n in enumerate(answers[s:s + batch_size].tolist()):
            pred, target = logits[r, keep - n - 1:keep - 1], x[r, -n:]
            ce += F.cross_entropy(pred, target).item()
            hits += bool((pred.argmax(-1) == target).all())
    return ce / tokens.size(0), hits / tokens.size(0)


# --- time table

def random_cache(kind, batch, n_kv_head, tokens, head_dim, device, keys, group=GROUP):
    """(codes, scales) of a random cache tensor in kind's layout (scales None for bf16)."""
    x = torch.randn(batch, n_kv_head, tokens, head_dim, device=device, dtype=torch.bfloat16)
    if kind == "bf16":
        return x, None
    return quantize_keys(x, kind, group) if keys else quantize_values(x, kind)


def reference_decode(q, k, ks, v, vs, key, value, group=GROUP):
    """What DecodeAttention computes, in float32 from the dequantized cache."""
    kf = k.float() if key == "bf16" else dequantize_keys(k, ks, key, group)
    vf = v.float() if value == "bf16" else dequantize_values(v, vs, value)
    B, H, D = q.shape
    qg = q.float().view(B, kf.shape[1], H // kf.shape[1], D)
    a = torch.softmax(qg @ kf.transpose(-1, -2) * D ** -0.5, dim=-1)
    return (a @ vf).view(B, H, D)


@torch.no_grad()
def check_kernel(layout, key, value, device, tokens=1000, group=GROUP, block_t=64):
    """Largest error of the kernel against reference_decode on a random cache (raises if it is wrong)."""
    from nanochat.compress_kv2_kernel import DecodeAttention
    B, (H, Hk, D) = 2, (layout.n_head, layout.n_kv_head, layout.head_dim)
    q = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
    k, ks = random_cache(key, B, Hk, tokens, D, device, True, group)
    v, vs = random_cache(value, B, Hk, tokens, D, device, False)
    out = DecodeAttention(B, H, Hk, D, tokens, key, value, group, block_t=block_t, device=device)(q, k, ks, v, vs)
    ref = reference_decode(q, k, ks, v, vs, key, value, group)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 0.02, f"the decode kernel for keys {key}, values {value} is off by {err:.3f} of the largest output"
    return err


KERNEL_CONFIGS = [(f, b, w, s) for f in (1, 2, 4) for b in (64, 128) for w in (2, 4) for s in (2, 3)]  # split factor, block, warps, stages
KERNEL_VERSION = 2  # of the decode kernel: tables of another version are measured again


@torch.no_grad()
def time_fused(batch, layout, tokens, key, value, device, group=GROUP):
    """GPU time (us) of one decode step of the kernel over `tokens` cached tokens per sequence: the fastest launch
    configuration, with copies of the cache that do not fit in the L2 cache together."""
    from nanochat.compress_kv2_kernel import DecodeAttention
    H, Hk, D = layout.n_head, layout.n_kv_head, layout.head_dim
    tokens = math.ceil(tokens / group) * group
    q = torch.randn(batch, H, D, device=device, dtype=torch.bfloat16)
    nbytes = batch * tokens * (key_bytes(key, Hk, D, group) + value_bytes(value, Hk, D))
    caches = [(*random_cache(key, batch, Hk, tokens, D, device, True, group), *random_cache(value, batch, Hk, tokens, D, device, False))
              for _ in range(n_copies(nbytes, device))]
    best = (math.inf, None)
    for f, b, w, s in KERNEL_CONFIGS:
        attn = DecodeAttention(batch, H, Hk, D, tokens, key, value, group, split_factor=f, block_t=b, num_warps=w, num_stages=s, device=device)
        best = min(best, (graph_time_us([lambda c=c, a=attn: a(q, *c) for c in caches]), (f, b, w, s)))
    return best


def expand(kind, codes, scale, keys, group=GROUP):
    return (dequantize_keys(codes, scale, kind, group) if keys else dequantize_values(codes, scale, kind)).bfloat16()


@torch.no_grad()
def time_expand(kind, n_kv_head, tokens, head_dim, device, keys, group=GROUP):
    """GPU time (us) of expanding one cached tensor of one sequence to bf16 (a prompt chunk reads it once)."""
    nbytes = tokens * (key_bytes(kind, n_kv_head, head_dim, group) if keys else value_bytes(kind, n_kv_head, head_dim))
    copies = [random_cache(kind, 1, n_kv_head, tokens, head_dim, device, keys, group) for _ in range(n_copies(nbytes, device))]
    return graph_time_us([lambda c=c, s=s: expand(kind, c, s, keys, group) for c, s in copies])


@torch.no_grad()
def build_kv2_table(layout, keys, values, windows, batch, chunk, context, device, group=GROUP, log=print):
    """Per layer, in us. decode: one step of `batch` sequences, attention_us["<key>/<value>"][window]; prompt: one chunk of
    `chunk` tokens, attention_us[window] (bf16) and read_us[kind][window] (one expand of a quantized tensor)."""
    warm_up(device)
    heads = (layout.n_head, layout.n_kv_head, layout.head_dim)
    for k in keys:
        for v in values:
            err = max(check_kernel(layout, k, v, device, group=group, block_t=b) for b in sorted({b for _, b, _, _ in KERNEL_CONFIGS}))
            log(f"  kernel check, keys {k}, values {v}: largest error {err:.4f} of the output")
    table = {"keys": list(keys), "values": list(values), "windows": list(windows), "context": context, "group": group, "sinks": SINKS,
             "kernel_version": KERNEL_VERSION,
             "decode": {"batch": batch, "attention_us": {}, "kernel_us": {}, "kernel_config": {}, "bf16_reference_us": {}},
             "prompt": {"batch": chunk, "attention_us": {}, "read_us": {}}}
    for w in windows:
        tokens, wk = timed_tokens(w, context), f"{w:g}"
        reference = min(time_attention(batch, *heads, tokens, device, impl) for impl in ATTENTION)
        table["decode"]["bf16_reference_us"][wk] = round(reference, 3)
        for k in keys:
            for v in values:
                fused, launch = time_fused(batch, layout, tokens, k, v, device, group)
                kv = f"{k}/{v}"
                table["decode"]["kernel_us"].setdefault(kv, {})[wk] = round(fused, 3)
                table["decode"]["kernel_config"].setdefault(kv, {})[wk] = launch
                table["decode"]["attention_us"].setdefault(kv, {})[wk] = round(min(fused, reference) if kv == "bf16/bf16" else fused, 3)
        table["prompt"]["attention_us"][wk] = round(time_prompt_attention(chunk, *heads, tokens, device) * chunk, 3)
        for kinds, is_key in ((keys, True), (values, False)):
            for kind in kinds:
                if kind != "bf16":
                    table["prompt"]["read_us"].setdefault(kind, {})[wk] = round(time_expand(kind, layout.n_kv_head, tokens, layout.head_dim, device, is_key, group), 3)
        d = table["decode"]["attention_us"]
        log(f"  window {wk} ({tokens} tokens), us per layer and decode step: bf16 {d['bf16/bf16'][wk]:.1f} (kernel "
            f"{table['decode']['kernel_us']['bf16/bf16'][wk]:.1f}, torch {reference:.1f}), " + ", ".join(
                f"{kv} {t[wk]:.1f}" for kv, t in d.items() if kv != "bf16/bf16")
            + f"; prompt chunk {table['prompt']['attention_us'][wk]:.0f} us + reads " + ", ".join(
                f"{kind} {t[wk]:.1f}" for kind, t in table["prompt"]["read_us"].items()))
    return table


# --- the compressor

class KV2Compressor(MixCompressor):
    """MixCompressor without sharing, with the cache genes above: apply() also gives the model's attention its cache
    (restore() takes it back) and cost() takes attention and KV memory from kv2_table (build_kv2_table)."""

    def __init__(self, model, layout, table, prompt_table, kv2_table, act_sq, removal_order, context=32000, **kw):
        assert layout.pair, "kv2_attention is an attention function of HF Llama/Qwen2 models"
        super().__init__(model, layout, table, prompt_table, act_sq, removal_order, sharing=False, **kw)
        self.kv_table, self.context, self.group = kv2_table, context, kv2_table["group"]
        self.key_kinds, self.value_kinds = tuple(kv2_table["keys"]), tuple(kv2_table["values"])
        self.kv_windows = tuple(kv2_table["windows"])
        for name, n in (("key", len(self.key_kinds)), ("val", len(self.value_kinds)), ("window", len(self.kv_windows))):
            self.genes += [(f"{name}_b{b}", n) for b in range(self.bands)]
        self._attn_impl = None

    # --- genes <-> configurations

    def config(self, genes):
        g = dict(zip((name for name, _ in self.genes), genes))
        per_band = lambda name, options: tuple(options[g[f"{name}_b{b}"]] for b in range(self.bands))
        base = super().config(genes)
        return KVConfig(base.fmt, base.lm_head, base.embed, base.keep, base.skip_attn, base.skip_mlp, None,
                        per_band("key", self.key_kinds), per_band("val", self.value_kinds), per_band("window", self.kv_windows))

    def uniform_genes(self, fmt, key="bf16", value="bf16", window=1.0):
        """All matrices in fmt, nothing pruned or removed, one cache format and window for all layers."""
        g = dict(zip((n for n, _ in self.genes), super().uniform_genes(fmt)))
        for name in g:
            if name.startswith("key_"):
                g[name] = self.key_kinds.index(key)
            elif name.startswith("val_"):
                g[name] = self.value_kinds.index(value)
            elif name.startswith("window_"):
                g[name] = self.kv_windows.index(window)
        return tuple(g.values())

    def extra_seeds(self):
        """Starting points for the cache: each key and value precision alone and together, with bf16 and with the
        cheapest weights, and each window alone."""
        cheapest = self.formats[-1]
        pairs = [(k, "bf16") for k in self.key_kinds[1:]] + [("bf16", v) for v in self.value_kinds[1:]] \
            + [(self.key_kinds[-1], self.value_kinds[-1])]
        seeds = [self.uniform_genes(f, key=k, value=v) for f in ("bf16", cheapest) for k, v in pairs]
        return list(dict.fromkeys(seeds + [self.uniform_genes("bf16", window=w) for w in self.kv_windows[1:]]))

    def describe(self, cfg):
        short = {"bf16": "B", "int8c": "8", "int4c": "4", "int8": "8", "int4": "4"}
        return (super().describe(cfg) + f" key:{''.join(short[p] for p in cfg.key)} val:{''.join(short[p] for p in cfg.value)}"
                + f" win:{'/'.join(f'{w:g}' for w in cfg.window)}")

    # --- the cache in the loss

    def apply(self, cfg):
        stats = super().apply(cfg)
        L = self.layout
        if any(k != "bf16" for k in cfg.key + cfg.value) or any(w < 1 for w in cfg.window):
            for i in range(L.n_layer):
                b = self.band[i]
                getattr(L.module(L.layers[i]), L.attn_attr).kv2 = (cfg.key[b], cfg.value[b], cfg.window[b], self.group)
            self._attn_impl = self.model.config._attn_implementation
            self.model.config._attn_implementation = IMPL
        return stats

    def restore(self):
        if self._attn_impl is not None:
            self.model.config._attn_implementation = self._attn_impl
            self._attn_impl = None
            L = self.layout
            for i in range(L.n_layer):
                del getattr(L.module(L.layers[i]), L.attn_attr).kv2
        super().restore()

    # --- cost model

    def cost(self, cfg):
        """As MixCompressor.cost, with the attention and the KV memory of cfg's cache."""
        base = super().cost(cfg)
        L, D, P = self.layout, self.kv_table["decode"], self.kv_table["prompt"]
        attn_off, _ = self.removed(cfg)
        decode, prompt, nbytes = 0.0, 0.0, 0.0
        for i in range(L.n_layer):
            if i in attn_off:
                continue
            b = self.band[i]
            k, v, w = cfg.key[b], cfg.value[b], f"{cfg.window[b]:g}"
            decode += D["attention_us"][f"{k}/{v}"][w] - self.table["attn_us"]
            prompt += P["attention_us"][w] + sum(P["read_us"][x][w] for x in (k, v) if x != "bf16") - self.prompt_table["attn_us"]
            nbytes += kept_tokens(cfg.window[b], self.context) * (key_bytes(k, L.n_kv_head, L.head_dim, self.group)
                                                                  + value_bytes(v, L.n_kv_head, L.head_dim))
        decode_us = base["decode_us"] + decode / self.table["batch"]
        prompt_us = base["prompt_us"] + prompt / self.prompt_table["batch"]
        kv_per_agent = nbytes / 1e6
        kv_mb = kv_per_agent * self.table["batch"]
        room = self.vram_mb * (1 - self.reserve) - base["weights_mb"]
        full = (L.n_layer - len(attn_off)) * L.kv_bytes() * self.context
        return {**base, "latency_ms": (self.new_tokens * prompt_us + self.gen_tokens * decode_us) / 1e3, "decode_us": decode_us,
                "prompt_us": prompt_us, "kv_mb": kv_mb, "memory_mb": base["weights_mb"] + kv_mb,
                "agents_fit": int(max(0.0, room) // kv_per_agent) if kv_per_agent > 0 else 10 ** 9,
                "kv_pct": 100 * nbytes / full if full else 0.0,
                "window_pct": 100 * sum(cfg.window[self.band[i]] for i in range(L.n_layer)) / L.n_layer}
