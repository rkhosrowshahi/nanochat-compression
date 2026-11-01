"""
The KV cache as a decision of the search (scripts/compress_search_kv.py), next to the weight formats of
compress_hw and the cost of an agent's turn of compress_mix. In the mix run the cache was 66% of the turn and
61% of the deployed memory, and no gene could change either.

Per depth band, besides the genes of compress_hw:
    key_b<i>, val_b<i>    precision of the cached keys and of the cached values
                              bf16
                              int8, int4: symmetric codes with one bf16 scale per token and KV head
    window_b<i>           fraction of the context the layers of this band keep: the most recent tokens, as a sliding
                          window (HF Qwen2 has per-layer windows natively, as layer_types)

Loss. apply() hooks the model on top of the written weights, so the eval of a configuration sees its cache:
    precision   the outputs of k_proj and v_proj are fake quantized. A deployed cache holds the keys after the rotary
                embedding, which rotates pairs of dimensions, so the error has the same size. The new token reads its
                own quantized entry, as it would.
    window      the attention mask keeps the last round(fraction * T) tokens of a T-token window. Calibration windows
                are far shorter than the context (2048 vs 32000), so this is a PROXY: it says how much text of the
                same relative length a layer can forget, and nothing about an agent losing a file it read 20000
                tokens ago. Fronts with windows below 1 need a check on real agent tasks.

Time and memory. A quantized cache has no attention kernel: each read expands it to bf16 first. build_kv_table measures
that read and the attention at each window, per layer, for decode (each sequence reads its own cache, every token) and
prompts (a chunk reads one cache together, once per chunk). Writing new entries (a few tokens per step) is left out,
and so is the short-lived bf16 copy of one layer's cache during a read.
"""

from dataclasses import dataclass

import torch

from nanochat.compress_hw import graph_time_us, n_copies, time_attention, warm_up
from nanochat.compress_mix import MixCompressor, MixConfig, time_prompt_attention

KV_PRECISIONS = ("bf16", "int8", "int4")  # most accurate first
KV_WINDOWS = (1.0, 0.75, 0.5, 0.25)
KV_GENES = ("key_", "val_", "window_")
LEVELS = {"int8": 127, "int4": 7}
BITS = {"bf16": 16, "int8": 8, "int4": 4}
SCALE_BYTES = 2  # one bf16 scale per token and KV head


def kv_tensor_bytes(precision, n_kv_head, head_dim):
    """Bytes of one cached tensor (keys or values) per layer and token."""
    payload = n_kv_head * head_dim * BITS[precision] // 8
    return payload + (0 if precision == "bf16" else SCALE_BYTES * n_kv_head)


def window_tokens(fraction, length):
    return max(1, round(fraction * length))


@torch.no_grad()
def fake_quantize(x, precision, n_kv_head):
    """k_proj or v_proj output [..., T, n_kv_head * head_dim] as the cache would give it back."""
    if precision == "bf16":
        return x
    v = x.float().view(*x.shape[:-1], n_kv_head, -1)
    scale = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / LEVELS[precision]
    return ((v / scale).round() * scale).to(x.dtype).view_as(x)


def window_mask(mask, length, window, device):
    """The attention mask with only the last `window` tokens visible to each position. mask is None (causal),
    boolean (True attends) or float (added to the scores), [..., T, T]."""
    band = torch.ones(length, length, dtype=torch.bool, device=device)
    band = band.tril() & ~band.tril(-window)
    if mask is None:
        return band[None, None]
    if mask.dtype == torch.bool:
        return mask & band
    return mask.masked_fill(~band, torch.finfo(mask.dtype).min)


def expanded(precision, codes, scale):
    """One read of a cached tensor: its bf16 values."""
    if precision == "int8":
        return codes.to(torch.bfloat16) * scale
    return (torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).to(torch.bfloat16) - 8) * scale


@torch.no_grad()
def time_read(precision, batch, n_kv_head, tokens, head_dim, device):
    """GPU time (us) of one read of a cached tensor of `tokens` tokens per sequence."""
    assert head_dim % 2 == 0, "int4 packs two codes per byte"
    shape = (batch, n_kv_head, tokens)
    copies = []
    for _ in range(n_copies(kv_tensor_bytes(precision, n_kv_head, head_dim) * batch * tokens, device)):
        if precision == "int8":
            codes = torch.randint(-127, 128, (*shape, head_dim), device=device, dtype=torch.int8)
        else:
            codes = torch.randint(0, 256, (*shape, head_dim // 2), device=device, dtype=torch.uint8)
        copies.append((codes, torch.rand(*shape, 1, device=device).bfloat16()))
    return graph_time_us([lambda c=c, s=s: expanded(precision, c, s) for c, s in copies])


@torch.no_grad()
def build_kv_table(layout, precisions, windows, batch, chunk, context, attn_impl, device, log=print):
    """Per layer, in us: decode: one step of `batch` sequences, prompt: one chunk of `chunk` tokens. attention_us by
    window, read_us by precision and window (one tensor; bf16 needs no read)."""
    warm_up(device)
    heads = (layout.n_head, layout.n_kv_head, layout.head_dim)
    table = {"precisions": list(precisions), "windows": list(windows), "context": context,
             "decode": {"batch": batch, "attention_us": {}, "read_us": {}},
             "prompt": {"batch": chunk, "attention_us": {}, "read_us": {}}}
    for w in windows:
        tokens, key = window_tokens(w, context), f"{w:g}"
        table["decode"]["attention_us"][key] = round(time_attention(batch, *heads, tokens, device, attn_impl), 3)
        table["prompt"]["attention_us"][key] = round(time_prompt_attention(chunk, *heads, tokens, device) * chunk, 3)
        log(f"  window {key} ({tokens} tokens): attention {table['decode']['attention_us'][key]:.1f} us per decode step, "
            f"{table['prompt']['attention_us'][key]:.1f} us per prompt chunk")
    for p in precisions:
        if p == "bf16":
            continue
        for part, b in (("decode", batch), ("prompt", 1)):
            table[part]["read_us"][p] = {f"{w:g}": round(time_read(p, b, layout.n_kv_head, window_tokens(w, context), layout.head_dim, device), 3)
                                         for w in windows}
        log(f"  {p}: one read at the full context {table['decode']['read_us'][p][f'{windows[0]:g}']:.1f} us per decode step, "
            f"{table['prompt']['read_us'][p][f'{windows[0]:g}']:.1f} us per prompt chunk")
    return table


@dataclass(frozen=True)
class KVConfig(MixConfig):
    key: tuple = ()      # precision of the cached keys per band
    value: tuple = ()    # of the cached values per band
    window: tuple = ()   # fraction of the context kept per band


class KVCompressor(MixCompressor):
    """MixCompressor without sharing, with the KV genes: its apply() also hooks the attention of the model (restore()
    removes the hooks) and its cost() takes attention and KV memory from kv_table (build_kv_table)."""

    def __init__(self, model, layout, table, prompt_table, kv_table, act_sq, removal_order, context=32000, **kw):
        assert layout.pair, "the KV genes hook HF Llama/Qwen2 attention"
        super().__init__(model, layout, table, prompt_table, act_sq, removal_order, sharing=False, **kw)
        self.kv_table, self.context = kv_table, context
        self.kv_precisions, self.kv_windows = tuple(kv_table["precisions"]), tuple(kv_table["windows"])
        for name, n in (("key", len(self.kv_precisions)), ("val", len(self.kv_precisions)), ("window", len(self.kv_windows))):
            self.genes += [(f"{name}_b{b}", n) for b in range(self.bands)]
        self._hooks = []

    # --- genes <-> configurations

    def config(self, genes):
        g = dict(zip((name for name, _ in self.genes), genes))
        per_band = lambda name, options: tuple(options[g[f"{name}_b{b}"]] for b in range(self.bands))
        base = super().config(genes)
        return KVConfig(base.fmt, base.lm_head, base.embed, base.keep, base.skip_attn, base.skip_mlp, None,
                        per_band("key", self.kv_precisions), per_band("val", self.kv_precisions), per_band("window", self.kv_windows))

    def uniform_genes(self, fmt, precision="bf16", window=1.0):
        """All matrices in fmt, nothing pruned or removed, one precision and window for the whole cache."""
        g = dict(zip((n for n, _ in self.genes), super().uniform_genes(fmt)))
        for name in g:
            if name.startswith(("key_", "val_")):
                g[name] = self.kv_precisions.index(precision)
            elif name.startswith("window_"):
                g[name] = self.kv_windows.index(window)
        return tuple(g.values())

    def extra_seeds(self):
        """Starting points for the cache: each precision and window alone, and with the cheapest weights."""
        cheapest = self.formats[-1]
        seeds = [self.uniform_genes(f, precision=p) for f in ("bf16", cheapest) for p in self.kv_precisions[1:]]
        return seeds + [self.uniform_genes("bf16", window=w) for w in self.kv_windows[1:]]

    def describe(self, cfg):
        short = {"bf16": "B", "int8": "8", "int4": "4"}
        return (super().describe(cfg) + f" key:{''.join(short[p] for p in cfg.key)} val:{''.join(short[p] for p in cfg.value)}"
                + f" win:{'/'.join(f'{w:g}' for w in cfg.window)}")

    # --- the cache in the loss

    def apply(self, cfg):
        stats = super().apply(cfg)
        self._hook(cfg)
        return stats

    def restore(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        super().restore()

    def _hook(self, cfg):
        L = self.layout
        for i, blk in enumerate(L.blocks):
            b = self.band[i]
            _, k_proj, v_proj = blk["qkv"]
            for name, precision in ((k_proj, cfg.key[b]), (v_proj, cfg.value[b])):
                if precision != "bf16":
                    self._hooks.append(L.module(name).register_forward_hook(
                        lambda m, args, out, p=precision: fake_quantize(out, p, L.n_kv_head)))
            if cfg.window[b] < 1:
                def narrow(module, args, kwargs, fraction=cfg.window[b]):
                    length = kwargs["hidden_states"].shape[1] if "hidden_states" in kwargs else args[0].shape[1]
                    device = kwargs["hidden_states"].device if "hidden_states" in kwargs else args[0].device
                    kwargs["attention_mask"] = window_mask(kwargs.get("attention_mask"), length, window_tokens(fraction, length), device)
                    return args, kwargs
                attn = getattr(L.module(L.layers[i]), L.attn_attr)
                self._hooks.append(attn.register_forward_pre_hook(narrow, with_kwargs=True))

    # --- cost model

    def _layer_us(self, part, cfg, band):
        """GPU time per layer of one decode step or prompt chunk: attention over the window plus the two reads."""
        T, w = self.kv_table[part], f"{cfg.window[band]:g}"
        return T["attention_us"][w] + sum(T["read_us"][p][w] for p in (cfg.key[band], cfg.value[band]) if p != "bf16")

    def cost(self, cfg):
        """As MixCompressor.cost, with the attention and the KV memory of cfg's cache."""
        base = super().cost(cfg)
        L = self.layout
        attn_off, _ = self.removed(cfg)
        decode, prompt, per_token = 0.0, 0.0, 0.0
        for i in range(L.n_layer):
            if i in attn_off:
                continue
            b = self.band[i]
            decode += self._layer_us("decode", cfg, b) - self.table["attn_us"]
            prompt += self._layer_us("prompt", cfg, b) - self.prompt_table["attn_us"]
            per_token += cfg.window[b] * (kv_tensor_bytes(cfg.key[b], L.n_kv_head, L.head_dim) + kv_tensor_bytes(cfg.value[b], L.n_kv_head, L.head_dim))
        decode_us = base["decode_us"] + decode / self.table["batch"]
        prompt_us = base["prompt_us"] + prompt / self.prompt_table["batch"]
        kv_per_agent = self.context * per_token / 1e6
        kv_mb = kv_per_agent * self.table["batch"]
        room = self.vram_mb * (1 - self.reserve) - base["weights_mb"]
        full = (L.n_layer - len(attn_off)) * L.kv_bytes()
        return {**base, "latency_ms": (self.new_tokens * prompt_us + self.gen_tokens * decode_us) / 1e3, "decode_us": decode_us,
                "prompt_us": prompt_us, "kv_mb": kv_mb, "memory_mb": base["weights_mb"] + kv_mb,
                "agents_fit": int(max(0.0, room) // kv_per_agent) if kv_per_agent > 0 else 10 ** 9,
                "kv_pct": 100 * per_token / full if full else 0.0,
                "window_pct": 100 * sum(cfg.window[self.band[i]] for i in range(L.n_layer)) / L.n_layer}
