"""
Time per output token (TPOT) of a causal LM as deployed: the decode-speed measure of the LLM compression literature
(GPTQ, AWQ, SqueezeLLM, AQLM report ms/token or tokens/s at batch 1) and of serving benchmarks (vLLM, MLPerf
Inference, GenAI-Perf), with the prompt processed first and timed apart:

    TTFT   time to first token: processing a prompt of `prompt_tokens` and choosing the first new token
    TPOT   (end-to-end time - TTFT) / (gen_tokens - 1): the mean time of one decode step after the first token,
           over contexts from prompt_tokens to prompt_tokens + gen_tokens
    tok/s  1000 / TPOT per sequence; at batch B every sequence gets one token per step, so the total is B times it.
           Per-sequence time is never divided by the batch

How (decode_timing): greedy decoding of random prompt tokens (as llama-bench does: the time does not depend on the
text, and the end-of-sequence token is ignored) into a static KV cache (transformers.StaticCache), with each decode
step - forward, argmax, next position - captured once as a CUDA graph and replayed. Replaying the graph is what
serving runtimes do (vLLM, llama.cpp, TensorRT-LLM): eager PyTorch spends most of a small model's step launching
kernels from Python (Qwen2.5-0.5B on an RTX 3060: about 31 ms per token eager, 7.9 ms replayed), which would hide
every speedup that compression gives. The tokens the graph produces are checked against eager decoding.
Times come from CUDA events, the median over `repeats` runs after warm-up.

Attention (attention="grouped", the default): each decode step runs GROUPED_IMPL, where every query head reads its
KV head's cache in place (compress_hw.grouped_attention with the causal mask). transformers' own sdpa path cannot do
that: with a mask, SDPA has no grouped-query mode, so it first copies the keys and values out to every query head
(repeat_kv) on every step. On Qwen2.5-0.5B (14 query heads, 2 KV heads) that copy and the attention after it took
about 2 ms of a 7.8 ms step, and they grow with the batch. Serving runtimes (vLLM, llama.cpp, FlashAttention's decode
kernels) read the cache in place, and so does the latency table. The prompt still runs through sdpa (unchanged TTFT).
attention="model" keeps the model's own implementation.

The static cache attends over its whole length (prompt_tokens + gen_tokens) at every step, masking the positions not
yet written, as transformers' static cache always does; a paged-attention runtime reads only the written part.
"""

import contextlib
import statistics

import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

from nanochat.compress_hw import ROLES, Config, HWCompressor

GROUPED_IMPL = "tpot_grouped"
MiB = 1 << 20


class TPOTCompressor(HWCompressor):
    """HWCompressor for scripts/compress_search_tpot.py: a configuration costs its decode TPOT (the latency table's step at
    batch 1, see the module docstring) and its memory as loaded on the GPU (the model's tensors), nothing else (no KV
    cache or activations, no separate stored size). With bands = n_layer every layer has its own format per role. Genes with one option (MLP width when nothing
    is pruned, sublayer removal when max_skip is 0) are left out of the gene vector and take their only value.

    Memory: with an output head tied to the embedding, HWCompressor.cost counts the shared matrix once, in the head's
    format. A head in int4 is a second copy, though: the embedding lookup still reads the bf16 matrix (measured on
    Qwen2.5-Coder-1.5B: 1.199 GiB, not 1.083). cost() adds that bf16 embedding back."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed = {name: 0 for name, n in self.genes if n == 1}
        self.genes = [(name, n) for name, n in self.genes if n > 1]

    def config(self, genes):
        g = {**self.fixed, **dict(zip((name for name, _ in self.genes), genes))}
        fmt = tuple(tuple(self.formats[g[f"{role}_b{b}"]] for b in range(self.bands)) for role in ROLES)
        return Config(fmt, self.formats[g["lm_head"]], self.formats[g["embed"]] if self.compress_embeddings else "bf16",
                      tuple(self.mlp_keep[g[f"keep_b{b}"]] for b in range(self.bands)), g["skip_attn"], g["skip_mlp"])

    def describe(self, cfg):
        """As HWCompressor.describe, without the MLP widths and removals when they are not searched."""
        text = super().describe(cfg)
        if all(name.startswith("keep_") or name.startswith("skip_") for name in self.fixed) and self.fixed:
            pruned = any(f"keep_b{b}" not in self.fixed for b in range(self.bands)) or "skip_attn" not in self.fixed
            if not pruned:
                text = text.split(" keep:")[0]
        return text

    def apply(self, cfg):
        """As HWCompressor.apply; returns the (no) stored-size statistics the search driver merges into its row."""
        super().apply(cfg)
        return {}

    def cost(self, cfg, as_profiled=False):
        c = super().cost(cfg, as_profiled)
        weights = c["weights_mb"] * 1e6
        if self.layout.tied and self.effective(self.layout.lm_head, cfg.lm_head) != "bf16":
            rows, cols, _ = self.shapes[self.layout.lm_head]
            weights += 2 * rows * cols  # the bf16 embedding the lookup still reads
        return {"tpot_ms": c["latency_ms"], "latency_ms": c["latency_ms"], "tok_s": 1000 / c["latency_ms"],
                "memory_mb": weights / 1e6, "size_mb": weights / 1e6, "weights_gib": weights / (1 << 30),
                "bits_per_weight": c["bits_per_weight"], "int4_pct": c["int4_pct"], "mlp_kept_pct": c["mlp_kept_pct"],
                "n_removed_attn": c["n_removed_attn"], "n_removed_mlp": c["n_removed_mlp"]}


def grouped_decode_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query [B, H, t, D], key and value [B, Hkv, T, D], attention_mask sdpa's boolean [B, 1, t, T] (True: attend) or
    None. One new token (t = 1): grouped-query attention reading the cache in place. A prompt (t > 1): sdpa."""
    if query.shape[2] > 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    assert kwargs.get("sliding_window") is None and not dropout, "sliding windows and dropout are not supported"
    B, H, _, D = query.shape
    Hkv, T = key.shape[1], key.shape[2]
    scores = (query.reshape(B, Hkv, H // Hkv, D) @ key.transpose(-1, -2)).float() * (D ** -0.5 if scaling is None else scaling)
    if attention_mask is not None:
        mask = attention_mask.reshape(B, 1, 1, T)
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
    out = torch.softmax(scores, dim=-1).to(value.dtype) @ value  # [B, Hkv, H/Hkv, D]
    return out.reshape(B, 1, H, D), None  # [B, t, H, D], as transformers' attention functions return


AttentionInterface.register(GROUPED_IMPL, grouped_decode_attention)
AttentionMaskInterface.register(GROUPED_IMPL, sdpa_mask)


@contextlib.contextmanager
def attention_implementation(model, impl):
    """Run the model with the attention function registered as impl (None: leave it as it is)."""
    if impl is None:
        yield
        return
    configs = {id(c): c for c in (model.config, model.config.get_text_config())}.values()
    saved = [(c, c._attn_implementation) for c in configs]
    try:
        for c in configs:
            c._attn_implementation = impl
        yield
    finally:
        for c, v in saved:
            c._attn_implementation = v


def deployed_bytes(model):
    """Bytes of the deployed model's weights (parameters and buffers, a tied matrix once) and of its input embeddings."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    emb = model.get_input_embeddings().weight
    return {"weights_bytes": total, "embedding_bytes": emb.numel() * emb.element_size()}


def decode_timing(model, batch=1, prompt_tokens=512, gen_tokens=256, repeats=5, check_tokens=32, seed=0,
                  attention="grouped"):
    """TTFT, TPOT and tok/s of greedy decoding on this GPU (see the module docstring). Returns a dict."""
    assert attention in ("grouped", "model"), attention
    with attention_implementation(model, GROUPED_IMPL if attention == "grouped" else None):
        r = _decode_timing(model, batch, prompt_tokens, gen_tokens, repeats, check_tokens, seed)
    return {**r, "attention": attention}


@torch.no_grad()
def _decode_timing(model, batch, prompt_tokens, gen_tokens, repeats, check_tokens, seed):
    from transformers import StaticCache
    assert gen_tokens >= 2 and check_tokens <= gen_tokens - 1
    device = next(model.parameters()).device
    vocab = model.config.get_text_config().vocab_size
    prompt = torch.randint(0, vocab, (batch, prompt_tokens), generator=torch.Generator().manual_seed(seed)).to(device)
    positions = torch.arange(prompt_tokens, device=device)
    cache = StaticCache(config=model.config, max_cache_len=prompt_tokens + gen_tokens)
    ids = torch.zeros(batch, 1, dtype=torch.long, device=device)  # the static inputs of the graph
    pos = torch.zeros(1, dtype=torch.long, device=device)

    def prefill():
        """Rewind the cache and process the prompt: ids, pos and the cache then hold the first decode step's inputs."""
        for layer in cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.cumulative_length.zero_()  # where the cache writes next (a GPU counter: the graph advances it)
        logits = model(input_ids=prompt, past_key_values=cache, cache_position=positions, use_cache=True,
                       logits_to_keep=1).logits
        ids.copy_(logits[:, -1:].argmax(-1))
        pos.fill_(prompt_tokens)

    def step():
        logits = model(input_ids=ids, past_key_values=cache, cache_position=pos, use_cache=True, logits_to_keep=1).logits
        ids.copy_(logits[:, -1:].argmax(-1))
        pos.add_(1)

    torch.cuda.reset_peak_memory_stats(device)
    prefill()
    side = torch.cuda.Stream(device)  # warm-up before capture, as torch.cuda.graph requires
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream(device).wait_stream(side)
    prefill()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    def tokens(run_step):
        prefill()
        out = [ids.clone()]
        for _ in range(check_tokens):
            run_step()
            out.append(ids.clone())
        return torch.cat(out, 1)

    eager, graphed = tokens(step), tokens(graph.replay)
    matched = int((eager == graphed).all(0).cumprod(0).sum())  # leading positions equal in every sequence

    ttft, tpot = [], []
    for _ in range(repeats + 1):  # the first run warms up
        e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e[0].record()
        prefill()
        e[1].record()
        for _ in range(gen_tokens - 1):
            graph.replay()
        e[2].record()
        torch.cuda.synchronize(device)
        ttft.append(e[0].elapsed_time(e[1]))
        tpot.append(e[1].elapsed_time(e[2]) / (gen_tokens - 1))
    ttft, tpot = ttft[1:], tpot[1:]
    t = statistics.median(tpot)
    return {"batch": batch, "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens, "repeats": repeats,
            "ttft_ms": statistics.median(ttft), "tpot_ms": t, "tpot_ms_runs": tpot,
            "tok_s": 1000 / t, "tok_s_total": batch * 1000 / t,
            "e2e_ms": statistics.median(ttft) + (gen_tokens - 1) * t,
            "graph_matches_eager": f"{matched}/{check_tokens + 1}",
            "peak_bytes": torch.cuda.max_memory_allocated(device)}
