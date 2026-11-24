"""
Perplexity evaluation for the compression experiments: the calibration/test loss of a
causal LM, and the two standard test sets of the LLM compression literature
(GPTQ, AWQ, SparseGPT), cut into non-overlapping windows of seq_len tokens:
    WikiText-2: the whole test split, documents joined by blank lines
    C4:         the first 1100 validation documents joined by spaces, first n windows
                (C4 has no test split; GPTQ uses 256 windows of 2048 tokens)

Tokenized data is cached in $COMPRESS_CACHE (default ~/.cache/nanochat-compress), shared by every
run directory on the machine, so later runs need neither the download nor Hugging Face being up.
"""

import hashlib
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset


@torch.no_grad()
def eval_loss(model, tokens, batch_size=1):
    """Mean next-token cross-entropy (nats) over all positions of all sequences."""
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, tokens.size(0), batch_size):
        x = tokens[i:i + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False).logits[:, :-1]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        total += nll.item()
        count += x[:, 1:].numel()
    return total / count


@torch.no_grad()
def reference_hidden(model, tokens, batch_size=1):
    """What eval_kld compares against: the uncompressed model's input to its output head at every position but the
    last (bf16, CPU: N * (T-1) * hidden values, 50 MB for 8 windows of 2048 tokens at hidden size 1536), and a copy of
    the head's weight and bias. With these, eval_kld recomputes the reference's whole next-token distribution with
    one matrix product, where storing it would take N * T * vocab floats (5 GB for a 150k vocabulary). The logits must
    be the head's output as it is (HF Llama/Qwen2): checked here, since a model that changes them after its head (e.g.
    nanochat's softcap) would be compared against the wrong distribution."""
    device = next(model.parameters()).device
    head = model.get_output_embeddings()
    captured = []
    hook = head.register_forward_hook(lambda m, inp, out: captured.append((inp[0], out)))
    hidden = []
    try:
        for i in range(0, tokens.size(0), batch_size):
            captured.clear()
            logits = model(input_ids=tokens[i:i + batch_size].to(device), use_cache=False).logits
            (h, out), = captured
            assert out.shape == logits.shape and torch.equal(out, logits),                 "the model's logits are not its output head's output: eval_kld cannot recompute them"
            hidden.append(h[:, :-1].cpu())
    finally:
        hook.remove()
    bias = getattr(head, "bias", None)
    return {"hidden": torch.cat(hidden), "weight": head.weight.detach().clone(),
            "bias": None if bias is None else bias.detach().clone()}


@torch.no_grad()
def eval_kld(model, tokens, reference, batch_size=1, chunk=512):
    """KL(reference || model) per next-token position over the whole vocabulary, the measure of llama.cpp's
    --kl-divergence and of Unsloth's quant benchmarks: sum_v p_v (log p_v - log q_v), with p the uncompressed model's
    distribution, recomputed from reference_hidden, and q this model's. Returns the mean, median and 99th percentile
    over positions, the share of positions where both models' most likely token is the same (top1_agree, llama.cpp's
    "same top p"), and the mean next-token loss (nats) from the same forward pass."""
    device = next(model.parameters()).device
    weight, bias = reference["weight"].to(device), reference["bias"]
    bias = None if bias is None else bias.to(device)
    klds, agree, nll, count = [], 0, 0.0, 0
    for i in range(0, tokens.size(0), batch_size):
        x = tokens[i:i + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False).logits[:, :-1]
        hidden = reference["hidden"][i:i + x.size(0)]
        for t in range(0, logits.size(1), chunk):
            logq = torch.log_softmax(logits[:, t:t + chunk].float(), -1)
            h = hidden[:, t:t + chunk].to(device=device, dtype=weight.dtype)
            logp = torch.log_softmax(F.linear(h, weight, bias).float(), -1)
            klds.append((logp.exp() * (logp - logq)).sum(-1).clamp_min(0).flatten().cpu())
            agree += (logq.argmax(-1) == logp.argmax(-1)).sum().item()
            nll -= logq.gather(-1, x[:, t + 1:t + 1 + logq.size(1), None]).sum().item()
            count += logq.size(0) * logq.size(1)
    kld = torch.cat(klds)
    return {"kld": kld.mean().item(), "kld_median": kld.median().item(), "kld_p99": kld.quantile(0.99).item(),
            "top1_agree": agree / count, "loss": nll / count}


def consecutive_windows(ids, seq_len, n_seqs=-1):
    """Cut a token stream into non-overlapping seq_len windows; n_seqs = -1 keeps all."""
    n_all = len(ids) // seq_len
    n = n_all if n_seqs < 0 else min(n_seqs, n_all)
    assert n > 0, f"only {len(ids)} tokens, less than one window"
    return torch.tensor(ids[:n * seq_len], dtype=torch.long).view(n, seq_len)


def wikitext2_test(tokenizer, seq_len, n_seqs=-1):
    texts = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    return consecutive_windows(tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"], seq_len, n_seqs)


CODE_DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"  # 50,661 Python instructions with executed solutions
CODE_TEST_DOCS = 2000  # held out for test, never in a calibration window
CODE_SPLIT_SEED = 1234  # the fixed permutation that separates test from calibration documents


def code_instruct_windows(tokenizer, seq_len, n_seqs, part, seed=0):
    """Windows of seq_len tokens of coding conversations, each document an instruction (user) and its solution
    (assistant) in the model's chat template, documents joined back to back as an Instruct model reads a session.
    part "test": the CODE_TEST_DOCS held-out documents in a fixed order (n_seqs = -1 keeps every window);
    part "calib": the other documents, in an order drawn with `seed`, so calibration batches never contain test text."""
    assert part in ("calib", "test"), part
    assert hasattr(tokenizer, "apply_chat_template"), "code windows need a tokenizer with a chat template"
    ds = load_dataset(CODE_DATASET, split="train")
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(CODE_SPLIT_SEED)).tolist()
    docs = order[:CODE_TEST_DOCS] if part == "test" else order[CODE_TEST_DOCS:]
    if part == "calib":
        docs = [docs[i] for i in torch.randperm(len(docs), generator=torch.Generator().manual_seed(seed)).tolist()]
    ids, need = [], None if n_seqs < 0 else n_seqs * seq_len
    for d in docs:
        ex = ds[d]
        text = tokenizer.apply_chat_template([{"role": "user", "content": ex["instruction"]},
                                              {"role": "assistant", "content": ex["response"]}], tokenize=False)
        ids += tokenizer(text, add_special_tokens=False)["input_ids"]
        if need is not None and len(ids) >= need:
            break
    return consecutive_windows(ids, seq_len, n_seqs)


def c4_validation_test(tokenizer, seq_len, n_seqs=256):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for _, ex in zip(range(1100), ds)]
    return consecutive_windows(tokenizer(" ".join(texts), add_special_tokens=False)["input_ids"], seq_len, n_seqs)


CACHE_DIR = os.environ.get("COMPRESS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "nanochat-compress")


def cache_path(kind, key):
    """File in the shared cache for data identified by the dict `key` (e.g. dataset, seed, model)."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{kind}_{digest}.pt")


def save_atomic(obj, path):
    """torch.save that never leaves a half-written file (runs may share the cache)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_test_sets(tokenizer, seq_len, wikitext_seqs=-1, c4_seqs=256, cache_key=None):
    """{name: windows} for the test sets that are not skipped (0 windows = skip). With cache_key
    (anything that identifies the tokenizer, e.g. the model id) they are cached once tokenized."""
    path = cache_path("test", {"tokenizer": cache_key, "seq_len": seq_len, "wikitext2": wikitext_seqs,
                               "c4": c4_seqs}) if cache_key else None
    if path and os.path.exists(path):
        return torch.load(path)
    sets = {}
    if wikitext_seqs != 0:
        sets["wikitext2"] = wikitext2_test(tokenizer, seq_len, wikitext_seqs)
    if c4_seqs != 0:
        sets["c4"] = c4_validation_test(tokenizer, seq_len, c4_seqs)
    if path and sets:
        save_atomic(sets, path)
    return sets
