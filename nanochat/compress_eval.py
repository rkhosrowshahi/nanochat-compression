"""
Perplexity evaluation for the compression experiments: the calibration/test loss of a
causal LM, and the two standard test sets of the LLM compression literature
(GPTQ, AWQ, SparseGPT), cut into non-overlapping windows of seq_len tokens:
    WikiText-2: the whole test split, documents joined by blank lines
    C4:         the first 1100 validation documents joined by spaces, first n windows
                (C4 has no test split; GPTQ uses 256 windows of 2048 tokens)
"""

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


def consecutive_windows(ids, seq_len, n_seqs=-1):
    """Cut a token stream into non-overlapping seq_len windows; n_seqs = -1 keeps all."""
    n_all = len(ids) // seq_len
    n = n_all if n_seqs < 0 else min(n_seqs, n_all)
    assert n > 0, f"only {len(ids)} tokens, less than one window"
    return torch.tensor(ids[:n * seq_len], dtype=torch.long).view(n, seq_len)


def wikitext2_test(tokenizer, seq_len, n_seqs=-1):
    texts = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    return consecutive_windows(tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"], seq_len, n_seqs)


def c4_validation_test(tokenizer, seq_len, n_seqs=256):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for _, ex in zip(range(1100), ds)]
    return consecutive_windows(tokenizer(" ".join(texts), add_special_tokens=False)["input_ids"], seq_len, n_seqs)


def load_test_sets(tokenizer, seq_len, wikitext_seqs=-1, c4_seqs=256):
    """{name: windows} for the test sets that are not skipped (0 windows = skip)."""
    sets = {}
    if wikitext_seqs != 0:
        sets["wikitext2"] = wikitext2_test(tokenizer, seq_len, wikitext_seqs)
    if c4_seqs != 0:
        sets["c4"] = c4_validation_test(tokenizer, seq_len, c4_seqs)
    return sets
