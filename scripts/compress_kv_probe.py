"""
Probe for the next KV search (after compress_runs/qwen_kv): which cache settings are worth genes. Each setting is written
into the model with hooks (as nanochat/compress_kv.py does) and scored on
    calib    calibration loss minus the uncompressed model's (a run's calib_pool.pt)
    recall   a fact ("The access code for project 7 is 48213.") planted at a random depth of a calibration window and asked
             for at its end: cross-entropy of the answer tokens and exact-match accuracy. Calibration loss hardly sees
             what a window forgets; this does.
Settings:
    keys and values per token (as qwen_kv) or keys per channel in groups of --group tokens (as KIVI), at 8, 4 and 2 bits
    windows that keep the first token visible (an attention sink, as StreamingLLM), against plain ones
    windows mixed with full-context layers (as Gemma 3 and gpt-oss)
Windows are fractions of each window of --seq-len tokens, as in the search: a proxy of the same fraction of the context.
Per-channel keys use scales of their whole group, so a token reads scales that depend on the up to group-1 tokens after it in
the group; a deployed cache keeps an unfinished group in bf16 instead.

Outputs in <out-dir>: results.csv (one row per setting) and run.log.

    python -m scripts.compress_kv_probe --calib compress_runs/qwen_kv/calib_pool.pt --out-dir compress_runs/kv_probe
"""

import argparse
import csv
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from nanochat import compress_hw, compress_models
from nanochat.compress_kv import fake_quantize, window_tokens

parser = argparse.ArgumentParser(description="Probe KV cache precisions and windows on calibration loss and planted-fact recall")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--calib", type=str, default="compress_runs/qwen_kv/calib_pool.pt", help="a search run's calib_pool.pt")
parser.add_argument("--calib-seqs", type=int, default=8)
parser.add_argument("--recall-seqs", type=int, default=64)
parser.add_argument("--group", type=int, default=64, help="tokens per scale of the per-channel keys")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out-dir", type=str, default="compress_runs/kv_probe")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(sys.stdout, open(os.path.join(args.out_dir, "run.log"), "a", encoding="utf-8"))
print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.compress_kv_probe {' '.join(sys.argv[1:])}", flush=True)

device = torch.device("cuda")
model, tokenizer, _ = compress_models.load(args.model, "bfloat16", device)
L = compress_hw.Layout(model)
calib = torch.load(args.calib)["tokens"][:args.calib_seqs]
T = calib.size(1)
print(f"{torch.cuda.get_device_name(device)}, {L.n_layer} layers, {calib.size(0)} x {T} calibration tokens from {args.calib}", flush=True)

# --- precisions

LEVELS = {"int8": 127, "int4": 7, "int2": 1}


def per_token(x, precision):
    """One scale per token and KV head (as qwen_kv; int2 added)."""
    if precision in ("int8", "int4"):
        return fake_quantize(x, precision, L.n_kv_head)
    v = x.float().view(*x.shape[:-1], L.n_kv_head, -1)
    s = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / LEVELS[precision]
    return ((v / s).round() * s).to(x.dtype).view_as(x)


def per_channel(x, precision):
    """One scale per channel and group of args.group tokens."""
    B, n, C = x.shape
    v = F.pad(x.float(), (0, 0, 0, (-n) % args.group)).view(B, -1, args.group, C)
    s = v.abs().amax(2, keepdim=True).clamp_min(1e-12) / LEVELS[precision]
    return ((v / s).round() * s).view(B, -1, C)[:, :n].to(x.dtype)


def quantize(x, kind):
    """kind: bf16, int8/int4/int2 (per token) or int8c/int4c/int2c (per channel)."""
    if kind == "bf16":
        return x
    return per_channel(x, kind[:-1]) if kind.endswith("c") else per_token(x, kind)


# --- hooks


def apply(key="bf16", value="bf16", windows=None, sinks=1):
    """key/value: one kind for all layers; windows: a fraction per layer (None: all 1); sinks: first tokens always visible."""
    hooks = []
    for i, blk in enumerate(L.blocks):
        _, k_proj, v_proj = blk["qkv"]
        for name, kind in ((k_proj, key), (v_proj, value)):
            if kind != "bf16":
                hooks.append(L.module(name).register_forward_hook(lambda m, a, out, kind=kind: quantize(out, kind)))
        w = 1.0 if windows is None else windows[i]
        if w < 1:
            def narrow(module, a, kwargs, w=w):
                x = kwargs["hidden_states"] if "hidden_states" in kwargs else a[0]
                n = x.shape[1]
                causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
                keep = causal & ~causal.tril(-window_tokens(w, n))
                keep[:, :sinks] = True
                keep &= causal
                mask = kwargs.get("attention_mask")
                if mask is None:
                    kwargs["attention_mask"] = keep[None, None]
                elif mask.dtype == torch.bool:
                    kwargs["attention_mask"] = mask & keep
                else:
                    kwargs["attention_mask"] = mask.masked_fill(~keep, torch.finfo(mask.dtype).min)
                return a, kwargs
            hooks.append(getattr(L.module(L.layers[i]), L.attn_attr).register_forward_pre_hook(narrow, with_kwargs=True))
    return hooks


# --- scores

rng = random.Random(args.seed)
filler = [t for row in calib for t in row.tolist()]
recall_set = []
for j in range(args.recall_seqs):
    code = str(rng.randint(10000, 99999))
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    fact, ask, answer = enc(f" The access code for project {j} is {code}. Remember it."), \
        enc(f" What is the access code for project {j}? The access code for project {j} is"), enc(f" {code}")
    body = T - len(fact) - len(ask) - len(answer)
    start = rng.randint(0, len(filler) - body - 1)
    at = int(rng.uniform(0.0, 0.9) * body)
    text = filler[start:start + body]
    recall_set.append((torch.tensor(text[:at] + fact + text[at:] + ask + answer), len(answer)))


@torch.no_grad()
def calib_loss():
    total = 0.0
    for row in calib:
        ids = row[None].to(device)
        total += F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item()
    return total / len(calib)


@torch.no_grad()
def recall():
    ce, hits = 0.0, 0
    for ids, n in recall_set:
        x = ids[None].to(device)
        logits = model(x).logits[0, -n - 1:-1].float()
        ce += F.cross_entropy(logits, x[0, -n:]).item()
        hits += bool((logits.argmax(-1) == x[0, -n:]).all())
    return ce / len(recall_set), hits / len(recall_set)


base_loss = calib_loss()
rows = []
out = open(os.path.join(args.out_dir, "results.csv"), "w", newline="")
writer = csv.DictWriter(out, fieldnames=["group", "setting", "delta_loss", "recall_ce", "recall_acc", "seconds"])
writer.writeheader()


def run(group, setting, **kw):
    t0 = time.time()
    hooks = apply(**kw)
    try:
        loss = calib_loss()
        ce, acc = recall()
    finally:
        for h in hooks:
            h.remove()
    r = {"group": group, "setting": setting, "delta_loss": round(loss - base_loss, 5), "recall_ce": round(ce, 4),
         "recall_acc": round(acc, 4), "seconds": round(time.time() - t0, 1)}
    writer.writerow(r)
    out.flush()
    print(f"[{group}] {setting:<46} delta {r['delta_loss']:+.4f} | recall CE {ce:.3f}, {acc:.0%} right ({r['seconds']:.0f}s)", flush=True)


print(f"Baseline calibration loss {base_loss:.4f}; {len(recall_set)} recall windows", flush=True)
n = L.n_layer
band = lambda per_band: [per_band[i * len(per_band) // n] for i in range(n)]
run("base", "uncompressed")
for kind in ("int8", "int8c", "int4", "int4c", "int2", "int2c"):
    run("keys", f"keys {kind}", key=kind)
for kind in ("int8", "int4", "int2"):
    run("values", f"values {kind}", value=kind)
for k, v in (("int8", "int8"), ("int4c", "int8"), ("int4c", "int4"), ("int2c", "int4")):
    run("keys+values", f"keys {k}, values {v}", key=k, value=v)
for w in (0.75, 0.25):
    run("sink", f"window {w}, plain", windows=[w] * n, sinks=0)
for w in (0.75, 0.5, 0.25, 0.125, 0.0625):
    run("window", f"window {w} + first token", windows=[w] * n)
for k, v in (("int8", "int8"), ("int4c", "int4")):
    run("window+precision", f"window 0.25 + first token, keys {k}, values {v}", windows=[0.25] * n, key=k, value=v)
run("mixed", "0.25, band 0 full", windows=band([1.0, 0.25, 0.25, 0.25]))
run("mixed", "0.25, band 3 full", windows=band([0.25, 0.25, 0.25, 1.0]))
run("mixed", "0.125, band 2 full", windows=band([0.125, 0.125, 1.0, 0.125]))
for w in (0.125, 0.0625):
    run("mixed", f"every 4th layer full, the rest {w}", windows=[1.0 if i % 4 == 3 else w for i in range(n)])
    run("mixed", f"every 6th layer full, the rest {w}", windows=[1.0 if i % 6 == 5 else w for i in range(n)])
print(f"Done: {os.path.join(args.out_dir, 'results.csv')}", flush=True)
