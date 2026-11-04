"""
Re-check a finished kv2 run (scripts/compress_search_kv2.py) without searching again:
    1. time    the KV2 table measured again with the current decode kernel (compress_kv2.KERNEL_VERSION) and windows timed
               at aligned lengths, and the time of every evaluation in evals.csv recomputed with it. Loss, recall and
               memory do not depend on the table, so they are kept
    2. front   the feasible evaluations (the run's delta_loss and recall constraints) that no other one beats in delta_loss,
               time and memory together
    3. recall  the fastest and the smallest feasible solution under each of --limits (delta loss), and the uncompressed
               model, on --recall-seqs NEW facts planted in C4 validation text. The search chose its solutions on 64 facts
               in calibration text, so their recall there is biased upwards; this is the held-out check

Outputs in <run-dir>/recheck: kv2_table.json, evals_recosted.csv (latency_ms etc. recomputed, the run's in old_*),
front.csv, recall.csv and run.log. The KV2 table is cached per GPU as the search caches it, so a later search on the
same settings reuses it. The GPU should otherwise be idle while it is measured.

    python -m scripts.compress_kv2_recheck --run-dir compress_runs/qwen_kv2
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from nanochat import compress_eval, compress_hw, compress_kv2, compress_models
from scripts.compress_search_common import Tee

parser = argparse.ArgumentParser(description="Re-time a finished kv2 run with the current kernel and check its recall on new facts")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--recall-seqs", type=int, default=256, help="new facts, one per window")
parser.add_argument("--recall-seed", type=int, default=1, help="the search used --data-seed (0) on calibration text")
parser.add_argument("--recall-batch", type=int, default=8)
parser.add_argument("--c4-seqs", type=int, default=64, help="C4 validation windows the filler text comes from")
parser.add_argument("--seq-len", type=int, default=None, help="tokens per recall window (default: the run's --seq-len)")
parser.add_argument("--limits", type=str, default="0.01,0.02,0.05,0.1,0.2", help="delta-loss limits: the fastest and smallest under each are checked")
parser.add_argument("--out-dir", type=str, default=None, help="default <run-dir>/recheck")
args = parser.parse_args()
out_dir = args.out_dir or os.path.join(args.run_dir, "recheck")
os.makedirs(out_dir, exist_ok=True)
out = lambda name: os.path.join(out_dir, name)
log = open(out("run.log"), "a", encoding="utf-8", buffering=1)
sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.compress_kv2_recheck {' '.join(sys.argv[1:])}", flush=True)

run = json.load(open(os.path.join(args.run_dir, "config.json")))
a = SimpleNamespace(**run["args"])
seq_len = args.seq_len or a.seq_len
device = torch.device("cuda")
model, tokenizer, _ = compress_models.load(a.model, a.dtype, device)
L = compress_hw.Layout(model)
gpu = torch.cuda.get_device_name(device)
print(f"{a.model} on {gpu}; run {args.run_dir}", flush=True)

# --- 1. the KV2 table, with the settings the search caches it under

import triton  # noqa: E402

formats = [f for f in a.formats.split(",") if f]
mlp_keep = sorted({float(k) for k in a.mlp_keep.split(",")} | {1.0}, reverse=True)
keys, values = [x for x in a.kv_keys.split(",") if x], [x for x in a.kv_values.split(",") if x]
windows = sorted({float(w) for w in a.kv_windows.split(",")} | {1.0}, reverse=True)
settings = {"version": compress_hw.TABLE_VERSION, "kind": "KV2", "gpu": gpu, "torch": str(torch.__version__), "model": a.model,
            "batch": a.serve_batch, "context": a.context, "formats": formats, "mlp_keep": mlp_keep, "prompt_batch": a.prompt_batch,
            "keys": keys, "values": values, "windows": windows, "group": a.kv_group, "sinks": compress_kv2.SINKS,
            "triton": triton.__version__, "kernel_version": compress_kv2.KERNEL_VERSION}
cache_file = compress_eval.cache_path("latency", settings)
if os.path.exists(cache_file):
    kv2_table = torch.load(cache_file)
    print(f"KV2 table from {cache_file}", flush=True)
else:
    t0 = time.time()
    print(f"Measuring the KV2 table on {gpu}: batch {a.serve_batch}, {a.context} tokens of context", flush=True)
    kv2_table = {"settings": settings, **compress_kv2.build_kv2_table(L, keys, values, windows, a.serve_batch, a.prompt_batch, a.context,
                                                                       device, a.kv_group, log=lambda m: print(m, flush=True))}
    compress_eval.save_atomic(kv2_table, cache_file)
    print(f"...done in {time.time() - t0:.0f}s", flush=True)
json.dump(kv2_table, open(out("kv2_table.json"), "w"), indent=2)
old_table = json.load(open(os.path.join(args.run_dir, "kv2_table.json")))
print("Decode attention per layer at the whole context (us), the run's table -> now: " + ", ".join(
    f"{kv} {old_table['decode']['attention_us'][kv]['1']:.1f} -> {t['1']:.1f}" for kv, t in kv2_table["decode"]["attention_us"].items()), flush=True)

# --- the compressor of the run, with the new table

decode_table = json.load(open(os.path.join(args.run_dir, "latency_table.json")))
prompt_table = json.load(open(os.path.join(args.run_dir, "prompt_table.json")))
act_sq = torch.load(os.path.join(args.run_dir, "act_stats.pt"))["stats"]
if a.max_skip > 0:
    removal = json.load(open(os.path.join(args.run_dir, "removal.json")))
    removal = {k: removal[k] for k in ("attn", "mlp")}
else:
    removal = {k: [(i, 0.0) for i in range(L.n_layer)] for k in ("attn", "mlp")}
compressor = compress_kv2.KV2Compressor(
    model, L, decode_table, prompt_table, kv2_table, act_sq, removal, context=a.context, new_tokens=a.new_tokens, gen_tokens=a.gen_tokens,
    vram_gb=a.vram_gb, reserve=a.reserve, formats=formats, bands=a.bands, mlp_keep=mlp_keep, max_skip=a.max_skip,
    compress_embeddings=not a.no_embeddings, clip_ratios=[float(r) for r in a.clip_ratios.split(",")], offload=a.offload_originals)
assert [list(g) for g in compressor.genes] == [list(g) for g in run["genes"]], "the genes differ from the run's: other settings or code"
names = [n for n, _ in compressor.genes]

# --- time of every evaluation, recomputed

evals = pd.read_csv(os.path.join(args.run_dir, "evals.csv"))
TIMES = ("latency_ms", "decode_us", "prompt_us")
recost = [compressor.cost(compressor.config(tuple(int(r[n]) for n in names))) for _, r in evals[names].iterrows()]
for col in TIMES:
    evals[f"old_{col}"] = evals[col]
    evals[col] = [c[col] for c in recost]
base_cfg = compressor.config(compressor.uniform_genes("bf16"))
base = compressor.cost(base_cfg)
evals.to_csv(out("evals_recosted.csv"), index=False)
print(f"Uncompressed turn: {run['baseline']['latency_ms']:.0f} ms in the run, {base['latency_ms']:.0f} ms now; "
      f"{len(evals)} evaluations re-timed", flush=True)

# --- 2. the front of all evaluations

feasible = evals[(evals.delta_loss <= a.max_delta_loss) & (evals.recall_acc >= a.min_recall)].reset_index(drop=True)
F = feasible[["delta_loss", "latency_ms", "memory_mb"]].to_numpy()
front = feasible.iloc[NonDominatedSorting().do(F, only_non_dominated_front=True)].sort_values("latency_ms")
front.to_csv(out("front.csv"), index=False)
print(f"{len(feasible)} feasible evaluations, {len(front)} on the front of delta_loss, time and memory", flush=True)

# --- 3. recall on new facts

picks = {}
for limit in (float(x) for x in args.limits.split(",")):
    under = feasible[feasible.delta_loss <= limit]
    if len(under):
        for what, col in (("fastest", "latency_ms"), ("smallest", "memory_mb")):
            row = under.loc[under[col].idxmin()]
            picks.setdefault(row["key"], (row, []))[1].append(f"{what} <= {limit:g}")
c4 = compress_eval.load_test_sets(tokenizer, seq_len, 0, args.c4_seqs, cache_key=a.model)["c4"]
filler = [t for row in c4.tolist() for t in row]
tokens, answers = compress_kv2.recall_set(tokenizer, filler, args.recall_seqs, seq_len, args.recall_seed)
print(f"Recall on {args.recall_seqs} new facts in {seq_len}-token windows of C4 validation text, for {len(picks)} solutions "
      f"and the uncompressed model", flush=True)


def recall(cfg):
    compressor.apply(cfg)
    try:
        return compress_kv2.recall_score(model, tokens, answers, args.recall_batch)
    finally:
        compressor.restore()


ce, acc = recall(base_cfg)
rows = [{"picked": "uncompressed", "delta_loss": 0.0, "latency_ms": base["latency_ms"], "old_latency_ms": run["baseline"]["latency_ms"],
         "memory_mb": base["memory_mb"], "agents_fit": base["agents_fit"], "recall_search": run["baseline"].get("recall_acc"),
         "recall_new": acc, "recall_ce_new": ce, "config": compressor.describe(base_cfg), "key": ""}]
print(f"  uncompressed: {acc:.1%} recalled (answer cross-entropy {ce:.3f})", flush=True)
for key, (r, why) in sorted(picks.items(), key=lambda kv: kv[1][0]["latency_ms"]):
    t0 = time.time()
    ce, acc = recall(compressor.config(tuple(int(r[n]) for n in names)))
    rows.append({"picked": "; ".join(why), "delta_loss": r["delta_loss"], "latency_ms": r["latency_ms"], "old_latency_ms": r["old_latency_ms"],
                 "memory_mb": r["memory_mb"], "agents_fit": r["agents_fit"], "recall_search": r["recall_acc"], "recall_new": acc,
                 "recall_ce_new": ce, "config": r["config"], "key": key})
    print(f"  {'; '.join(why)}: delta {r['delta_loss']:+.4f}, turn {r['old_latency_ms']:.0f} -> {r['latency_ms']:.0f} ms, "
          f"{r['memory_mb']:.0f} MB | recall {r['recall_acc']:.0%} on the search's facts, {acc:.1%} on new ones "
          f"({time.time() - t0:.0f}s) | {r['config']}", flush=True)
pd.DataFrame(rows).to_csv(out("recall.csv"), index=False)
print(f"Done: {out_dir}", flush=True)
