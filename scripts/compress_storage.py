"""
The optional storage step of the simplified mix (nanochat/compress_mix_simple.py), on the final front of a finished search of
scripts/compress_search_mix_simple.py or scripts/compress_search_hw.py. Every solution is tried as it is and with its bf16 matrices on
a shared grid of each --k. That changes its stored size and its loss, and it runs as before, so its time and memory stay.

Outputs in <run-dir>:
    storage.csv           every solution at every k (k 0 = as it is); rerunning continues it
    storage_front.csv     the rows no other row beats on delta_loss and stored size
    storage_scores.json   up to --max-points of that front on the test sets

    python -m scripts.compress_storage --run-dir compress_runs/qwen_mix_simple
    python -m scripts.compress_storage --run-dir compress_runs/qwen_hw_3obj_p50_g100
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from nanochat import compress_eval, compress_hw, compress_mix, compress_models
from nanochat.compress_mix_simple import STORAGE_K, has_bf16, shared_storage, storage_front

parser = argparse.ArgumentParser(description="Share the bf16 matrices of a finished front, for the stored size")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--k", type=str, default=",".join(map(str, STORAGE_K)), help="grid levels to try")
parser.add_argument("--max-delta-loss", type=float, default=1.0, help="rows above this stay out of the storage front")
parser.add_argument("--max-points", type=int, default=6, help="solutions of the storage front scored on the test sets (0 = none)")
args = parser.parse_args()
path = lambda name: os.path.join(args.run_dir, name)

config = json.load(open(path("config.json")))
run = config["args"]
front = pd.read_csv(path("front.csv"))
front = front[front["gen"] == front["gen"].max()].drop_duplicates("key")
print(f"{args.run_dir}: {len(front)} solutions of generation {front['gen'].iloc[0]}", flush=True)

assert torch.cuda.is_available(), "the int4 clipping statistics live on a CUDA GPU in these searches"
device = torch.device("cuda")
model, tokenizer, _ = compress_models.load(run["model"], run["dtype"], device)
layout = compress_hw.Layout(model)
calib = torch.load(path("calib_pool.pt"))["tokens"][:run["calib_seqs"]]
act_sq = torch.load(path("act_stats.pt"))["stats"]
if os.path.exists(path("removal.json")):
    removal = json.load(open(path("removal.json")))
else:  # nothing was removed: the order does not matter
    removal = {kind: [(i, 0.0) for i in range(layout.n_layer)] for kind in ("attn", "mlp")}
table = json.load(open(path("latency_table.json")))  # only for which kernels run which shapes: the costs come from front.csv
compressor = compress_mix.MixCompressor(model, layout, table, table, act_sq, {k: removal[k] for k in ("attn", "mlp")}, sharing=False,
                                        formats=[f for f in run["formats"].split(",") if f], bands=run["bands"],
                                        mlp_keep=sorted({float(k) for k in run["mlp_keep"].split(",")} | {1.0}, reverse=True),
                                        max_skip=run["max_skip"], compress_embeddings=not run["no_embeddings"],
                                        clip_ratios=[float(r) for r in run["clip_ratios"].split(",")], offload=run["offload_originals"])
assert [list(g) for g in compressor.genes] == [list(g) for g in config["genes"]], "this run's genes are not those of a deployed-formats search"
gene_names = [name for name, _ in compressor.genes]
eval_loss = lambda tokens: compress_eval.eval_loss(model, tokens, run["eval_batch_size"])

baseline_loss = eval_loss(calib)
FIELDS = ["source_key", "k", "config", "loss", "delta_loss", "size_mb", "shared_pct", "k_merged", "latency_ms", "memory_mb", "weights_mb", "seconds"]
rows = []
if os.path.exists(path("storage.csv")):
    with open(path("storage.csv"), newline="") as f:
        rows = [{n: (r[n] if n in ("source_key", "config") else float(r[n])) for n in FIELDS} for r in csv.DictReader(f)]
    print(f"Loaded {len(rows)} previous rows", flush=True)
done = {(r["source_key"], int(r["k"])) for r in rows}
new_file = not os.path.exists(path("storage.csv"))
out = open(path("storage.csv"), "a", newline="")
writer = csv.DictWriter(out, fieldnames=FIELDS)
if new_file:
    writer.writeheader()


def configuration(row, k):
    base = compressor.config(compressor.decode([row[n] for n in gene_names]))
    return base if k == 0 else shared_storage(base, k)


ks = [int(k) for k in args.k.split(",") if k]
t_all = time.time()
for _, row in front.iterrows():
    source = compressor.config(compressor.decode([row[n] for n in gene_names]))
    for k in [0] + (ks if has_bf16(source) else []):
        if (row["key"], k) in done:
            continue
        t0 = time.time()
        cfg = configuration(row, k)
        try:
            stored = compressor.apply(cfg)
            loss = eval_loss(calib)
        finally:
            compressor.restore()
        r = {"source_key": row["key"], "k": k, "config": compressor.describe(cfg), "loss": loss, "delta_loss": loss - baseline_loss, **stored,
             "latency_ms": row["latency_ms"], "memory_mb": row["memory_mb"], "weights_mb": row["weights_mb"], "seconds": round(time.time() - t0, 2)}
        rows.append(r)
        writer.writerow(r)
        out.flush()
        print(f"  k {k:>3} | delta {r['delta_loss']:+.4f} stored {r['size_mb']:.0f} MB deployed {r['weights_mb']:.0f} MB turn/step {r['latency_ms']:.2f} ms | {r['config']}", flush=True)
out.close()
print(f"{len(rows)} rows, this time in {time.time() - t_all:.0f}s", flush=True)

best = storage_front([r for r in rows if r["delta_loss"] <= args.max_delta_loss and math.isfinite(r["delta_loss"])])
pd.DataFrame(best).to_csv(path("storage_front.csv"), index=False)
print(f"\nStorage front ({len(best)} points):\n{'delta':>8} {'k':>4} {'stored':>7} {'deployed':>8} {'turn/step':>9}  configuration", flush=True)
for r in best:
    print(f"{r['delta_loss']:+8.4f} {int(r['k']):4d} {r['size_mb']:5.0f}MB {r['weights_mb']:6.0f}MB {r['latency_ms']:7.2f}ms  {r['config']}", flush=True)

if args.max_points <= 0 or (run["wikitext_seqs"] == 0 and run["c4_test_seqs"] == 0):
    raise SystemExit
scored = best if len(best) <= args.max_points else [best[j] for j in np.linspace(0, len(best) - 1, args.max_points).round().astype(int)]
test_sets = compress_eval.load_test_sets(tokenizer, run["test_seq_len"], run["wikitext_seqs"], run["c4_test_seqs"], cache_key=run["model"])
settings = {"seq_len": run["test_seq_len"], "wikitext_seqs": run["wikitext_seqs"], "c4_test_seqs": run["c4_test_seqs"]}
score_path = path("storage_scores.json")
results = json.load(open(score_path)) if os.path.exists(score_path) else None
if results is None or results["settings"] != settings:
    results = {"settings": settings, "baseline": {}, "pareto": []}
if not results["baseline"]:
    for name, tokens in test_sets.items():
        results["baseline"][f"{name}_loss"] = loss = eval_loss(tokens)
        results["baseline"][f"{name}_ppl"] = math.exp(loss)
scored_keys = {(p["source_key"], int(p["k"])) for p in results["pareto"]}
by_key = front.set_index("key")
for r in scored:
    if (r["source_key"], int(r["k"])) in scored_keys:
        continue
    entry = dict(r)
    try:
        compressor.apply(configuration(by_key.loc[r["source_key"]], int(r["k"])))
        for name, tokens in test_sets.items():
            entry[f"{name}_loss"] = loss = eval_loss(tokens)
            entry[f"{name}_ppl"] = math.exp(min(loss, 700.0))
    finally:
        compressor.restore()
    results["pareto"].append(entry)
    with open(score_path + ".tmp", "w") as f:
        json.dump(results, f, indent=2)
    os.replace(score_path + ".tmp", score_path)
    print(f"  k {int(r['k'])} stored {r['size_mb']:.0f} MB: " + ", ".join(f"{n} ppl {entry[f'{n}_ppl']:.2f} (base {results['baseline'][f'{n}_ppl']:.2f})" for n in test_sets), flush=True)
print(f"Done: {path('storage.csv')}, {path('storage_front.csv')}, {score_path}", flush=True)
