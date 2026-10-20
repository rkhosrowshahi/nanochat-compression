"""
Check a compress_search_hw run against the real deployed models: build selected final-front solutions
(int4 matrices on PyTorch's tinygemm kernel, MLPs cut to the kept neurons, removed sublayers dropped)
and measure on this GPU
    gpu_ms      summed GPU kernel time of one decode-shaped forward (latency-batch sequences x 1 token,
                no KV cache), from the profiler: what a runtime without launch overhead sees
    eager_ms    wall-clock time of that forward in eager PyTorch, Python and kernel launches included
    real_loss   calibration loss with the real kernels, against fake_loss (the search's fake-quantized model)
and compare gpu_ms with the latency table's prediction for the same shape (context 1: no attention over
a cache). The uncompressed model is measured first, as the reference. Results in <run-dir>/validation.json.

    python -m scripts.compress_hw_validate --run-dir compress_runs/qwen_hw_3obj --points 6
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import pandas as pd
import torch

from nanochat import compress_eval, compress_hw, compress_models

parser = argparse.ArgumentParser(description="measure deployed compress_search_hw solutions against the predictions")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--points", type=int, default=6, help="final-front solutions, spread evenly by predicted latency")
parser.add_argument("--val-seqs", type=int, default=2, help="calibration windows for the fake vs real loss")
parser.add_argument("--repeats", type=int, default=20, help="timed forwards for the eager wall-clock")
args = parser.parse_args()

run = lambda name: os.path.join(args.run_dir, name)
search = json.load(open(run("config.json")))["args"]
table = json.load(open(run("latency_table.json")))
device = torch.device("cuda")
if table["gpu"] != torch.cuda.get_device_name(device):
    print(f"WARNING: the latency table was measured on {table['gpu']}, this is {torch.cuda.get_device_name(device)}")
act_sq = torch.load(run("act_stats.pt"))["stats"]
removal = json.load(open(run("removal.json")))
tokens = torch.load(run("calib_pool.pt"))["tokens"][:args.val_seqs]

front = pd.read_csv(run("front.csv"))
front = front[front["gen"] == front["gen"].max()].sort_values("latency_ms")
picked = front.iloc[np.unique(np.linspace(0, len(front) - 1, min(args.points, len(front))).round().astype(int))]


def load():
    model, _, _ = compress_models.load(search["model"], search["dtype"], device)
    layout = compress_hw.Layout(model)
    comp = compress_hw.HWCompressor(
        model, layout, table, act_sq, {k: removal[k] for k in ("attn", "mlp")}, formats=search["formats"].split(","),
        bands=search["bands"], mlp_keep=[float(k) for k in search["mlp_keep"].split(",")], max_skip=search["max_skip"],
        compress_embeddings=not search["no_embeddings"], clip_ratios=[float(r) for r in search["clip_ratios"].split(",")],
        offload=search["offload_originals"])
    return model, comp


def eager_ms(model, x):
    for _ in range(3):
        model(input_ids=x, use_cache=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        model(input_ids=x, use_cache=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


x = torch.zeros(table["batch"], 1, dtype=torch.long, device=device)
results = []
print(f"{'':>13} {'predicted':>10} {'measured':>9} {'error':>7} {'eager':>8} {'fake loss':>10} {'real loss':>10}  configuration")
for label, genes in [("uncompressed", None)] + [(f"front #{i}", tuple(int(r[n]) for n, _ in json.load(open(run("config.json")))["genes"]))
                                                 for i, (_, r) in zip(picked.index, picked.iterrows())]:
    model, comp = load()
    genes = genes or comp.uniform_genes("bf16")
    cfg = comp.config(genes)
    with torch.no_grad():
        comp.apply(cfg)
        fake = compress_eval.eval_loss(model, tokens)
        comp.restore()
        comp.materialize(cfg)
        real = compress_eval.eval_loss(model, tokens)
        gpu = compress_hw.kernel_us(model, x) / 1000
        eager = eager_ms(model, x)
    predicted = comp.cost(cfg, as_profiled=True)["latency_ms"]
    results.append({"label": label, "config": comp.describe(cfg), "genes": list(genes), "predicted_gpu_ms": predicted,
                    "measured_gpu_ms": gpu, "eager_ms": eager, "fake_loss": fake, "real_loss": real,
                    "predicted_ms_at_search_context": comp.cost(cfg)["latency_ms"], "memory_mb": comp.cost(cfg)["memory_mb"]})
    print(f"{label:>13} {predicted:8.2f}ms {gpu:7.2f}ms {100 * (predicted - gpu) / gpu:+6.1f}% {eager:6.2f}ms "
          f"{fake:10.4f} {real:10.4f}  {comp.describe(cfg)}", flush=True)
    del model, comp
    torch.cuda.empty_cache()

base = results[0]
for r in results:
    r["predicted_speedup"] = base["predicted_gpu_ms"] / r["predicted_gpu_ms"]
    r["measured_gpu_speedup"] = base["measured_gpu_ms"] / r["measured_gpu_ms"]
    r["eager_speedup"] = base["eager_ms"] / r["eager_ms"]
print("\nspeedup over the uncompressed model (predicted / measured GPU time / eager wall-clock):")
for r in results[1:]:
    print(f"  {r['label']:>10}: {r['predicted_speedup']:.2f}x / {r['measured_gpu_speedup']:.2f}x / {r['eager_speedup']:.2f}x")
with open(run("validation.json"), "w") as f:
    json.dump({"gpu": torch.cuda.get_device_name(device), "batch": table["batch"], "val_seqs": args.val_seqs,
               "points": results}, f, indent=2)
print(f"Results in {run('validation.json')}")
