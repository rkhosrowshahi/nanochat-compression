"""
Plot the final Pareto fronts of several compress_search runs in one figure.

The y-axis is the calibration delta_loss, so the runs should share the calibration batch
(same --calib-every 0 batch: seed, --calib-seqs, --seq-len); a warning is printed otherwise.
The legend gives each front's hypervolume with ONE normalization for all runs: each objective
divided by the worst value over the initial-front nadirs of all runs, reference (1, 1). (Each
run's own logged hypervolume uses its own nadir, so those are not comparable across runs.)

    python -m scripts.compress_compare \
        --run-dirs compress_runs/local_50_linear_k256_calib_fixed_8x2048 compress_runs/local_50_linear_k256_calib_fixed_8x2048_maxdelta1 \
        --labels "No constraint" "$\\Delta \\leq 1$" --out compress_runs/final_fronts_constraint
    -> <out>.png/.pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

parser = argparse.ArgumentParser(description="final Pareto fronts of several runs in one plot")
parser.add_argument("--run-dirs", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None, help="legend label per run (default: the folder names)")
parser.add_argument("--out", type=str, default="compress_runs/final_fronts", help="output path without extension")
parser.add_argument("--title", type=str, default="Final Pareto fronts")
args = parser.parse_args()
labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
assert len(labels) == len(args.run_dirs), "give one label per run"

CALIB_KEYS = ["calib_dataset", "calib_split", "data_seed", "seq_len", "calib_seqs", "calib_every"]
configs = [json.load(open(os.path.join(d, "config.json"))) for d in args.run_dirs]
calib = {tuple(c["args"].get(k) for k in CALIB_KEYS) for c in configs}
if len(calib) > 1 or any(c["args"].get("calib_every", 1) != 0 for c in configs):
    print("warning: the runs were not all scored on one fixed calibration batch, so their delta_loss "
          "values are not strictly comparable")

nadir = np.max([json.load(open(os.path.join(d, "normalization.json")))["nadir"] for d in args.run_dirs], axis=0)
print(f"hypervolume: objectives divided by the shared nadir {np.round(nadir, 4).tolist()}, reference (1, 1)")

fig, ax = plt.subplots(figsize=(8, 5))
x_min = float("inf")
for d, label, marker in zip(args.run_dirs, labels, "osD^v<>"):
    front = pd.read_csv(os.path.join(d, "front.csv")).rename(columns={"loss_increase": "delta_loss"})
    last = front[front["gen"] == front["gen"].max()].sort_values("size_mb")
    hv = HV(ref_point=np.ones(2))(last[["delta_loss", "size_mb"]].to_numpy() / nadir)
    ax.plot(last["size_mb"], last["delta_loss"], marker=marker, ms=5, lw=1.5,
            label=f"{label} (N={len(last)}, Gens={int(last['gen'].iloc[0])}, HV={hv:.4f})")
    print(f"{label}: generation {int(last['gen'].iloc[0])}, {len(last)} points, HV {hv:.5f}")
    x_min = min(x_min, last["size_mb"].min())
baseline_mb = configs[0]["baseline_size_mb"]
ax.plot(baseline_mb, 0, "D", color="black", label="Uncompressed")
ax.set_yscale("symlog", linthresh=0.01)  # linear near 0, log above
pad = 0.05 * (baseline_mb - x_min)
ax.set_xlim(x_min - pad, baseline_mb + pad)
ax.set_ylim(bottom=-0.005)
ax.set_xlabel("Model size (MB)")
ax.set_ylabel("Calibration loss")
ax.set_title(args.title)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", dpi=120)
print(f"{len(args.run_dirs)} runs -> {args.out}.png/.pdf")
