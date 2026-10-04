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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

parser = argparse.ArgumentParser(description="final Pareto fronts of several runs in one plot")
parser.add_argument("--run-dirs", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None, help="legend label per run (default: the folder names)")
parser.add_argument("--out", type=str, default="compress_runs/final_fronts", help="output path without extension")
parser.add_argument("--title", type=str, default="Final Pareto fronts")
# categorical colors in a fixed order (validated palette: distinguishable with color-vision
# deficiency and in grayscale): blue, aqua, orange, then yellow, magenta, green, violet, red
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
parser.add_argument("--colors", nargs="+", default=COLORS, help="one color per run, in --run-dirs order")
parser.add_argument("--y", type=str, default="log", choices=["log", "linear"],
                    help="y-axis scale of the calibration loss: log (symlog, linear near 0) or linear")
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
x_min, details = float("inf"), []
assert len(args.run_dirs) <= len(args.colors), f"give --colors for all {len(args.run_dirs)} runs"
for d, label, marker, color in zip(args.run_dirs, labels, "osD^v<>", args.colors):
    front = pd.read_csv(os.path.join(d, "front.csv")).rename(columns={"loss_increase": "delta_loss"})
    last = front[front["gen"] == front["gen"].max()].sort_values("size_mb")
    hv = HV(ref_point=np.ones(2))(last[["delta_loss", "size_mb"]].to_numpy() / nadir)
    ax.plot(last["size_mb"], last["delta_loss"], marker=marker, ms=5, lw=1.5, color=color, label=label)
    details.append(f"(N={len(last)}, Gens={int(last['gen'].iloc[0])}, HV={hv:.4f})")
    print(f"{label}: generation {int(last['gen'].iloc[0])}, {len(last)} points, HV {hv:.5f}")
    x_min = min(x_min, last["size_mb"].min())
baseline_mb = configs[0]["baseline_size_mb"]
ax.plot(baseline_mb, 0, "D", color="black", label="Uncompressed")
if args.y == "log":
    ax.set_yscale("symlog", linthresh=0.01)  # linear near 0, log above
    ax.set_ylim(bottom=-0.005)
else:
    ax.set_ylim(bottom=-0.02 * ax.get_ylim()[1])
ax.set_xlim(0, 1.05 * baseline_mb)  # from 0 MB, as in compress_plot
ax.set_xlabel("Model size (MB)")
ax.set_ylabel(r"Calibration loss ($\Delta$)")
ax.set_title(args.title)
ax.grid(True, alpha=0.3)
# below the axes, never over the data; two columns, names | details, so the details line up
# whatever the width of each name (matplotlib ignores tabs and its fonts are proportional)
handles, names = ax.get_legend_handles_labels()
blank = Line2D([], [], linestyle="none")
ax.legend(handles + [blank] * len(handles), names + details + [""] * (len(handles) - len(details)), ncol=2,
          loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, columnspacing=0.0)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", dpi=120, bbox_inches="tight")
print(f"{len(args.run_dirs)} runs -> {args.out}.png/.pdf")
