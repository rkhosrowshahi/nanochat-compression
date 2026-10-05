"""
Plot the final Pareto fronts of several compress_search runs in one figure.

The y-axis is the calibration delta_loss, so the runs should share the calibration batch
(same --calib-every 0 batch: seed, --calib-seqs, --seq-len); a warning is printed otherwise.
The legend gives each front's hypervolume with ONE normalization for all runs: each objective
divided by the worst value over the initial-front nadirs of all runs, reference (1, 1). (Each
run's own logged hypervolume uses its own nadir, so those are not comparable across runs.)

    python -m scripts.compress_compare \
        --run-dirs compress_runs/local_50_linear_k256_calib_fixed_8x2048 compress_runs/local_50_linear_k256_calib_fixed_8x2048_maxdelta1 \
        --labels "No constraint" "$\\Delta \\leq 1$" --out compress_runs/comparisons/final_fronts_constraint
    -> <out>.png/.pdf

--metric wikitext2 | c4 plots the test scores of the final front instead (WikiText-2 test or C4
validation perplexity, from the run's test_scores.json, or pareto.json if it has none), with the
uncompressed model's perplexity as a dashed line. Works for one run as well:
    python -m scripts.compress_compare --run-dirs <run> --metric wikitext2 --out <run>/test_front_wikitext2
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

parser = argparse.ArgumentParser(description="final Pareto fronts of several runs in one plot")
parser.add_argument("--run-dirs", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None, help="legend label per run (default: the folder names)")
parser.add_argument("--out", type=str, default="compress_runs/comparisons/final_fronts", help="output path without extension")
parser.add_argument("--title", type=str, default=None, help="default: by metric")
parser.add_argument("--metric", type=str, default="calib", choices=["calib", "wikitext2", "c4"],
                    help="y-axis: calibration delta_loss of the final front, or its test perplexity on "
                         "WikiText-2 test / C4 validation")
# categorical colors in a fixed order (validated palette: distinguishable with color-vision
# deficiency and in grayscale): blue, aqua, orange, then yellow, magenta, green, violet, red
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
parser.add_argument("--colors", nargs="+", default=COLORS, help="one color per run, in --run-dirs order")
parser.add_argument("--y", type=str, default="log", choices=["log", "linear"],
                    help="y-axis scale: log (for the calibration loss symlog, linear near 0) or linear")
args = parser.parse_args()
labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
assert len(labels) == len(args.run_dirs), "give one label per run"
METRIC_NAME = {"wikitext2": "WikiText-2 test perplexity", "c4": "C4 validation perplexity"}
if args.title is None:
    args.title = "Final Pareto fronts" if args.metric == "calib" else f"Final Pareto fronts: {METRIC_NAME[args.metric]}"


def test_scores(run_dir):
    """(baseline, DataFrame of the scored front) from test_scores.json (compress_score) or pareto.json."""
    for name in ("test_scores.json", "pareto.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            results = json.load(open(path))
            df = pd.DataFrame(results["pareto"])
            if f"{args.metric}_ppl" in df.columns:
                return results["baseline"], df.sort_values("size_mb")
    raise SystemExit(f"{run_dir}: no {args.metric} test scores (run python -m scripts.compress_score --run-dir {run_dir})")

CALIB_KEYS = ["calib_dataset", "calib_split", "data_seed", "seq_len", "calib_seqs", "calib_every"]
configs = [json.load(open(os.path.join(d, "config.json"))) for d in args.run_dirs]
calib = {tuple(c["args"].get(k) for k in CALIB_KEYS) for c in configs}
if len(calib) > 1 or any(c["args"].get("calib_every", 1) != 0 for c in configs):
    print("warning: the runs were not all scored on one fixed calibration batch, so their delta_loss "
          "values are not strictly comparable")

if args.metric == "calib":
    nadir = np.max([json.load(open(os.path.join(d, "normalization.json")))["nadir"] for d in args.run_dirs], axis=0)
    print(f"hypervolume: objectives divided by the shared nadir {np.round(nadir, 4).tolist()}, reference (1, 1)")

fig, ax = plt.subplots(figsize=(8, 5))
x_min, details = float("inf"), []
assert len(args.run_dirs) <= len(args.colors), f"give --colors for all {len(args.run_dirs)} runs"
baselines = []
for d, label, marker, color in zip(args.run_dirs, labels, "osD^v<>", args.colors):
    front = pd.read_csv(os.path.join(d, "front.csv")).rename(columns={"loss_increase": "delta_loss"})
    n_gen = int(front["gen"].max())
    if args.metric == "calib":
        last = front[front["gen"] == n_gen].sort_values("size_mb")
        hv = HV(ref_point=np.ones(2))(last[["delta_loss", "size_mb"]].to_numpy() / nadir)
        ax.plot(last["size_mb"], last["delta_loss"], marker=marker, ms=5, lw=1.5, color=color, label=label)
        details.append(f"(N={len(last)}, Gens={n_gen}, HV={hv:.4f})")
        print(f"{label}: generation {n_gen}, {len(last)} points, HV {hv:.5f}")
    else:  # the front's test scores; the line follows model size
        base, last = test_scores(d)
        baselines.append(base[f"{args.metric}_ppl"])
        ax.plot(last["size_mb"], last[f"{args.metric}_ppl"], marker=marker, ms=5, lw=1.5, color=color, label=label)
        details.append(f"(N={len(last)} scored, Gens={n_gen})")
        print(f"{label}: {len(last)} scored, {METRIC_NAME[args.metric]} {last[f'{args.metric}_ppl'].min():.3f}-"
              f"{last[f'{args.metric}_ppl'].max():.3f} (uncompressed {baselines[-1]:.3f})")
    x_min = min(x_min, last["size_mb"].min())
baseline_mb = configs[0]["baseline_size_mb"]
if args.metric == "calib":
    ax.plot(baseline_mb, 0, "D", color="black", label="Uncompressed")
    if args.y == "log":
        ax.set_yscale("symlog", linthresh=0.01)  # linear near 0, log above
        ax.set_ylim(bottom=-0.005)
    else:
        ax.set_ylim(bottom=-0.02 * ax.get_ylim()[1])
else:
    if max(baselines) - min(baselines) > 1e-6 * max(baselines):
        print("warning: the runs have different uncompressed perplexities (different models?); the line shows the first")
    ax.axhline(baselines[0], color="black", ls="--", lw=1, zorder=0)
    ax.plot(baseline_mb, baselines[0], "D", color="black", label=f"Uncompressed ({baselines[0]:.2f})")
    if args.y == "log":
        ax.set_yscale("log")
        plain = FuncFormatter(lambda v, _: f"{v:g}")  # 40, not 4x10^1
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
ax.set_xlim(0, 1.05 * baseline_mb)  # from 0 MB, as in compress_plot
ax.set_xlabel("Model size (MB)")
ax.set_ylabel(r"Calibration loss ($\Delta$)" if args.metric == "calib" else METRIC_NAME[args.metric])
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
