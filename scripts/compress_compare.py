"""
Plot the final Pareto fronts of several compress_search runs in one figure.

The y-axis is the calibration delta_loss, so the runs should share the calibration batch
(same --calib-every 0 batch: seed, --calib-seqs, --seq-len); a warning is printed otherwise.
The legend gives each front's hypervolume with ONE normalization for all runs: each objective
divided by the worst value over the initial-front nadirs of all runs, reference (1, 1). (Each
run's own logged hypervolume uses its own nadir, so those are not comparable across runs.)

    python -m scripts.compress_compare \
        --run-dirs compress_runs/local_50_linear_k256_calib_fixed_8x2048 compress_runs/local_50_linear_k256_calib_fixed_8x2048_maxdelta1 \
        --labels "No constraint" "$\\Delta \\leq 1$" --out compress_runs/comparisons/qwen_constraint/final_fronts_constraint
    -> <out>.png/.pdf

--metric wikitext2 | c4 plots the test scores of the final front instead (WikiText-2 test or C4
validation perplexity, from the run's test_scores.json, or pareto.json if it has none), with the
uncompressed model's perplexity as a dashed line. Works for one run as well:
    python -m scripts.compress_compare --run-dirs <run> --metric wikitext2 --out <run>/test_front_wikitext2

--x latency | memory puts the predicted decode latency or the deployed memory of compress_search_hw
runs on the x-axis instead of the model size, e.g. its two scenarios:
    python -m scripts.compress_compare --run-dirs compress_runs/qwen_hw_2obj compress_runs/qwen_hw_3obj \
        --labels "2 objectives" "3 objectives" --x latency --out compress_runs/comparisons/qwen_hw_2obj_vs_3obj/qwen_hw_fronts_latency
A front is a line when x is its run's only objective besides delta_loss; otherwise (a 3-objective front,
or memory for a run that did not optimize it) it is not monotone in x, so it is drawn as points only.

--3d plots compress_search_hw fronts in 3-D instead: delta_loss (or with --metric the test perplexity)
over the deployed memory and the predicted decode latency, with the hypervolume of all three objectives:
    python -m scripts.compress_compare --run-dirs compress_runs/qwen_hw_2obj compress_runs/qwen_hw_3obj \
        --labels "2 objectives" "3 objectives" --3d --out compress_runs/comparisons/qwen_hw_2obj_vs_3obj/qwen_hw_fronts_3d

--hv plots the hypervolume of each run's front per generation instead, with the same shared normalization
(of delta_loss and x; with --3d of delta_loss, latency and memory, e.g. to judge a 2-objective run by all
three). Runs with the same label are seeds of one setting: the line is their mean, the band their range,
and with two settings the final values are compared by a Mann-Whitney U test:
    python -m scripts.compress_compare --run-dirs compress_runs/qwen_hw_2obj compress_runs/qwen_hw_3obj \
        --labels "2 objectives" "3 objectives" --3d --hv --out compress_runs/comparisons/qwen_hw_2obj_vs_3obj/qwen_hw_hv_3d
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
parser.add_argument("--out", type=str, default="compress_runs/comparisons/final_fronts",
                    help="output path without extension; one folder per comparison, e.g. "
                         "compress_runs/comparisons/qwen_ubq_vs_ecsq/final_fronts_log")
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
parser.add_argument("--x", type=str, default="size", choices=["size", "latency", "memory"],
                    help="x-axis: model size, or (compress_search_hw runs) predicted decode latency / deployed memory")
parser.add_argument("--3d", dest="three_d", action="store_true",
                    help="compress_search_hw runs: 3-D plot over deployed memory (x) and predicted decode latency (y) "
                         "(--x and --y do not apply)")
parser.add_argument("--hv", action="store_true",
                    help="the hypervolume per generation instead of the final fronts (with --3d: of all three "
                         "objectives); runs with the same label are seeds of one setting")
args = parser.parse_args()
labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
assert len(labels) == len(args.run_dirs), "give one label per run"
METRIC_NAME = {"wikitext2": "WikiText-2 test perplexity", "c4": "C4 validation perplexity"}
X = {"size": "size_mb", "latency": "latency_ms", "memory": "memory_mb"}[args.x]
X_LABEL = {"size": "Model size (MB)", "latency": "Predicted decode latency (ms per token)",
           "memory": "Deployed memory: weights + KV cache (MB)"}[args.x]
if args.title is None:
    args.title = ("Hypervolume per generation, objectives " + ("{delta loss, latency, memory}" if args.three_d else f"{{delta loss, {args.x}}}")
                  if args.hv else
                  "Final Pareto fronts" if args.metric == "calib" else f"Final Pareto fronts: {METRIC_NAME[args.metric]}")


def test_scores(run_dir):
    """(baseline, DataFrame of the scored front) from test_scores.json (compress_score) or pareto.json."""
    for name in ("test_scores.json", "pareto.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            results = json.load(open(path))
            df = pd.DataFrame(results["pareto"])
            if f"{args.metric}_ppl" in df.columns:
                return results["baseline"], df.sort_values(X) if X in df.columns else df
    raise SystemExit(f"{run_dir}: no {args.metric} test scores (run python -m scripts.compress_score --run-dir {run_dir})")


def objectives(run_dir):
    """The run's objectives (compress_search runs: delta_loss and size)."""
    return json.load(open(os.path.join(run_dir, "normalization.json"))).get("objectives", ["delta_loss", "size_mb"])


def run_nadir(run_dir, cols):
    """The run's nadir in cols; for an objective it did not optimize, the worst value on its first front."""
    nadir, names = json.load(open(os.path.join(run_dir, "normalization.json")))["nadir"], objectives(run_dir)
    front = pd.read_csv(os.path.join(run_dir, "front.csv"))
    first = front[front["gen"] == front["gen"].min()]
    return [nadir[names.index(c)] if c in names else first[c].max() for c in cols]


def save(fig, ax, details):
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


CALIB_KEYS = ["calib_dataset", "calib_split", "data_seed", "seq_len", "calib_seqs", "calib_every"]
configs = [json.load(open(os.path.join(d, "config.json"))) for d in args.run_dirs]
calib = {tuple(c["args"].get(k) for k in CALIB_KEYS) for c in configs}
if len(calib) > 1 or any(c["args"].get("calib_every", 1) != 0 for c in configs):
    print("warning: the runs were not all scored on one fixed calibration batch, so their delta_loss "
          "values are not strictly comparable")
assert len(args.run_dirs) <= len(args.colors), f"give --colors for all {len(args.run_dirs)} runs"

if args.hv:  # hypervolume of each run's front per generation, one shared normalization
    cols = ["delta_loss", "latency_ms", "memory_mb"] if args.three_d else ["delta_loss", X]
    nadir = np.max([run_nadir(d, cols) for d in args.run_dirs], axis=0)
    print(f"hypervolume of {cols}: divided by the shared nadir {np.round(nadir, 4).tolist()}, reference {(1,) * len(cols)}")
    groups = {}  # label -> [(generations, hypervolumes)], one per seed
    for d, label in zip(args.run_dirs, labels):
        front = pd.read_csv(os.path.join(d, "front.csv")).rename(columns={"loss_increase": "delta_loss"})
        gens = sorted(front["gen"].unique())
        hvs = [HV(ref_point=np.ones(len(cols)))(front[front["gen"] == g][cols].to_numpy() / nadir) for g in gens]
        groups.setdefault(label, []).append((gens, hvs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    finals, details = {}, []
    for (label, runs), marker, color in zip(groups.items(), "osD^v<>", args.colors):
        n = min(len(gens) for gens, _ in runs)  # the generations every seed has
        hv = np.array([hvs[:n] for _, hvs in runs])
        finals[label] = hv[:, -1]
        ax.plot(runs[0][0][:n], hv.mean(axis=0), marker=marker, ms=3, color=color, label=label)
        if len(runs) > 1:
            ax.fill_between(runs[0][0][:n], hv.min(axis=0), hv.max(axis=0), color=color, alpha=0.2)
        details.append(f"(final {hv[:, -1].mean():.4f}" + (f" ± {hv[:, -1].std(ddof=1):.4f}, {len(runs)} seeds)" if len(runs) > 1 else ")"))
        print(f"{label}: {len(runs)} run(s), final hypervolume " + ", ".join(f"{v:.5f}" for v in hv[:, -1]))
    n = min(len(gens) for runs in groups.values() for gens, _ in runs)
    print("generation  " + "  ".join(f"{label:>14}" for label in groups))  # mean over the seeds of each setting
    for i in range(n):
        print(f"{groups[labels[0]][0][0][i]:10d}  " + "  ".join(
            f"{np.mean([hvs[i] for _, hvs in runs]):14.5f}" for runs in groups.values()))
    if len(finals) == 2 and all(len(v) > 1 for v in finals.values()):
        from scipy.stats import mannwhitneyu
        (a, va), (b, vb) = finals.items()
        print(f"Mann-Whitney U, final hypervolume of {a} vs {b}: p = {mannwhitneyu(va, vb).pvalue:.4f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    save(fig, ax, details)
    raise SystemExit

if args.three_d:  # compress_search_hw fronts over deployed memory (x) and predicted latency (y)
    cols = ["delta_loss", "latency_ms", "memory_mb"]
    if args.metric == "calib":
        nadir = np.max([run_nadir(d, cols) for d in args.run_dirs], axis=0)
        print(f"hypervolume: objectives divided by the shared nadir {np.round(nadir, 4).tolist()}, reference (1, 1, 1)")
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    details = []
    for d, label, marker, color in zip(args.run_dirs, labels, "osD^v<>", args.colors):
        front = pd.read_csv(os.path.join(d, "front.csv"))
        n_gen = int(front["gen"].max())
        if args.metric == "calib":
            last = front[front["gen"] == n_gen]
            z = last["delta_loss"]
            hv = HV(ref_point=np.ones(3))(last[cols].to_numpy() / nadir)
            details.append(f"(N={len(last)}, Gens={n_gen}, HV={hv:.4f})")
            print(f"{label}: generation {n_gen}, {len(last)} points, HV {hv:.5f}")
        else:
            base, last = test_scores(d)
            z = last[f"{args.metric}_ppl"]
            details.append(f"(N={len(last)} scored, Gens={n_gen})")
        ax.scatter(last["memory_mb"], last["latency_ms"], z, marker=marker, s=20, color=color, depthshade=False, label=label)
    base = configs[0]["baseline"]
    base_z = 0 if args.metric == "calib" else test_scores(args.run_dirs[0])[0][f"{args.metric}_ppl"]
    ax.scatter(base["memory_mb"], base["latency_ms"], base_z, marker="D", s=30, color="black", depthshade=False,
               label="Uncompressed")
    ax.set_xlabel("Deployed memory (MB)")
    ax.set_ylabel("Predicted decode latency (ms per token)")
    ax.set_zlabel(r"$\Delta$ Calibration loss (nats)" if args.metric == "calib" else METRIC_NAME[args.metric])
    ax.set_title(args.title)
    save(fig, ax, details)
    raise SystemExit

if args.metric == "calib":
    nadir = np.max([run_nadir(d, ["delta_loss", X]) for d in args.run_dirs], axis=0)
    print(f"hypervolume: objectives divided by the shared nadir {np.round(nadir, 4).tolist()}, reference (1, 1)")

fig, ax = plt.subplots(figsize=(8, 5))
x_min, details = float("inf"), []
baselines = []
for d, label, marker, color in zip(args.run_dirs, labels, "osD^v<>", args.colors):
    front = pd.read_csv(os.path.join(d, "front.csv")).rename(columns={"loss_increase": "delta_loss"})
    n_gen = int(front["gen"].max())
    ls = "-" if objectives(d) == ["delta_loss", X] else "none"  # a line only where the front is monotone in x
    if args.metric == "calib":
        last = front[front["gen"] == n_gen].sort_values(X)
        hv = HV(ref_point=np.ones(2))(last[["delta_loss", X]].to_numpy() / nadir)
        ax.plot(last[X], last["delta_loss"], marker=marker, ms=5, lw=1.5, ls=ls, color=color, label=label)
        details.append(f"(N={len(last)}, Gens={n_gen}, HV={hv:.4f})")
        print(f"{label}: generation {n_gen}, {len(last)} points, HV {hv:.5f}")
    else:  # the front's test scores; the line follows x
        base, last = test_scores(d)
        baselines.append(base[f"{args.metric}_ppl"])
        ax.plot(last[X], last[f"{args.metric}_ppl"], marker=marker, ms=5, lw=1.5, ls=ls, color=color, label=label)
        details.append(f"(N={len(last)} scored, Gens={n_gen})")
        print(f"{label}: {len(last)} scored, {METRIC_NAME[args.metric]} {last[f'{args.metric}_ppl'].min():.3f}-"
              f"{last[f'{args.metric}_ppl'].max():.3f} (uncompressed {baselines[-1]:.3f})")
    x_min = min(x_min, last[X].min())
baseline_mb = configs[0]["baseline_size_mb"] if args.x == "size" else configs[0]["baseline"][X]
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
ax.set_xlim(0, 1.05 * baseline_mb)  # from 0, as in compress_plot
ax.set_xlabel(X_LABEL)
ax.set_ylabel(r"$\Delta$ Calibration loss (nats)" if args.metric == "calib" else METRIC_NAME[args.metric])
ax.set_title(args.title)
ax.grid(True, alpha=0.3)
save(fig, ax, details)
