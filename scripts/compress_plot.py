"""
Animate the Pareto front of a compress_search run, one frame per generation.
Works on a finished or running run.

    python -m scripts.compress_plot --run-dir compress_runs/local_100
    -> <run-dir>/front_evolution.gif (with a progress bar of the generations under the plot)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --with-init-pop
    -> <run-dir>/front_evolution_with_init_pop.gif (the whole initial population in the background)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --with-pop-every-gen
    -> <run-dir>/front_evolution_with_pop_every_gen.gif (each generation's whole surviving population
       in the background; its Pareto front is part of it)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --x bins
    -> <run-dir>/front_evolution_bins.gif (on the x-axis, instead of the model size, the number of
       non-empty bins per matrix, i.e. the lookup-table size k_used_mean, averaged over the matrices;
       not the k gene. Combines with the other options, e.g. --last-gen -> pareto_front_bins.png/.pdf)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --hv
    -> <run-dir>/hypervolume.png/.pdf (hypervolume of the front per generation)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --last-gen
    -> <run-dir>/pareto_front.png/.pdf (the latest generation as a still image instead of the GIF)

    python -m scripts.compress_plot --run-dir compress_runs/local_100 --all
    -> every plot above: the GIF and the still image for each x-axis (bins only if the run logs
       the used bins) and each population view, plus the hypervolume
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--fps", type=int, default=6)
pop_mode = parser.add_mutually_exclusive_group()
pop_mode.add_argument("--with-init-pop", action="store_true", help="also plot every candidate of the initial population")
pop_mode.add_argument("--with-pop-every-gen", action="store_true",
                      help="also plot the current population of every generation (instead of the initial one)")
parser.add_argument("--x", type=str, default="size", choices=["size", "bins"],
                    help="x-axis: model size in MB, or the number of non-empty bins per matrix (log2 axis)")
parser.add_argument("--hv", action="store_true", help="plot the hypervolume per generation instead of the GIF")
parser.add_argument("--last-gen", action="store_true", help="save only the latest generation as a still image (PNG + PDF) instead of the GIF")
parser.add_argument("--all", action="store_true", help="make every plot (all combinations of the options above)")
args = parser.parse_args()
for name in ("front.csv", "generations.csv"):
    if not os.path.exists(os.path.join(args.run_dir, name)):
        sys.exit(f"{args.run_dir} has no {name}: it was made by an early version of compress_search "
                 "that did not log every generation, so it cannot be plotted")

if args.all:  # one process per plot, so no matplotlib state carries over between them
    front_csv = pd.read_csv(os.path.join(args.run_dir, "front.csv"))
    has_bins = "k_used_mean" in front_csv.columns
    runs = [["--hv"]] + ([] if front_csv.empty else [[*x, *pop, *still] for x, pop, still in itertools.product(
        [[], ["--x", "bins"]] if has_bins else [[]], [[], ["--with-init-pop"], ["--with-pop-every-gen"]], [[], ["--last-gen"]])])
    for flags in runs:
        subprocess.run([sys.executable, "-m", "scripts.compress_plot", "--run-dir", args.run_dir, "--fps", str(args.fps), *flags],
                       check=True)
    if front_csv.empty:
        print("no front plots: no generation has a feasible solution yet (see --max-delta-loss)")
    elif not has_bins:
        print("no bins plots: this run does not log the used bins (k_used_mean)")
    raise SystemExit

if args.hv:
    gens = pd.read_csv(os.path.join(args.run_dir, "generations.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens["gen"], gens["hypervolume"], "o-", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Hypervolume per generation (final {gens['hypervolume'].iloc[-1]:.4f})")
    ax.grid(True, alpha=0.3)
    out = os.path.join(args.run_dir, "hypervolume")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=120)
    print(f"{len(gens)} generations -> {out}.png/.pdf")
    raise SystemExit

# delta_loss = loss of the compressed model - loss of the uncompressed one on the same batch
# (runs started before the renames call it loss_increase, and k_init k)
read = lambda name: pd.read_csv(os.path.join(args.run_dir, name)).rename(columns={"loss_increase": "delta_loss", "k": "k_init"})
X = "size_mb" if args.x == "size" else "k_used_mean"
front = read("front.csv")
if front.empty:
    sys.exit(f"{args.run_dir}: the front is empty, no generation has a feasible solution yet (see --max-delta-loss)")
assert X in front.columns, f"{args.run_dir} does not log {X} (run made before the lookup tables)"
gens = pd.read_csv(os.path.join(args.run_dir, "generations.csv")).set_index("gen")
# along the front: left to right, and top to bottom among solutions with the same x
front = front[front["gen"].isin(gens.index)].sort_values(["gen", X, "delta_loss"], ascending=[True, True, False])
config = json.load(open(os.path.join(args.run_dir, "config.json")))
baseline_mb = config["baseline_size_mb"]
n_gen = max(config["args"]["n_gen"], gens.index.max())  # generations the run was started for

fig, ax = plt.subplots(figsize=(8, 5))
first = front[front["gen"] == front["gen"].min()]
y_max = front["delta_loss"].max()
if args.with_init_pop:
    pop = read("population.csv")
    init = pop[pop["gen"] == pop["gen"].min()]
    ax.scatter(init[X], init["delta_loss"], s=16, color="lightgray", label="Initial population")
    y_max = max(y_max, init["delta_loss"].max())
pop_scatter = None
if args.with_pop_every_gen:
    pop = read("population.csv")
    pop = pop[pop["gen"].isin(gens.index)]
    pop_scatter = ax.scatter([], [], s=16, color="lightgray", label="Population")
    y_max = max(y_max, pop["delta_loss"].max())
# the front is Pareto-optimal in size, not in bins: in the bins view a line joining its points
# in bins order would zigzag, so there the fronts are drawn as points only
bins = args.x == "bins"
if not args.with_pop_every_gen:  # the population view shows the current generation only
    ax.plot(first[X], first["delta_loss"], "o" if bins else "--", color="gray", mfc="none", label="Initial Pareto front")
line, = ax.plot([], [], "o" if bins else "o-", label="Pareto front")
ax.set_yscale("symlog", linthresh=0.01)  # linear near 0, log above: shows both +0.001 and +15
if args.x == "size":
    ax.plot(baseline_mb, 0, "D", color="black", label="Uncompressed")
    pad = 0.05 * (baseline_mb - front["size_mb"].min())  # 5% of the size range on each side
    ax.set_xlim(front["size_mb"].min() - pad, baseline_mb + pad)
    ax.set_xlabel("Model size (MB)")
else:  # the uncompressed model has no bins (16-bit floats), so it has no point here
    shown = pd.concat([front[X], pop[X] if args.with_init_pop or args.with_pop_every_gen else front[X]])
    lo, hi = shown.min(), shown.max()
    ax.set_xscale("log", base=2)  # equal steps per extra bit
    ax.set_xlim(lo / 1.15, hi * 1.15)
    ticks = sorted({round(lo), *(2 ** b for b in range(1, 11) if lo <= 2 ** b <= hi)})
    ax.set_xticks(ticks, labels=[str(k) for k in ticks])
    ax.set_xlabel("Non-empty bins per matrix (mean)")
ax.set_ylim(-0.005, y_max * 1.5)
ax.set_ylabel("Calibration loss")
ax.grid(True, alpha=0.3)
legend = ax.legend(loc="upper right")
front_label = next(t for t in legend.get_texts() if t.get_text() == "Pareto front")  # gets the front size per frame
title = ax.set_title("")
bar = None
if not args.last_gen:  # progress bar of the generations, only in the GIF: a rounded track that fills up
    fig.subplots_adjust(bottom=0.2)
    bar_ax = fig.add_axes([ax.get_position().x0, 0.035, ax.get_position().width, 0.06])
    bar_ax.set_xlim(0, n_gen)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    pill = dict(lw=11, solid_capstyle="round", clip_on=False)
    bar_ax.plot([0, n_gen], [0.3, 0.3], color="#e3e7ee", zorder=1, **pill)
    for g in range(10, n_gen, 10):  # a faint tick every 10 generations
        bar_ax.plot(g, 0.3, "o", ms=2.5, color="#b9c1ce", zorder=1.5)
    bar, = bar_ax.plot([0, 0], [0.3, 0.3], color=line.get_color(), zorder=2, **pill)
    bar_label = bar_ax.text(0, 0.95, "", ha="left", va="center", fontsize=10, color="#333333")
    bar_pct = bar_ax.text(n_gen, 0.95, "", ha="right", va="center", fontsize=10, color="#777777")

def draw(gen):
    f = front[front["gen"] == gen]
    line.set_data(f[X], f["delta_loss"])
    title.set_text(f"Generation {gen}   HV {gens.loc[gen, 'hypervolume']:.4f}")
    front_label.set_text(f"Pareto front (N={len(f)})")
    if pop_scatter is not None:
        p = pop[pop["gen"] == gen]
        pop_scatter.set_offsets(p[[X, "delta_loss"]].to_numpy())
    if bar is None:
        return line, title
    bar.set_xdata([0, gen])
    bar_label.set_text(f"Generation {gen} of {n_gen}")
    bar_pct.set_text(f"{100 * gen / n_gen:.0f}%")
    return line, title, bar, bar_label, bar_pct, *([pop_scatter] if pop_scatter is not None else [])


suffix = ("_bins" if args.x == "bins" else "") + \
         ("_with_init_pop" if args.with_init_pop else "_with_pop_every_gen" if args.with_pop_every_gen else "")
if args.last_gen:
    draw(gens.index[-1])
    out = os.path.join(args.run_dir, "pareto_front" + suffix)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=120, bbox_inches="tight")
    print(f"generation {gens.index[-1]} -> {out}.png/.pdf")
    raise SystemExit

out = os.path.join(args.run_dir, f"front_evolution{suffix}.gif")
FuncAnimation(fig, draw, frames=list(gens.index)).save(out, writer="pillow", fps=args.fps)
print(f"{len(gens)} generations -> {out}")
