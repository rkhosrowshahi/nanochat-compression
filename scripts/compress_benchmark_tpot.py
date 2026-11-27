"""
Coding benchmarks (lm-evaluation-harness: HumanEval and MBPP, pass@1 with greedy decoding) for the final Pareto front of a
compress_search_tpot run (scripts/compress_search_tpot.py: per-matrix formats among bf16 and int4 g32/g64/g128, objectives
KL divergence, TPOT and memory).

Which solutions: as scripts/compress_compare.py --archive --archive-size N plots them: from every evaluation of the run
(evals.csv), the N that NSGA-II's survival keeps (non-dominated sorting, then crowding distance; default N = 50, the
population size). --archive-size 0 takes the last population's front (front.csv) instead. --indices picks some of them
(see --list).

Each solution is written into the model in place by the run's compressor (TPOTCompressor.apply, rebuilt from the run's
config.json, latency table and activation statistics, checked against its genes and costs), so every matrix gets the
values its int4 format gives, rounded as the search rounded them; then it is evaluated and restored. The uncompressed
model is evaluated first, as the reference.

The code tasks run the code the model writes (lm-eval's confirm_run_unsafe_code, HF_ALLOW_CODE_EVAL=1): run this on a
machine or container where that is acceptable.

Outputs in <run-dir>/benchmarks_<selection> (or --out-dir):
    benchmarks.json       scores per solution, saved after every solution: an interrupted run resumes
    benchmarks.md/.csv    pass@1 (%) of every evaluated solution, next to its KLD, TPOT and memory
    samples/<point>.jsonl every problem: the model's completion and whether it passed

Needs lm-eval (see scripts/compress_benchmark.py for its environment). From the repo root:

    .venv-eval/bin/python -m scripts.compress_benchmark_tpot --run-dir compress_runs/coder15_tpot --list
    .venv-eval/bin/python -m scripts.compress_benchmark_tpot --run-dir compress_runs/coder15_tpot             # all 50
    .venv-eval/bin/python -m scripts.compress_benchmark_tpot --run-dir compress_runs/coder15_tpot --indices 0,10,25,49
    # a quick check: 10 problems per task
    .venv-eval/bin/python -m scripts.compress_benchmark_tpot --run-dir compress_runs/coder15_tpot --indices 0 --limit 10
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="HumanEval / MBPP pass@1 of a compress_search_tpot run's Pareto front")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--archive-size", type=int, default=50,
                    help="solutions chosen from every evaluation by non-dominated sorting and crowding distance "
                         "(0: the last population's front)")
parser.add_argument("--out-dir", type=str, default=None, help="default: <run-dir>/benchmarks_<selection>")
parser.add_argument("--list", action="store_true", help="print the solutions with indices and exit (no GPU work)")
parser.add_argument("--dry-run", action="store_true", help="print what would be evaluated and exit (no GPU work)")
parser.add_argument("--indices", type=str, default=None, help="comma-separated indices from --list (default: all)")
parser.add_argument("--tasks", type=str, default="humaneval:0,mbpp:3",
                    help="comma-separated lm-eval tasks as task:shots (e.g. humaneval_plus:0,mbpp_plus:3)")
parser.add_argument("--limit", type=int, default=None, help="problems per task (default: all); for quick checks only")
parser.add_argument("--batch-size", type=str, default="16", help="an integer, or auto")
parser.add_argument("--offload", type=str, default=None, choices=["yes", "no", "auto"],
                    help="keep the original weights on the CPU (default: as the run did; yes on an 8 GB GPU)")
parser.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template")
args = parser.parse_args()

run_path = lambda name: os.path.join(args.run_dir, name)
config = json.load(open(run_path("config.json")))
a = config["args"]
names = json.load(open(run_path("normalization.json")))["objectives"]
assert names[0] == "kld" and "tpot_ms" in config["baseline"], f"{args.run_dir} is not a compress_search_tpot run"
gene_names = [n for n, _ in config["genes"]]
selection = f"archive{args.archive_size}" if args.archive_size else "population"
out_dir = args.out_dir or run_path(f"benchmarks_{selection}")
out_path, samples_dir = os.path.join(out_dir, "benchmarks.json"), os.path.join(out_dir, "samples")
shots = {s.split(":")[0]: int(s.split(":")[1]) if ":" in s else 0 for s in args.tasks.split(",")}
tasks = list(shots)
GiB = 1 << 30

# --- the solutions


def nondominated(F):
    return np.array([not ((F <= f).all(1) & (F < f).any(1)).any() for f in F])


if args.archive_size:
    from pymoo.algorithms.moo.nsga2 import RankAndCrowding
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem

    rows = pd.read_csv(run_path("evals.csv")).drop_duplicates("key")
    if a.get("max_kld") is not None:
        rows = rows[rows["kld"] <= a["max_kld"]]
    F = rows[names].to_numpy()
    pop = Population.new("F", F, "index", np.arange(len(F)))
    kept = RankAndCrowding().do(Problem(n_var=1, n_obj=F.shape[1]), pop, n_survive=args.archive_size)
    rows = rows.iloc[kept.get("index")]
    rows = rows[nondominated(rows[names].to_numpy())]
else:
    rows = pd.read_csv(run_path("front.csv"))
    rows = rows[rows["gen"] == rows["gen"].max()].drop_duplicates("key")
rows = rows[~rows[gene_names].eq(0).all(axis=1)].sort_values("tpot_ms")  # all genes 0: the uncompressed model itself
front = []
for r in rows.to_dict("records"):
    p = {k: v.item() if hasattr(v, "item") else v for k, v in r.items()}
    p["genes"] = tuple(int(p[n]) for n in gene_names)
    front.append(p)
base = config["baseline"]
print(f"{args.run_dir}: {len(front)} compressed solutions ({selection}); uncompressed TPOT {base['tpot_ms']:.2f} ms/tok, "
      f"{base['memory_mb'] * 1e6 / GiB:.2f} GiB", flush=True)

if args.list or args.dry_run:
    print(f"{'idx':>4} {'KLD':>7} {'TPOT ms/tok':>17} {'Memory GiB':>17}  config")
    picked = set(map(int, args.indices.split(","))) if args.indices else None
    for i, p in enumerate(front):
        if args.list or picked is None or i in picked:
            print(f"{i:4d} {p['kld']:7.4f} {p['tpot_ms']:6.2f} ({base['tpot_ms'] / p['tpot_ms']:4.2f}x) "
                  f"{p['memory_mb'] * 1e6 / GiB:6.2f} ({base['memory_mb'] / p['memory_mb']:4.2f}x)  {p['config']}")
    if args.dry_run:
        print(f"tasks {shots}, limit {args.limit}, outputs in {out_dir}")
    raise SystemExit

points = [(i, front[i]) for i in map(int, args.indices.split(","))] if args.indices else list(enumerate(front))


def point_id(p):
    return "uncompressed" if p is None else p["key"]


def samples_path(p):
    return os.path.join(samples_dir, ("uncompressed" if p is None else p["key"].replace(".", "-")) + ".jsonl")


settings = {"tasks": shots, "limit": args.limit, "chat_template": args.chat_template, "selection": selection}
os.makedirs(samples_dir, exist_ok=True)
results = json.load(open(out_path)) if os.path.exists(out_path) else {"settings": settings, "results": {}}
assert {k: results["settings"].get(k) for k in settings} == settings, f"{out_path} was made with other settings: {results['settings']}"
todo = [(i, p) for i, p in [(None, None), *points] if point_id(p) not in results["results"]]
print(f"{len(points)} solutions selected, {len(todo)} evaluations to run (incl. the uncompressed model) on {shots}", flush=True)

# --- evaluate

if todo:
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM

    from nanochat import compress_hw, compress_models, compress_tpot

    device = "cuda"
    model, tokenizer, _ = compress_models.load(a["model"], a["dtype"], device)
    L = compress_hw.Layout(model)
    table = json.load(open(run_path("latency_table.json")))
    act_sq = torch.load(run_path("act_stats.pt"), map_location=device)["stats"]  # the search's clipping statistics
    order = {k: [(i, 0.0) for i in range(L.n_layer)] for k in ("attn", "mlp")}  # nothing is removed
    comp = compress_tpot.TPOTCompressor(model, L, table, act_sq, order, formats=a["formats"].split(","), bands=a["bands"],
                                        mlp_keep=[float(k) for k in str(a["mlp_keep"]).split(",")], max_skip=a["max_skip"],
                                        compress_embeddings=not a.get("no_embeddings", False),
                                        clip_ratios=[float(r) for r in a["clip_ratios"].split(",")],
                                        offload=args.offload or a["offload_originals"])
    assert [list(g) for g in comp.genes] == [list(g) for g in config["genes"]], "the genes differ from the run's"
    results["settings"]["lm_eval_version"] = lm_eval.__version__
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)  # sees the in-place changes
    for idx, p in todo:
        t0 = time.time()
        entry = {"front_index": idx}
        if p is not None:
            cfg = comp.config(p["genes"])
            cost = comp.cost(cfg)
            assert abs(cost["tpot_ms"] - p["tpot_ms"]) < 1e-6 * p["tpot_ms"] and abs(cost["memory_mb"] - p["memory_mb"]) < 1e-3, \
                f"#{idx}: the rebuilt costs differ from the run's"
            entry.update({k: p[k] for k in ("key", "config", "kld", "top1_agree", "tpot_ms", "memory_mb", "bits_per_weight",
                                            "int4_pct")})
        try:
            if p is not None:
                comp.apply(cfg)
            out = {"results": {}, "samples": {}}
            for n in sorted(set(shots.values())):  # one lm-eval call per number of shots
                o = lm_eval.simple_evaluate(model=lm, tasks=[t for t in tasks if shots[t] == n], num_fewshot=n,
                                            limit=args.limit, log_samples=True, confirm_run_unsafe_code=True,
                                            apply_chat_template=args.chat_template, fewshot_as_multiturn=args.chat_template,
                                            random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
                out["results"].update(o["results"])
                out["samples"].update(o["samples"])
        finally:
            comp.restore()
        entry["benchmarks"] = {t: {m: v for m, v in out["results"][t].items() if isinstance(v, (int, float))} for t in tasks}
        entry["seconds"] = round(time.time() - t0)
        results["results"][point_id(p)] = entry
        with open(out_path + ".tmp", "w") as f:
            json.dump(results, f, indent=2)
        os.replace(out_path + ".tmp", out_path)
        with open(samples_path(p), "w", encoding="utf-8") as f:
            for task, samples in out["samples"].items():
                for s in samples:
                    metric = next(m for m in s["metrics"] if "pass" in m) if any("pass" in m for m in s["metrics"]) else s["metrics"][0]
                    f.write(json.dumps({"task": task, "doc_id": s["doc_id"], "passed": bool(s[metric]),
                                        "completion": s["resps"][0][0] if s["resps"] else None}, ensure_ascii=False) + "\n")
        print(f"[{'uncompressed' if p is None else f'#{idx}'}] " + ", ".join(
            f"{t}: " + " ".join(f"{m}={v:.3f}" for m, v in r.items() if "stderr" not in m) for t, r in entry["benchmarks"].items())
            + f" | {entry['seconds']}s", flush=True)

# --- table: pass@1 (%) of every evaluated solution next to the uncompressed model

TASK_NAMES = {"humaneval": "HumanEval", "humaneval_plus": "HumanEval+", "humaneval_instruct": "HumanEval (instruct)",
              "mbpp": "MBPP", "mbpp_plus": "MBPP+", "mbpp_instruct": "MBPP (instruct)"}


def score(entry, task):
    """(metric, pass@1 in percent, its standard error in percent) of one task."""
    r = entry["benchmarks"][task]
    m = next((m for m in r if m.startswith("pass") and "stderr" not in m), None) or \
        next(m for m in ("acc_norm,none", "acc,none", "exact_match,flexible-extract") if m in r)
    metric, filt = m.split(",", 1) if "," in m else (m, "")
    stderr = r.get(f"{metric}_stderr,{filt}" if filt else f"{metric}_stderr", float("nan"))
    return m, 100 * r[m], 100 * stderr if isinstance(stderr, (int, float)) else float("nan")


ref = results["results"].get("uncompressed")
evaluated = sorted((e for k, e in results["results"].items() if k != "uncompressed"), key=lambda e: e["kld"])
if ref is not None:
    header = ["Configuration", "KLD (nats)", "TPOT (ms/tok)", "Memory (GiB)", "Bits/weight",
              *[f"{TASK_NAMES.get(t, t)} {'pass@1' if score(ref, t)[0].startswith('pass') else 'accuracy'} ({shots[t]}-shot)"
                for t in tasks], "Average", "Retained (%)"]
    ref_avg = np.mean([score(ref, t)[1] for t in tasks])
    table = []
    for e in [ref, *evaluated]:
        accs = [score(e, t)[1] for t in tasks]
        c = e if e is not ref else {**base, "kld": 0.0, "bits_per_weight": 16.0, "config": f"uncompressed ({a['dtype']})"}
        table.append([c["config"], f"{c['kld']:.4f}", f"{c['tpot_ms']:.2f} ({base['tpot_ms'] / c['tpot_ms']:.2f}x)",
                      f"{c['memory_mb'] * 1e6 / GiB:.2f} ({base['memory_mb'] / c['memory_mb']:.2f}x)", f"{c['bits_per_weight']:.2f}",
                      *[f"{x:.1f}" for x in accs], f"{np.mean(accs):.1f}",
                      f"{100 * np.mean(accs) / ref_avg:.1f}" if ref_avg else "-"])
    notes = [f"pass@1 in percent, greedy decoding. Retained: the average as a percentage of the uncompressed model's. KLD: "
             f"mean KL divergence from the uncompressed model on the search's code calibration windows. TPOT: time per output "
             f"token from the run's latency table ({config['latency_table']['gpu']}), with the speedup; memory: the model as "
             f"loaded on the GPU (no KV cache), with the compression ratio.",
             "Standard errors of the uncompressed model: " + "; ".join(
                 f"{TASK_NAMES.get(t, t)} ±{score(ref, t)[2]:.1f}" for t in tasks) + "."]
    md = "\n".join([f"# {a['model']}: coding benchmarks of the Pareto front ({args.run_dir}, {selection})", "",
                    "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                    *["| " + " | ".join(row) + " |" for row in table], "", *notes, ""])
    with open(os.path.join(out_dir, "benchmarks.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "benchmarks.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *table])
    print("\n" + md, flush=True)
print(f"\nResults in {out_path}, completions in {samples_dir}", flush=True)
