"""
Downstream benchmarks (lm-evaluation-harness) for chosen solutions of the final
Pareto set of a compress_search run: each solution is written into the model in
place (same compressor settings as the search), evaluated, then restored. The
uncompressed model is always evaluated first as the reference.

Outputs in <run-dir>:
    benchmarks.json       scores per solution, updated after every solution, so an
                          interrupted run resumes where it stopped
    samples/<point>.jsonl every question: the question, the correct answer, the
                          model's answer and whether it was right

Needs lm-eval, installed in its own environment so it cannot change the packages
the search uses:

    python -m venv --system-site-packages .venv-eval
    .venv-eval/Scripts/python -m pip install lm-eval accelerate

Examples (run from the repo root):

    # 1) list the Pareto front with its index, size and test perplexities
    .venv-eval/Scripts/python -m scripts.compress_benchmark --run-dir compress_runs/local_100 --list

    # 2) full GSM8K + MMLU for the solutions you picked (and the uncompressed model)
    .venv-eval/Scripts/python -m scripts.compress_benchmark --run-dir compress_runs/local_100 --indices 5,20,41

    # 3) show 5 questions per task where a solution answers differently from the uncompressed model
    .venv-eval/Scripts/python -m scripts.compress_benchmark --run-dir compress_runs/local_100 --indices 5,20,41 --show 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # questions contain symbols the Windows console codepage lacks

parser = argparse.ArgumentParser(description="lm-eval benchmarks for Pareto solutions of a compress_search run")
parser.add_argument("--run-dir", type=str, required=True, help="output directory of compress_search (has config.json, pareto.json)")
parser.add_argument("--list", action="store_true", help="print the Pareto front with indices and exit")
parser.add_argument("--indices", type=str, default=None, help="comma-separated indices from --list (default: all solutions)")
parser.add_argument("--max-points", type=int, default=None, help="evaluate at most this many solutions, spread evenly by size")
parser.add_argument("--max-delta-loss", type=float, default=None, help="skip solutions whose C4 test loss is more than this above the uncompressed model")
parser.add_argument("--tasks", type=str, default="gsm8k,mmlu")
parser.add_argument("--num-fewshot", type=int, default=5, help="5-shot is the usual setting for both GSM8K and MMLU")
parser.add_argument("--limit", type=int, default=None, help="examples per task, per MMLU subject (default: all); for quick checks only")
parser.add_argument("--batch-size", type=str, default="4", help="an integer, or auto (auto probes with very large batches: use it only on an idle GPU)")
parser.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template")
parser.add_argument("--show", type=int, default=0, help="print this many questions per task where each solution disagrees with the uncompressed model")
args = parser.parse_args()

config = json.load(open(os.path.join(args.run_dir, "config.json")))
pareto = json.load(open(os.path.join(args.run_dir, "pareto.json")))
search = config["args"]
out_path = os.path.join(args.run_dir, "benchmarks.json")
samples_dir = os.path.join(args.run_dir, "samples")
tasks = args.tasks.split(",")
front = pareto["pareto"]  # sorted by size
base = pareto["baseline"]


def k_init(p):
    return p.get("k_init", p.get("k"))  # older runs called the grid size "k"


def ppl(p, name):
    return p.get(f"{name}_ppl", float("nan"))


if args.list:
    print(f"{'idx':>4} {'k':>4} {'used':>6} {'c':>6} {'alpha':>7} {'beta':>7} {'size MB':>8} {'ratio':>6} {'bits/w':>7} "
          f"{'sparsity':>8} {'wiki2 ppl':>10} {'c4 ppl':>9}")
    print(f"{'-':>4} {'-':>4} {'-':>6} {'-':>6} {'-':>7} {'-':>7} {base['size_mb']:8.1f} {1:6.1%} {16:7.2f} {0:7.1f}% "
          f"{ppl(base, 'wikitext2'):10.2f} {ppl(base, 'c4'):9.2f}   (uncompressed)")
    for i, p in enumerate(front):
        print(f"{i:4d} {k_init(p):4d} {p.get('k_used_mean', float('nan')):6.1f} {p['c']:6.3f} {p['alpha']:+7.3f} {p['beta']:+7.3f} {p['size_mb']:8.1f} "
              f"{p['size_ratio']:6.1%} {p['target_bits_per_weight']:7.2f} {p['sparsity_pct']:7.1f}% "
              f"{ppl(p, 'wikitext2'):10.2f} {ppl(p, 'c4'):9.2f}")
    raise SystemExit

# which Pareto solutions to evaluate
points = list(enumerate(front))
if args.indices is not None:
    points = [(i, front[i]) for i in map(int, args.indices.split(","))]
if args.max_delta_loss is not None:
    assert "c4_loss" in base, "--max-delta-loss needs C4 test losses in pareto.json"
    points = [(i, p) for i, p in points if p["c4_loss"] - base["c4_loss"] <= args.max_delta_loss]
if args.max_points is not None and len(points) > args.max_points:
    points = [points[j] for j in np.linspace(0, len(points) - 1, args.max_points).round().astype(int)]


def point_id(p):
    return "uncompressed" if p is None else f"k={k_init(p)} c={p['c']:.6f} alpha={p['alpha']:.6f} beta={p['beta']:.6f}"


def samples_path(p):
    name = "uncompressed" if p is None else f"k{k_init(p)}_c{p['c']:.4f}_a{p['alpha']:.4f}_b{p['beta']:.4f}"
    return os.path.join(samples_dir, name + ".jsonl")


settings = {"tasks": tasks, "num_fewshot": args.num_fewshot, "limit": args.limit, "chat_template": args.chat_template}
results = json.load(open(out_path)) if os.path.exists(out_path) else {"settings": settings, "results": {}}
assert {k: results["settings"][k] for k in settings} == settings, \
    f"{out_path} was made with different settings: {results['settings']}"
todo = [(i, p) for i, p in [(None, None), *points] if point_id(p) not in results["results"]]
print(f"{len(points)} solutions selected, {len(todo)} evaluations to run (incl. the uncompressed model) on {tasks}", flush=True)


def simplify(task, s):
    """One question as a small readable record."""
    doc = s["doc"]
    rec = {"task": task, "doc_id": s["doc_id"], "filter": s["filter"], "question": doc.get("question"),
           "correct": bool(s[s["metrics"][0]])}
    if "choices" in doc:  # multiple choice (MMLU): the answer is the choice with the highest log-likelihood
        loglik = [r[0] for r in s["filtered_resps"]]
        rec.update(choices=doc["choices"], answer="ABCD"[s["target"]], predicted="ABCD"[int(np.argmax(loglik))],
                   choice_logliks=loglik)
    else:  # generation (GSM8K): the full generated text and the number extracted from it
        rec.update(answer=str(s["target"]).split("####")[-1].strip(), predicted=s["filtered_resps"][0],
                   generation=s["resps"][0][0], reference_solution=s["target"])
    return rec


if todo:
    assert not search["model"].startswith("nanochat:"), \
        "lm-eval runs HuggingFace models only; for nanochat checkpoints use its own evals (scripts/chat_eval.py)"
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanochat.compress import Candidate, GlobalCompressor, include_from_config

    results["settings"]["lm_eval_version"] = lm_eval.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(search["model"])
    model = AutoModelForCausalLM.from_pretrained(search["model"], dtype=getattr(torch, search["dtype"])).to(device).eval()
    compressor = GlobalCompressor(model, include_from_config(search), search["size_mode"], search["offload_originals"],
                                  search["reconstruction"], formats=search["formats"].split(","),
                                  prune=not search.get("no_pruning", False), method=search.get("method", "ubq"))
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)  # sees the in-place weight changes
    os.makedirs(samples_dir, exist_ok=True)

    for idx, p in todo:
        t0 = time.time()
        entry = {"front_index": idx}
        if p is not None:
            entry.update({key: p[key] for key in ("b", "k_init", "k", "k_used_mean", "c", "alpha", "beta", "size_mb",
                                                  "size_ratio", "target_bits_per_weight", "sparsity_pct") if key in p})
        try:
            if p is not None:
                compressor.apply(Candidate(k_init(p), p["c"], p["alpha"], p["beta"], p.get("rho", 1.0)))
            out = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=args.num_fewshot, limit=args.limit,
                                          log_samples=True, apply_chat_template=args.chat_template,
                                          fewshot_as_multiturn=args.chat_template, random_seed=0, numpy_random_seed=0,
                                          torch_random_seed=0, fewshot_random_seed=0)
        finally:
            compressor.restore()
        # headline metrics: GSM8K exact match (both answer extractions), MMLU accuracy, with stderr
        entry["benchmarks"] = {task: {m: v for m, v in out["results"][task].items()
                                      if isinstance(v, (int, float)) and not m.startswith("sample_")}
                               for task in tasks}
        entry["seconds"] = round(time.time() - t0)
        with open(samples_path(p), "w", encoding="utf-8") as f:
            for task, samples in out["samples"].items():
                for s in samples:
                    f.write(json.dumps(simplify(task, s), ensure_ascii=False) + "\n")
        results["results"][point_id(p)] = entry
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, out_path)
        summary = ", ".join(f"{t}: " + " ".join(f"{m}={v:.3f}" for m, v in r.items() if "stderr" not in m)
                            for t, r in entry["benchmarks"].items())
        print(f"[{'uncompressed' if p is None else f'#{idx}'}] {entry.get('size_mb', base['size_mb']):.1f} MB | "
              f"{summary} | {entry['seconds']}s", flush=True)

# -----------------------------------------------------------------------------
# questions where a solution answers differently from the uncompressed model


def load_samples(p):
    path = samples_path(p)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        # GSM8K is stored once per answer extraction; show the lenient one
        return {(r["task"], r["doc_id"]): r for r in map(json.loads, f) if r["filter"] in ("none", "flexible-extract")}


def clip(text, n=400):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " ..."


if args.show > 0:
    ref = load_samples(None)
    assert ref is not None, "no samples of the uncompressed model yet"
    for idx, p in points:
        cur = load_samples(p)
        if cur is None:
            continue
        print(f"\n{'=' * 100}\n#{idx}: k={k_init(p)} c={p['c']:.3f} alpha={p['alpha']:+.3f} beta={p['beta']:+.3f}, "
              f"{p['size_mb']:.1f} MB ({p['size_ratio']:.1%}), sparsity {p['sparsity_pct']:.1f}%")
        for group in tasks:
            keys = [key for key in cur if key in ref and key[0].startswith(group)]
            lost = [key for key in keys if ref[key]["correct"] and not cur[key]["correct"]]
            gained = [key for key in keys if not ref[key]["correct"] and cur[key]["correct"]]
            print(f"\n--- {group}: {len(keys)} questions, both right {sum(ref[k]['correct'] and cur[k]['correct'] for k in keys)}, "
                  f"lost by compression {len(lost)}, gained {len(gained)}")
            for key in (lost + gained)[:args.show]:
                r, c = ref[key], cur[key]
                print(f"\n[{key[0]} #{key[1]}] {'LOST' if key in lost else 'GAINED'}")
                print(f"  Q: {clip(r['question'])}")
                if "choices" in r:
                    print("  " + "  ".join(f"{l}) {clip(ch, 80)}" for l, ch in zip("ABCD", r["choices"])))
                print(f"  correct answer: {r['answer']}")
                print(f"  uncompressed:   {r['predicted']}" + (f"   <- {clip(r['generation'], 300)}" if "generation" in r else ""))
                print(f"  compressed:     {c['predicted']}" + (f"   <- {clip(c['generation'], 300)}" if "generation" in c else ""))
print(f"\nResults in {out_path}, questions in {samples_dir}", flush=True)
