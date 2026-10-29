"""
Downstream benchmarks (lm-evaluation-harness) for chosen solutions of the final
Pareto set of a compress_search run: each solution is written into the model in
place (same compressor settings as the search), evaluated, then restored. The
uncompressed model is always evaluated first as the reference.

The front is read from pareto.json, or, for searches run without test windows,
from test_scores.json (scripts.compress_score).

Outputs in <run-dir>:
    benchmarks.json       scores per solution, updated after every solution, so an
                          interrupted run resumes where it stopped
    benchmarks.md/.csv    accuracy table (%) of every evaluated solution
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

    # 4) the smallest solutions within 0/1/2/5/10% of the uncompressed calibration loss, on the
    #    zero-shot suite of quantization papers plus 5-shot MMLU and GSM8K (task:shots)
    .venv-eval/Scripts/python -m scripts.compress_benchmark --run-dir compress_runs/local_100 --calib-error-pct 0,1,2,5,10 \
        --num-fewshot 0 --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu:5,gsm8k:5
"""

import argparse
import csv
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
parser.add_argument("--calib-error-pct", type=str, default=None,
                    help="comma-separated levels, e.g. 0,1,2,5,10: for each, the smallest (hardware-aware runs: fastest) solution "
                         "whose calibration loss is at most that many percent above the uncompressed model's (none within the "
                         "level: the most accurate solution)")
parser.add_argument("--reference-from", type=str, default=None,
                    help="run directory of the same model benchmarked with the same settings: reuse its uncompressed-model scores")
parser.add_argument("--tasks", type=str, default="gsm8k,mmlu", help="comma-separated lm-eval tasks, each optionally task:shots")
parser.add_argument("--num-fewshot", type=int, default=5, help="shots of tasks without :shots; 5-shot is the usual setting for both GSM8K and MMLU")
parser.add_argument("--limit", type=int, default=None, help="examples per task, per MMLU subject (default: all); for quick checks only")
parser.add_argument("--batch-size", type=str, default="4", help="an integer, or auto (auto probes with very large batches: use it only on an idle GPU)")
parser.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template")
parser.add_argument("--show", type=int, default=0, help="print this many questions per task where each solution disagrees with the uncompressed model")
args = parser.parse_args()

config = json.load(open(os.path.join(args.run_dir, "config.json")))
search = config["args"]
out_path = os.path.join(args.run_dir, "benchmarks.json")
samples_dir = os.path.join(args.run_dir, "samples")
shots = {spec.split(":")[0]: int(spec.split(":")[1]) if ":" in spec else args.num_fewshot for spec in args.tasks.split(",")}
tasks = list(shots)
pareto_path = os.path.join(args.run_dir, "pareto.json")
hw = "genes" in config  # a compress_search_hw run: formats per layer group, MLP pruning, sublayer removal
if hw:
    import pandas as pd
    rows = pd.read_csv(os.path.join(args.run_dir, "front.csv"))
    rows = rows[rows["gen"] == rows["gen"].max()]
    calib_base = float((rows["loss"] - rows["delta_loss"]).iloc[0])
    rows = rows[~rows["key"].str.fullmatch(r"0(\.0)*")]  # all genes 0: the uncompressed model itself
    scores = json.load(open(os.path.join(args.run_dir, "test_scores.json")))
    base = scores["baseline"]
    base["size_mb"] = base["memory_mb"]
    tested = {p["key"]: p for p in scores["pareto"]}  # only some of the front has test perplexities
    front = []
    for r in rows.sort_values("latency_ms").to_dict("records"):
        p = {k: v.item() if hasattr(v, "item") else v for k, v in r.items()}
        p.update({k: v for k, v in tested.get(p["key"], {}).items() if k.endswith(("_loss", "_ppl"))})
        p.update(calib_delta_loss=p["delta_loss"], size_mb=p["memory_mb"], size_ratio=p["memory_mb"] / base["memory_mb"],
                 target_bits_per_weight=p["bits_per_weight"])
        front.append(p)
elif os.path.exists(pareto_path):
    pareto = json.load(open(pareto_path))
    front = pareto["pareto"]  # sorted by size
    base = pareto["baseline"]
    calib_base = base["last_batch_loss"]
    for p in front:
        p["calib_delta_loss"] = p["last_batch_delta_loss"]
else:  # a search run without test windows: the front scored by scripts.compress_score
    scores = json.load(open(os.path.join(args.run_dir, "test_scores.json")))
    front = sorted(scores["pareto"], key=lambda p: p["size_mb"])
    base = scores["baseline"]
    with open(os.path.join(args.run_dir, "batch_baselines.csv")) as f:
        calib_base = float(list(csv.DictReader(f))[-1]["baseline_loss"])  # the batch of the last generation


def k_init(p):
    return p.get("k_init", p.get("k"))  # older runs called the grid size "k"


def ppl(p, name):
    return p.get(f"{name}_ppl", float("nan"))


def calib_error_pct(p):
    """Calibration loss above the uncompressed model's, in percent of it."""
    return 100 * p["calib_delta_loss"] / calib_base


if args.list and hw:
    print(f"{'idx':>4} {'decode ms':>9} {'memory MB':>9} {'bits/w':>7} {'calib err':>9} {'wiki2 ppl':>10} {'c4 ppl':>9}  config")
    print(f"{'-':>4} {base['latency_ms']:9.2f} {base['memory_mb']:9.1f} {16:7.2f} {0:8.2f}% {ppl(base, 'wikitext2'):10.2f} "
          f"{ppl(base, 'c4'):9.2f}  (uncompressed)")
    for i, p in enumerate(front):
        print(f"{i:4d} {p['latency_ms']:9.2f} {p['memory_mb']:9.1f} {p['bits_per_weight']:7.2f} {calib_error_pct(p):8.2f}% "
              f"{ppl(p, 'wikitext2'):10.2f} {ppl(p, 'c4'):9.2f}  {p['config']}")
    raise SystemExit
if args.list:
    print(f"{'idx':>4} {'k':>4} {'used':>6} {'c':>6} {'alpha':>7} {'beta':>7} {'size MB':>8} {'ratio':>6} {'bits/w':>7} "
          f"{'sparsity':>8} {'calib err':>9} {'wiki2 ppl':>10} {'c4 ppl':>9}")
    print(f"{'-':>4} {'-':>4} {'-':>6} {'-':>6} {'-':>7} {'-':>7} {base['size_mb']:8.1f} {1:6.1%} {16:7.2f} {0:7.1f}% "
          f"{0:8.2f}% {ppl(base, 'wikitext2'):10.2f} {ppl(base, 'c4'):9.2f}   (uncompressed)")
    for i, p in enumerate(front):
        print(f"{i:4d} {k_init(p):4d} {p.get('k_used_mean', float('nan')):6.1f} {p['c']:6.3f} {p['alpha']:+7.3f} {p['beta']:+7.3f} {p['size_mb']:8.1f} "
              f"{p['size_ratio']:6.1%} {p['target_bits_per_weight']:7.2f} {p['sparsity_pct']:7.1f}% "
              f"{calib_error_pct(p):8.2f}% {ppl(p, 'wikitext2'):10.2f} {ppl(p, 'c4'):9.2f}")
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
if args.calib_error_pct is not None:
    chosen = set()
    for level in map(float, args.calib_error_pct.split(",")):
        within = [(i, p) for i, p in points if calib_error_pct(p) <= level]
        chosen.add(min(within, key=lambda ip: ip[1]["latency_ms" if hw else "size_mb"])[0] if within else
                   min(points, key=lambda ip: ip[1]["calib_delta_loss"])[0])
    points = [(i, p) for i, p in points if i in chosen]


def point_id(p):
    if hw and p is not None:
        return f"hw {p['key']}"
    rho ="" if p is None or p.get("rho", 1.0) == 1.0 else f" rho={p['rho']:.6f}"  # no rho: the ids of runs from before it
    return "uncompressed" if p is None else f"k={k_init(p)} c={p['c']:.6f} alpha={p['alpha']:.6f} beta={p['beta']:.6f}" + rho


def samples_path(p):
    if p is None:
        name = "uncompressed"
    elif hw:
        name = "hw_" + p["key"].replace(".", "-")
    else:
        rho = "" if p.get("rho", 1.0) == 1.0 else f"_r{p['rho']:.4f}"
        name = f"k{k_init(p)}_c{p['c']:.4f}_a{p['alpha']:.4f}_b{p['beta']:.4f}" + rho
    return os.path.join(samples_dir, name + ".jsonl")


def describe(p):
    """One line: what the solution is."""
    if hw:
        return f"{p['config']}, {p['latency_ms']:.2f} ms, {p['memory_mb']:.1f} MB ({p['size_ratio']:.1%})"
    return (f"k={k_init(p)} c={p['c']:.3f} alpha={p['alpha']:+.3f} beta={p['beta']:+.3f}, "
            f"{p['size_mb']:.1f} MB ({p['size_ratio']:.1%}), sparsity {p['sparsity_pct']:.1f}%")


settings = {"tasks": args.tasks.split(","), "num_fewshot": args.num_fewshot, "limit": args.limit, "chat_template": args.chat_template}
results = json.load(open(out_path)) if os.path.exists(out_path) else {"settings": settings, "results": {}}
assert {k: results["settings"][k] for k in settings} == settings, \
    f"{out_path} was made with different settings: {results['settings']}"
if args.reference_from and "uncompressed" not in results["results"]:
    ref_results = json.load(open(os.path.join(args.reference_from, "benchmarks.json")))
    ref_model = json.load(open(os.path.join(args.reference_from, "config.json")))["args"]["model"]
    assert ref_model == search["model"] and {k: ref_results["settings"][k] for k in settings} == settings, \
        f"--reference-from {args.reference_from} benchmarked {ref_model} with {ref_results['settings']}"
    results["results"]["uncompressed"] = ref_results["results"]["uncompressed"]
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(args.reference_from, "samples", "uncompressed.jsonl"), encoding="utf-8") as src, \
            open(samples_path(None), "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"Uncompressed-model scores from {args.reference_from}", flush=True)
todo =[(i, p) for i, p in [(None, None), *points] if point_id(p) not in results["results"]]
print(f"{len(points)} solutions selected, {len(todo)} evaluations to run (incl. the uncompressed model) on {tasks}", flush=True)


LETTERS = "ABCDEFGHIJ"


def simplify(task, s):
    """One question as a small readable record."""
    doc = s["doc"]
    question = next((doc[key] for key in ("question", "goal", "ctx", "sentence", "query") if key in doc), None)
    rec = {"task": task, "doc_id": s["doc_id"], "filter": s["filter"], "question": question,
           "correct": bool(s[s["metrics"][0]])}
    if len(s["filtered_resps"]) > 1:  # multiple choice: the answer is the choice with the highest log-likelihood
        loglik = [r[0] for r in s["filtered_resps"]]
        if isinstance(doc.get("choices"), list):  # MMLU
            choices = doc["choices"]
        elif "option1" in doc:  # WinoGrande: the choices differ in the context, not the continuation
            choices = [doc["option1"], doc["option2"]]
        else:
            choices = [str(a[1]).strip() for a in s["arguments"]]
        target = s["target"]
        rec.update(choices=choices, answer=LETTERS[target] if isinstance(target, int) else str(target),
                   predicted=LETTERS[int(np.argmax(loglik))], choice_logliks=loglik)
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

    from nanochat import compress_hw
    from nanochat.compress import Candidate, GlobalCompressor, include_from_config

    results["settings"]["lm_eval_version"] = lm_eval.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(search["model"])
    model = AutoModelForCausalLM.from_pretrained(search["model"], dtype=getattr(torch, search["dtype"])).to(device).eval()
    if hw:  # rebuilt from what the search saved, so every solution gets the same clipping and removal order
        removal = json.load(open(os.path.join(args.run_dir, "removal.json")))
        compressor = compress_hw.HWCompressor(
            model, compress_hw.Layout(model), json.load(open(os.path.join(args.run_dir, "latency_table.json"))),
            torch.load(os.path.join(args.run_dir, "act_stats.pt"))["stats"], {k: removal[k] for k in ("attn", "mlp")},
            formats=search["formats"].split(","), bands=search["bands"],
            mlp_keep=sorted({float(k) for k in search["mlp_keep"].split(",")} | {1.0}, reverse=True),
            max_skip=search["max_skip"], compress_embeddings=not search["no_embeddings"],
            clip_ratios=[float(r) for r in search["clip_ratios"].split(",")], offload=search["offload_originals"])
        gene_names = [name for name, _ in compressor.genes]
        apply = lambda p: compressor.apply(compressor.config(compressor.decode([p[n] for n in gene_names])))
    else:
        compressor = GlobalCompressor(model, include_from_config(search), search["size_mode"], search["offload_originals"],
                                      search["reconstruction"], formats=search["formats"].split(","),
                                      prune=not search.get("no_pruning", False), method=search.get("method", "ubq"))
        apply = lambda p: compressor.apply(Candidate(k_init(p), p["c"], p["alpha"], p["beta"], p.get("rho", 1.0)))
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)  # sees the in-place weight changes
    os.makedirs(samples_dir, exist_ok=True)

    for idx, p in todo:
        t0 = time.time()
        entry = {"front_index": idx}
        if p is not None:
            entry.update({key: p[key] for key in ("b", "k_init", "k", "k_used_mean", "c", "alpha", "beta", "rho", "size_mb",
                                                  "size_ratio", "target_bits_per_weight", "sparsity_pct",
                                                  "calib_delta_loss", "key", "config", "latency_ms", "memory_mb",
                                                  "int4_pct", "mlp_kept_pct", "n_removed_attn", "n_removed_mlp") if key in p})
            entry["calib_error_pct"] = calib_error_pct(p)
        try:
            if p is not None:
                apply(p)
            out = {"results": {}, "samples": {}}
            for n in sorted(set(shots.values())):  # one lm-eval call per number of shots
                o = lm_eval.simple_evaluate(model=lm, tasks=[t for t in tasks if shots[t] == n], num_fewshot=n,
                                            limit=args.limit, log_samples=True, apply_chat_template=args.chat_template,
                                            fewshot_as_multiturn=args.chat_template, random_seed=0, numpy_random_seed=0,
                                            torch_random_seed=0, fewshot_random_seed=0)
                out["results"].update(o["results"])
                out["samples"].update(o["samples"])
        finally:
            compressor.restore()
        # headline metrics: GSM8K exact match (both answer extractions), MMLU accuracy, with stderr
        entry["benchmarks"] = {task: {m: v for m, v in out["results"][task].items()
                                      if isinstance(v, (int, float)) and not m.startswith("sample_")}
                               for task in tasks}
        entry["seconds"] = round(time.time() - t0)
        results["results"][point_id(p)] = entry  # saved before the samples: the scores are what took hours
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, out_path)
        with open(samples_path(p), "w", encoding="utf-8") as f:
            for task, samples in out["samples"].items():
                for s in samples:
                    f.write(json.dumps(simplify(task, s), ensure_ascii=False) + "\n")
        summary = ", ".join(f"{t}: " + " ".join(f"{m}={v:.3f}" for m, v in r.items() if "stderr" not in m)
                            for t, r in entry["benchmarks"].items())
        print(f"[{'uncompressed' if p is None else f'#{idx}'}] {entry.get('size_mb', base['size_mb']):.1f} MB | "
              f"{summary} | {entry['seconds']}s", flush=True)

# -----------------------------------------------------------------------------
# accuracy table (%) of every evaluated solution, next to the uncompressed model

TASK_NAMES = {"piqa": "PIQA", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "hellaswag": "HellaSwag",
              "winogrande": "WinoGrande", "boolq": "BoolQ", "openbookqa": "OBQA", "lambada_openai": "LAMBADA",
              "mmlu": "MMLU", "gsm8k": "GSM8K"}
METRICS = ("acc_norm,none", "acc,none", "exact_match,flexible-extract")  # the first one a task reports


def score(entry, task):
    """(metric, accuracy in percent, its standard error in percent) of one task."""
    r = entry["benchmarks"][task]
    m = next(m for m in METRICS if m in r)
    return m, 100 * r[m], 100 * r.get(m.replace(",", "_stderr,", 1), float("nan"))


ref = results["results"].get("uncompressed")
rows = sorted((e for k, e in results["results"].items() if k != "uncompressed"),
              key=lambda e: -e["latency_ms" if hw else "size_mb"])
if ref is not None:
    header = ["Model", "Calib. loss vs. uncompressed", *(["Decode (ms)", "Faster by", "Memory (MB)"] if hw else ["Size (MB)"]),
              "Smaller by", "Bits/weight", *[f"{TASK_NAMES.get(t, t)} ({shots[t]}-shot)" for t in tasks], "Average",
              "Accuracy kept"]
    ref_avg = np.mean([score(ref, t)[1] for t in tasks])
    table = []
    for e in [ref, *rows]:
        accs = [score(e, t)[1] for t in tasks]
        compressed = e is not ref
        speed = [f"{e['latency_ms'] if compressed else base['latency_ms']:.2f}",
                 f"{base['latency_ms'] / e['latency_ms']:.2f}x" if compressed else "1.00x"] if hw else []
        table.append(["#%d" % e["front_index"] if compressed else f"Uncompressed ({search['dtype']})",
                      f"+{e['calib_error_pct']:.2f}%" if "calib_error_pct" in e else "-",  # older benchmarks.json: not stored
                      *speed,
                      f"{e['size_mb'] if compressed else base['size_mb']:.0f}",
                      f"{1 / e['size_ratio']:.2f}x" if compressed else "1.00x",
                      f"{e['target_bits_per_weight']:.2f}" if compressed else "16.00",
                      *[f"{a:.1f}" for a in accs], f"{np.mean(accs):.1f}", f"{100 * np.mean(accs) / ref_avg:.1f}%"])
    notes = ["Scores are accuracy in percent (higher is better); 'Accuracy kept' is the average relative to the "
             "uncompressed model's. 'Calib. loss vs. uncompressed' is how much higher (in percent) each solution's loss "
             "on the search's calibration data is." +
             (f" Decode: predicted GPU time of one decode step ({search['latency_batch']} sequence, "
              f"{search['latency_context']} tokens of context); memory: weights plus that KV cache." if hw else ""),
             "Metrics: " + "; ".join(f"{TASK_NAMES.get(t, t)} {score(ref, t)[0].split(',')[0].replace('exact_match', 'exact match')}"
                                     f" (±{score(ref, t)[2]:.1f} standard error)" for t in tasks) + "."]
    md = "\n".join([f"# {search['model']}: benchmarks of compressed models", "",
                    "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                    *["| " + " | ".join(row) + " |" for row in table], "", *notes, ""])
    with open(os.path.join(args.run_dir, "benchmarks.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.run_dir, "benchmarks.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *table])
    print("\n" + md, flush=True)

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
        print(f"\n{'=' * 100}\n#{idx}: {describe(p)}")
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
                    print("  " + "  ".join(f"{l}) {clip(ch, 80)}" for l, ch in zip(LETTERS, r["choices"])))
                print(f"  correct answer: {r['answer']}")
                print(f"  uncompressed:   {r['predicted']}" + (f"   <- {clip(r['generation'], 300)}" if "generation" in r else ""))
                print(f"  compressed:     {c['predicted']}" + (f"   <- {clip(c['generation'], 300)}" if "generation" in c else ""))
print(f"\nResults in {out_path}, questions in {samples_dir}", flush=True)
