"""
Downstream benchmarks (lm-evaluation-harness) for the turn-time searches: scripts/compress_search_mix.py (weight sharing
with the deployed formats), scripts/compress_search_mix_simple.py and scripts/compress_search_kv2.py (the KV cache).
scripts/compress_benchmark.py covers compress_search (UBQ-ECSQ) and compress_search_hw; it cannot rebuild these runs'
compressors (the `shared` format and its genes, the cache genes).

Each chosen solution is written into the model in place by the compressor of its run, rebuilt from what the run saved and
checked against its genes, evaluated, then restored. The uncompressed model is evaluated first (or taken from
--reference-from: the same model benchmarked with the same tasks, e.g. compress_runs/qwen_hw_3obj_p50_g100).

Which solutions: for each level of --calib-error-pct, the solution of the front with the least --pick (default: the turn
time) among those whose calibration loss is at most that many percent above the uncompressed model's (none within the
level: the most accurate one). The front is the final generation of the run's front.csv, or --front, a CSV of feasible
solutions with the run's gene columns (scripts/compress_kv2_recheck.py writes recheck/front.csv: the kv2 front re-timed
with the current kernel). Time and memory are recomputed by the rebuilt compressor, so every speedup comes from one table;
for kv2 the KV2 table is the kv2_table.json next to the front file if there is one (the re-check's), else the run's.

The KV cache (kv2 runs):
    precisions  keys and values are quantized as the search scores them, also while generating (GSM8K): kv2_attention
                reads the earlier tokens from the cache and quantizes them as a whole window would
    windows     --kv-window context (default): as deployed, a window is a fraction of the run's --context (32000 tokens).
                Every benchmark sequence is shorter than the smallest window (8001 tokens at 0.25), so windows change
                nothing here and only the precisions count; each forward pass checks that. --kv-window sequence: the
                search's proxy, a fraction of each sequence (much harsher on prompts of a few hundred tokens)
    padding     batches are padded (generation pads on the left), so the attention gets the padding mask
                (compress_kv2.register_padding_mask); the key groups then count from the first padded entry

Outputs in <run-dir> (or --out-dir):
    benchmarks.json       scores per solution, updated after every solution, so an interrupted run resumes
    benchmarks.md/.csv    accuracy table (%) of every evaluated solution
    samples/<point>.jsonl every question: the question, the correct answer, the model's answer and whether it was right

Needs lm-eval, in its own environment (see scripts/compress_benchmark.py). Run from the repo root, e.g. with the tasks
and levels of the A100 benchmarks of compress_runs/qwen_hw_3obj_p50_g100:

    .venv-eval/bin/python -m scripts.compress_benchmark_mix --run-dir compress_runs/qwen_mix_simple --list
    .venv-eval/bin/python -m scripts.compress_benchmark_mix --run-dir compress_runs/qwen_mix_simple --calib-error-pct 0,1,2,5,10 \
        --num-fewshot 0 --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu:5,gsm8k:5 \
        --reference-from compress_runs/qwen_hw_3obj_p50_g100
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # questions contain symbols the Windows console codepage lacks

parser = argparse.ArgumentParser(description="lm-eval benchmarks for solutions of a mix, mix_simple or kv2 search")
parser.add_argument("--run-dir", type=str, required=True, help="output directory of the search (config.json, front.csv, the tables)")
parser.add_argument("--front", type=str, default=None, help="CSV of feasible solutions (default: the final generation of <run-dir>/front.csv)")
parser.add_argument("--out-dir", type=str, default=None, help="default: --run-dir")
parser.add_argument("--list", action="store_true", help="print the front with indices and exit (no GPU work)")
parser.add_argument("--dry-run", action="store_true", help="print the solutions that would be evaluated and exit (no GPU work)")
parser.add_argument("--indices", type=str, default=None, help="comma-separated indices from --list")
parser.add_argument("--calib-error-pct", type=str, default=None, help="comma-separated levels, e.g. 0,1,2,5,10 (see above)")
parser.add_argument("--pick", type=str, default="latency_ms", help="comma-separated: latency_ms, memory_mb or size_mb; one solution per level each")
parser.add_argument("--kv-window", type=str, default="context", choices=["context", "sequence"], help="what a window is a fraction of (kv2 runs)")
parser.add_argument("--reference-from", type=str, default=None, help="run directory with benchmarks.json of the uncompressed model on the same tasks")
parser.add_argument("--tasks", type=str, default="gsm8k,mmlu", help="comma-separated lm-eval tasks, each optionally task:shots")
parser.add_argument("--num-fewshot", type=int, default=5, help="shots of tasks without :shots")
parser.add_argument("--limit", type=int, default=None, help="examples per task, per MMLU subject (default: all); for quick checks only")
parser.add_argument("--batch-size", type=str, default="4", help="an integer, or auto (only on an idle GPU)")
parser.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template")
args = parser.parse_args()

run_path = lambda name: os.path.join(args.run_dir, name)
out_dir = args.out_dir or args.run_dir
os.makedirs(out_dir, exist_ok=True)
out_path, samples_dir = os.path.join(out_dir, "benchmarks.json"), os.path.join(out_dir, "samples")
config = json.load(open(run_path("config.json")))
a = config["args"]
names = [n for n, _ in config["genes"]]
if any(n.startswith("key_") for n in names):
    assert os.path.exists(run_path("kv2_table.json")), \
        "a KV search without kv2_table.json: the first one (compress_search_kv), whose plain windows lose token 0; not supported"
    kind = "kv2"
else:
    assert "keep_b0" in names and "prompt_table.json" in os.listdir(args.run_dir), \
        "not a turn-time search: use scripts/compress_benchmark.py for compress_search and compress_search_hw runs"
    kind = "mix" if any(n.startswith("share_") for n in names) else "mix_simple"
shots = {spec.split(":")[0]: int(spec.split(":")[1]) if ":" in spec else args.num_fewshot for spec in args.tasks.split(",")}
tasks = list(shots)
base_loss = config["baseline"]["loss"]

# --- the front

if args.front is None:
    rows = pd.read_csv(run_path("front.csv"))
    rows = rows[rows["gen"] == rows["gen"].max()]
    front_file = run_path("front.csv")
else:
    rows, front_file = pd.read_csv(args.front), args.front
rows = rows.drop_duplicates("key")
rows = rows[~rows[names].eq(0).all(axis=1)]  # all genes 0: the uncompressed model itself
if os.path.exists(run_path("test_scores.json")):  # test perplexities of the part of the front that has them
    tested = {p["key"]: p for p in json.load(open(run_path("test_scores.json")))["pareto"]}
else:
    tested = {}
front = []
for r in rows.to_dict("records"):
    p = {k: v.item() if hasattr(v, "item") else v for k, v in r.items()}
    p["genes"] = tuple(int(p[n]) for n in names)
    p["calib_error_pct"] = 100 * p["delta_loss"] / base_loss
    p.update({k: v for k, v in tested.get(p["key"], {}).items() if k.endswith(("_loss", "_ppl")) and k not in ("loss", "delta_loss")})
    front.append(p)

# --- the compressor of the run (the costs need only its tables, so --list builds it on the CPU)

import torch  # noqa: E402

from nanochat import compress_hw, compress_kv2, compress_mix, compress_models  # noqa: E402

device = "cuda" if torch.cuda.is_available() and not (args.list or args.dry_run) else "cpu"
model, tokenizer, _ = compress_models.load(a["model"], a["dtype"], device)
L = compress_hw.Layout(model)
decode_table, prompt_table = json.load(open(run_path("latency_table.json"))), json.load(open(run_path("prompt_table.json")))
act_sq = torch.load(run_path("act_stats.pt"), map_location=device)["stats"]
if a["max_skip"] > 0:
    removal = json.load(open(run_path("removal.json")))
    removal = {k: removal[k] for k in ("attn", "mlp")}
else:  # nothing is removed: the order does not matter
    removal = {k: [(i, 0.0) for i in range(L.n_layer)] for k in ("attn", "mlp")}
options = dict(new_tokens=a["new_tokens"], gen_tokens=a["gen_tokens"], vram_gb=a["vram_gb"], reserve=a["reserve"],
               formats=[f for f in a["formats"].split(",") if f], bands=a["bands"],
               mlp_keep=sorted({float(k) for k in a["mlp_keep"].split(",")} | {1.0}, reverse=True), max_skip=a["max_skip"],
               compress_embeddings=not a["no_embeddings"], clip_ratios=[float(r) for r in a["clip_ratios"].split(",")],
               offload="no" if device == "cpu" else a["offload_originals"])
if kind == "kv2":
    table_file = os.path.join(os.path.dirname(os.path.abspath(front_file)), "kv2_table.json")
    table_file = table_file if os.path.exists(table_file) else run_path("kv2_table.json")
    compressor = compress_kv2.KV2Compressor(model, L, decode_table, prompt_table, json.load(open(table_file)), act_sq, removal,
                                            context=a["context"], **options)
else:
    table_file = run_path("latency_table.json")
    compressor = compress_mix.MixCompressor(model, L, decode_table, prompt_table, act_sq, removal, sharing=kind == "mix", **options)
assert [list(g) for g in compressor.genes] == [list(g) for g in config["genes"]], "the genes differ from the run's: other settings or code"
base_cfg = compressor.config(compressor.uniform_genes("bf16"))
base = {**compressor.cost(base_cfg), "size_mb": config["baseline"]["size_mb"]}
for p in front:
    for k, v in compressor.cost(compressor.config(p["genes"])).items():
        if k in ("latency_ms", "decode_us", "prompt_us", "memory_mb", "weights_mb", "kv_mb", "agents_fit"):
            p[k] = v
front.sort(key=lambda p: p["latency_ms"])
print(f"{args.run_dir}: a {kind} search, {len(front)} solutions from {front_file}, times from {table_file}; uncompressed turn "
      f"{base['latency_ms']:.0f} ms, {base['memory_mb']:.0f} MB deployed, {base['size_mb']:.0f} MB stored", flush=True)

if args.list:
    recall = "recall_acc" in front[0] if front else False
    print(f"{'idx':>4} {'turn ms':>14} {'memory MB':>15} {'stored MB':>9} {'calib loss':>10}{' recall' if recall else ''} "
          f"{'c4 ppl':>7}  config")
    for i, p in enumerate(front):
        print(f"{i:4d} {p['latency_ms']:6.0f} ({base['latency_ms'] / p['latency_ms']:4.2f}x) {p['memory_mb']:6.0f} "
              f"({base['memory_mb'] / p['memory_mb']:5.2f}x) {p['size_mb']:9.0f} {p['calib_error_pct']:9.2f}%"
              + (f" {p['recall_acc']:6.0%}" if recall else "") + f" {p.get('c4_ppl', float('nan')):7.2f}  {p['config']}")
    raise SystemExit

# --- which solutions

points = list(enumerate(front))
why = {}
if args.indices is not None:
    points = [(i, front[i]) for i in map(int, args.indices.split(","))]
if args.calib_error_pct is not None:
    for level in map(float, args.calib_error_pct.split(",")):
        within = [(i, p) for i, p in points if p["calib_error_pct"] <= level]
        for col in args.pick.split(","):
            i = min(within, key=lambda ip: ip[1][col])[0] if within else min(points, key=lambda ip: ip[1]["delta_loss"])[0]
            why.setdefault(i, []).append(f"least {col} within {level:g}%" if within else f"most accurate (none within {level:g}%)")
    points = [(i, p) for i, p in points if i in why]


def point_id(p):
    return "uncompressed" if p is None else f"{kind} {p['key']}"


def samples_path(p):
    return os.path.join(samples_dir, ("uncompressed" if p is None else f"{kind}_" + p["key"].replace(".", "-")) + ".jsonl")


SHARED_SETTINGS = ("tasks", "num_fewshot", "limit", "chat_template")  # those of compress_benchmark.py
settings = {"tasks": args.tasks.split(","), "num_fewshot": args.num_fewshot, "limit": args.limit, "chat_template": args.chat_template,
            "kind": kind, "front": os.path.relpath(front_file, args.run_dir), "kv_window": args.kv_window if kind == "kv2" else None}
results = json.load(open(out_path)) if os.path.exists(out_path) else {"settings": settings, "results": {}}
assert {k: results["settings"].get(k) for k in settings} == settings, f"{out_path} was made with different settings: {results['settings']}"
if args.reference_from and "uncompressed" not in results["results"]:
    ref_results = json.load(open(os.path.join(args.reference_from, "benchmarks.json")))
    ref_model = json.load(open(os.path.join(args.reference_from, "config.json")))["args"]["model"]
    assert ref_model == a["model"] and all(ref_results["settings"][k] == settings[k] for k in SHARED_SETTINGS), \
        f"--reference-from {args.reference_from} benchmarked {ref_model} with {ref_results['settings']}"
    results["results"]["uncompressed"] = ref_results["results"]["uncompressed"]
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(args.reference_from, "samples", "uncompressed.jsonl"), encoding="utf-8") as src, \
            open(samples_path(None), "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"Uncompressed-model scores from {args.reference_from}", flush=True)
todo = [(i, p) for i, p in [(None, None), *points] if point_id(p) not in results["results"]]
print(f"{len(points)} solutions selected, {len(todo)} evaluations to run (incl. the uncompressed model) on {tasks}", flush=True)
for i, p in points:
    print(f"  #{i}: {'; '.join(why.get(i, ['--indices']))} | calib loss {p['calib_error_pct']:+.2f}%, turn {p['latency_ms']:.0f} ms, "
          f"{p['memory_mb']:.0f} MB | {p['config']}", flush=True)
if args.dry_run:
    raise SystemExit

LETTERS = "ABCDEFGHIJ"


def simplify(task, s):
    """One question as a small readable record (as compress_benchmark.py stores it)."""
    doc = s["doc"]
    question = next((doc[key] for key in ("question", "goal", "ctx", "sentence", "query") if key in doc), None)
    rec = {"task": task, "doc_id": s["doc_id"], "filter": s["filter"], "question": question, "correct": bool(s[s["metrics"][0]])}
    if len(s["filtered_resps"]) > 1:  # multiple choice: the answer is the choice with the highest log-likelihood
        loglik = [r[0] for r in s["filtered_resps"]]
        if isinstance(doc.get("choices"), list):  # MMLU
            choices = doc["choices"]
        elif "option1" in doc:  # WinoGrande
            choices = [doc["option1"], doc["option2"]]
        else:
            choices = [str(x[1]).strip() for x in s["arguments"]]
        target = s["target"]
        rec.update(choices=choices, answer=LETTERS[target] if isinstance(target, int) else str(target),
                   predicted=LETTERS[int(np.argmax(loglik))], choice_logliks=loglik)
    else:  # generation (GSM8K)
        rec.update(answer=str(s["target"]).split("####")[-1].strip(), predicted=s["filtered_resps"][0],
                   generation=s["resps"][0][0], reference_solution=s["target"])
    return rec


def context_windows(cfg):
    """--kv-window context: each layer keeps its whole sequence, which is right while every sequence fits in the smallest
    window of the configuration; a forward hook checks that. Returns the hook's handle."""
    floor = min(compress_kv2.kept_tokens(w, a["context"]) for w in cfg.window)
    for i in range(L.n_layer):
        attn = getattr(L.module(L.layers[i]), L.attn_attr)
        if hasattr(attn, "kv2"):
            attn.kv2 = (attn.kv2[0], attn.kv2[1], 1.0, attn.kv2[3])

    def check(module, f_args, f_kwargs):
        mask, ids, past = f_kwargs.get("attention_mask"), f_kwargs.get("input_ids", f_args[0] if f_args else None), f_kwargs.get("past_key_values")
        n = mask.shape[-1] if mask is not None else ids.shape[-1] + (past.get_seq_length() if past is not None else 0)
        assert n <= floor, f"a {n}-token sequence is longer than the smallest window ({floor} tokens): use --kv-window sequence"

    return model.register_forward_pre_hook(check, with_kwargs=True)


if todo:
    assert device == "cuda", "the int4 kernels and clipping statistics need a CUDA GPU"
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if kind == "kv2":
        compress_kv2.register_padding_mask()
    results["settings"]["lm_eval_version"] = lm_eval.__version__
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)  # sees the in-place changes
    os.makedirs(samples_dir, exist_ok=True)
    for idx, p in todo:
        t0, hook = time.time(), None
        entry = {"front_index": idx}
        if p is not None:
            entry.update({k: p[k] for k in ("key", "config", "delta_loss", "calib_error_pct", "latency_ms", "decode_us", "prompt_us",
                                            "memory_mb", "weights_mb", "kv_mb", "agents_fit", "size_mb", "bits_per_weight", "int4_pct",
                                            "shared_pct", "mlp_kept_pct", "recall_acc", "kv_pct", "window_pct", "c4_ppl", "wikitext2_ppl")
                          if k in p})
            entry["picked"] = why.get(idx, ["--indices"])
        try:
            if p is not None:
                cfg = compressor.config(p["genes"])
                compressor.apply(cfg)
                if kind == "kv2" and args.kv_window == "context" and model.config._attn_implementation == compress_kv2.IMPL:
                    hook = context_windows(cfg)
            out = {"results": {}, "samples": {}}
            for n in sorted(set(shots.values())):  # one lm-eval call per number of shots
                o = lm_eval.simple_evaluate(model=lm, tasks=[t for t in tasks if shots[t] == n], num_fewshot=n, limit=args.limit,
                                            log_samples=True, apply_chat_template=args.chat_template, fewshot_as_multiturn=args.chat_template,
                                            random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
                out["results"].update(o["results"])
                out["samples"].update(o["samples"])
        finally:
            if hook is not None:
                hook.remove()
            compressor.restore()
        entry["benchmarks"] = {task: {m: v for m, v in out["results"][task].items() if isinstance(v, (int, float)) and not m.startswith("sample_")}
                               for task in tasks}
        entry["seconds"] = round(time.time() - t0)
        results["results"][point_id(p)] = entry  # saved before the samples: the scores are what took hours
        with open(out_path + ".tmp", "w") as f:
            json.dump(results, f, indent=2)
        os.replace(out_path + ".tmp", out_path)
        with open(samples_path(p), "w", encoding="utf-8") as f:
            for task, samples in out["samples"].items():
                for s in samples:
                    f.write(json.dumps(simplify(task, s), ensure_ascii=False) + "\n")
        summary = ", ".join(f"{t}: " + " ".join(f"{m}={v:.3f}" for m, v in r.items() if "stderr" not in m) for t, r in entry["benchmarks"].items())
        print(f"[{'uncompressed' if p is None else f'#{idx}'}] {summary} | {entry['seconds']}s", flush=True)

# --- accuracy table (%) of every evaluated solution, next to the uncompressed model

TASK_NAMES = {"piqa": "PIQA", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "hellaswag": "HellaSwag", "winogrande": "WinoGrande",
              "boolq": "BoolQ", "openbookqa": "OBQA", "lambada_openai": "LAMBADA", "mmlu": "MMLU", "gsm8k": "GSM8K"}
METRICS = ("acc_norm,none", "acc,none", "exact_match,flexible-extract")  # the first one a task reports


def score(entry, task):
    """(metric, accuracy in percent, its standard error in percent) of one task."""
    r = entry["benchmarks"][task]
    m = next(m for m in METRICS if m in r)
    return m, 100 * r[m], 100 * r.get(m.replace(",", "_stderr,", 1), float("nan"))


ref = results["results"].get("uncompressed")
evaluated = sorted((e for k, e in results["results"].items() if k != "uncompressed"), key=lambda e: -e["latency_ms"])
if ref is not None:
    recall = any("recall_acc" in e for e in evaluated)
    header = ["Configuration", "Calib loss (%)", "Turn (ms)", "Memory (MB)", "Stored (MB)", "Bits/weight", *(["Recall (%)"] if recall else []),
              *[f"{TASK_NAMES.get(t, t)} ({shots[t]}-shot)" for t in tasks], "Average", "Accuracy retained (%)"]
    ref_avg = np.mean([score(ref, t)[1] for t in tasks])
    table = []
    for e in [ref, *evaluated]:
        accs = [score(e, t)[1] for t in tasks]
        c = e if e is not ref else {**base, "calib_error_pct": 0.0, "bits_per_weight": 16.0, "config": f"uncompressed ({a['dtype']})",
                                    "recall_acc": config["baseline"].get("recall_acc", float("nan"))}
        table.append([c["config"], f"{c['calib_error_pct']:.2f}", f"{c['latency_ms']:.0f} ({base['latency_ms'] / c['latency_ms']:.2f}x)",
                      f"{c['memory_mb']:.0f} ({base['memory_mb'] / c['memory_mb']:.2f}x)", f"{c['size_mb']:.0f}", f"{c['bits_per_weight']:.2f}",
                      *([f"{100 * c.get('recall_acc', float('nan')):.0f}"] if recall else []),
                      *[f"{x:.1f}" for x in accs], f"{np.mean(accs):.1f}", f"{100 * np.mean(accs) / ref_avg:.1f}"])
    notes = [f"Scores are accuracy in percent. Accuracy retained: the average as a percentage of the uncompressed model's. Calib loss: "
             f"how much higher (%) the loss on the search's calibration batch is. Turn: predicted GPU time of one agent's turn "
             f"({a['new_tokens']} prompt tokens, {a['gen_tokens']} generated, {a['serve_batch']} agents, {a['context']} tokens of "
             f"context), with the speedup over the uncompressed model; memory: deployed weights plus the KV cache, with the "
             f"compression ratio; stored: the stored size of the weights." +
             (f" Recall: planted facts recalled in the search. KV windows: a fraction of the {'context' if args.kv_window == 'context' else 'sequence'}."
              if kind == "kv2" else ""),
             "Metrics: " + "; ".join(f"{TASK_NAMES.get(t, t)} {score(ref, t)[0].split(',')[0].replace('exact_match', 'exact match')}"
                                     f" (±{score(ref, t)[2]:.1f} standard error)" for t in tasks) + "."]
    md = "\n".join([f"# {a['model']}: benchmarks of the {kind} search ({args.run_dir})", "", "| " + " | ".join(header) + " |",
                    "|" + "---|" * len(header), *["| " + " | ".join(row) + " |" for row in table], "", *notes, ""])
    with open(os.path.join(out_dir, "benchmarks.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "benchmarks.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *table])
    print("\n" + md, flush=True)
print(f"\nResults in {out_path}, questions in {samples_dir}", flush=True)
