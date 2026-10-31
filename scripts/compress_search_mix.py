"""
Multi-objective search (NSGA-II, pymoo) that mixes weight sharing + pruning with hardware-aware formats, for the
GPU time of an agent's work. See nanochat/compress_mix.py for the method: each role (qkv, o, up, down) and depth
band, the lm_head and the embeddings take one format (bf16, int4 with a group size, or shared: UBQ-ECSQ), with one
sharing candidate for the whole model, and MLP neurons are pruned per band.

Objectives (all minimized; delta_loss always, --objectives adds the others):
    delta_loss  mean next-token cross-entropy on the calibration batch (C4 train, one fixed batch) minus that of the
                uncompressed model
    time        predicted GPU time (ms) of one agent turn on this GPU: --new-tokens prompt tokens processed in chunks
                of --prompt-batch, then --gen-tokens generated among --serve-batch sequences, over --context tokens
                of cache each. Logged as latency_ms
    size        stored size (MB) of the model, entropy coded. Logged as size_mb. This is what sharing improves: it
                changes neither the time nor the GPU memory
    memory      deployed weights plus the KV cache of --serve-batch agents (MB). Logged as memory_mb
Constraints (optional): --max-delta-loss, --memory-budget-mb. Every solution also logs the agents of this context
that fit on a --vram-gb GPU (agents_fit), and the decode and prompt times per token.

Prepared once and saved in <out-dir> (reused on resume):
    act_stats.pt          E[x^2] of every linear input on the calibration batch (int4 clipping, MLP neuron ranking)
    removal.json          with --max-skip above 0: the loss increase of removing each sublayer alone
    latency_table.json    decode kernel times on this GPU, prompt_table.json the prompt ones (also cached per GPU in
                          $COMPRESS_CACHE; delete both to re-measure). The GPU should otherwise be idle

Outputs in <out-dir>: evals.csv (every evaluation), population.csv and front.csv (after every generation),
generations.csv (hypervolume), test_scores.json (the final front on WikiText-2 test / C4 validation), run.log,
checkpoint.pkl: rerunning the same command resumes. Plots: scripts/compress_plot.py --run-dir <out-dir> --x latency
(the time of a turn) or --x size or --x memory.

    python -m scripts.compress_search_mix --out-dir compress_runs/qwen_mix

    # quick local smoke test
    python -m scripts.compress_search_mix --pop-size 8 --n-gen 2 --calib-seqs 2 --wikitext-seqs 0 --c4-test-seqs 0 --out-dir compress_runs/mix_smoke
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import dill
import numpy as np
import torch
from datasets import load_dataset
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.problems.static import StaticProblem
from pymoo.termination import get_termination
from nanochat import compress_eval, compress_hw, compress_mix, compress_models

parser = argparse.ArgumentParser(description="NSGA-II search mixing weight sharing + pruning with hardware-aware formats")
# model
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HuggingFace model id or local path, or nanochat:<base|sft|rl>:<tag>[:<step>]")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16"], help="the int4 kernel takes bf16 activations")
parser.add_argument("--offload-originals", type=str, default="auto", choices=["auto", "yes", "no"])
parser.add_argument("--eval-batch-size", type=int, default=1)
# calibration: one fixed batch (the same cache as the other searches, so runs share it)
parser.add_argument("--calib-dataset", type=str, default="allenai/c4")
parser.add_argument("--calib-config", type=str, default="en")
parser.add_argument("--calib-split", type=str, default="train")
parser.add_argument("--calib-seqs", type=int, default=8)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--data-seed", type=int, default=0)
parser.add_argument("--calib-from", type=str, default=None, help="reuse another run's calib_pool.pt")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--importance-seqs", type=int, default=4, help="windows for the sublayer removal ranking (only with --max-skip)")
# objectives, constraints and the workload
parser.add_argument("--objectives", nargs="+", default=["time", "size"], choices=["time", "size", "memory"],
                    help="besides delta_loss")
parser.add_argument("--max-delta-loss", type=float, default=1.0, help="constraint: delta_loss above this is infeasible")
parser.add_argument("--memory-budget-mb", type=float, default=None, help="constraint: deployed memory above this is infeasible")
parser.add_argument("--serve-batch", type=int, default=4, help="agents decoding together")
parser.add_argument("--prompt-batch", type=int, default=256, help="prompt tokens processed together (memory of its attention grows with it)")
parser.add_argument("--context", type=int, default=32000, help="tokens each agent keeps in the KV cache")
parser.add_argument("--new-tokens", type=int, default=2000, help="prompt tokens processed per turn (the rest is cached)")
parser.add_argument("--gen-tokens", type=int, default=300, help="tokens generated per turn")
parser.add_argument("--vram-gb", type=float, default=80.0, help="GPU size for agents_fit")
parser.add_argument("--reserve", type=float, default=0.1, help="fraction of the GPU memory kept free in agents_fit")
# search space
parser.add_argument("--formats", type=str, default=",".join(compress_hw.FORMATS), help=f"kernels, from {','.join(compress_hw.FORMATS)}; shared is always added")
parser.add_argument("--bands", type=int, default=4, help="depth bands: one format gene per role and band")
parser.add_argument("--mlp-keep", type=str, default=",".join(map(str, compress_hw.MLP_KEEP)), help="MLP width fractions (1 is always included)")
parser.add_argument("--max-skip", type=int, default=0, help="most attention (and MLP) sublayers removed")
parser.add_argument("--no-embeddings", action="store_true", help="never compress untied embeddings (a tied one follows lm_head)")
parser.add_argument("--clip-ratios", type=str, default=",".join(map(str, compress_hw.CLIP_RATIOS)))
# search
parser.add_argument("--pop-size", type=int, default=50)
parser.add_argument("--n-gen", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
# test sets for the final front (0 windows = skip)
parser.add_argument("--wikitext-seqs", type=int, default=-1, help="WikiText-2 test windows: -1 = all, 0 = skip")
parser.add_argument("--c4-test-seqs", type=int, default=256, help="C4 validation windows, 0 = skip")
parser.add_argument("--test-seq-len", type=int, default=2048)
parser.add_argument("--test-max-delta-loss", type=float, default=1.0, help="only score front solutions with delta_loss up to this")
parser.add_argument("--test-max-points", type=int, default=None, help="score at most this many, spread evenly along the front")
parser.add_argument("--out-dir", type=str, default="compress_runs/mix_default")
args = parser.parse_args()

COST_STAT = {"time": "latency_ms", "size": "size_mb", "memory": "memory_mb"}
STATS = ["loss", "delta_loss", "ppl", "latency_ms", "decode_us", "prompt_us", "size_mb", "memory_mb", "weights_mb", "kv_mb",
         "agents_fit", "bits_per_weight", "int4_pct", "shared_pct", "k_merged", "mlp_kept_pct", "n_removed_attn", "n_removed_mlp"]
GEN_FIELDS = ["gen", "hypervolume", "front_size", "n_feasible", "n_offspring", "n_reused", "best_delta_loss",
              "min_latency_ms", "min_size_mb", "min_memory_mb", "seconds", "peak_gpu_mb"]
# arguments that change what a gene vector means or how it is scored: a resumed run must keep them
FIXED_ARGS = ("model", "objectives", "max_delta_loss", "memory_budget_mb", "serve_batch", "prompt_batch", "context", "new_tokens",
              "gen_tokens", "formats", "bands", "mlp_keep", "max_skip", "no_embeddings", "clip_ratios", "calib_dataset",
              "calib_config", "calib_split", "calib_seqs", "seq_len", "data_seed", "importance_seqs", "pop_size")


class CsvLog:
    """Append-only CSV; on resume, rows from generations after `keep_upto` are dropped."""

    def __init__(self, path, fields, keep_upto=None):
        self.path, self.fields = path, fields
        if os.path.exists(path) and keep_upto is not None:
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["gen"]) <= keep_upto]
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields, extrasaction="ignore")
        if self.file.tell() == 0:
            self.writer.writeheader()

    def write(self, rows):
        self.writer.writerows(rows)
        self.file.flush()


class Tee:
    """Write to the terminal and to <out-dir>/run.log at once."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def atomic_dump(obj, path, dumper):
    tmp = path + ".tmp"
    with open(tmp, "wb" if dumper is dill else "w") as f:
        dumper.dump(obj, f) if dumper is dill else json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def sample_windows(tokenizer, n_seqs):
    """n_seqs windows of seq_len tokens, each at a random offset of a different random document longer than
    seq_len (the GPTQ calibration recipe; the same sampling as scripts.compress_search)."""
    ds = load_dataset(args.calib_dataset, args.calib_config, split=args.calib_split, streaming=True).shuffle(
        seed=args.data_seed, buffer_size=10_000)
    rng = np.random.default_rng(args.data_seed)
    windows = []
    for example in ds:
        ids = tokenizer(example[args.text_field], add_special_tokens=False)["input_ids"]
        if len(ids) <= args.seq_len:
            continue
        start = int(rng.integers(0, len(ids) - args.seq_len + 1))
        windows.append(ids[start:start + args.seq_len])
        if len(windows) == n_seqs:
            return torch.tensor(windows, dtype=torch.long)
    raise ValueError(f"{args.calib_dataset}/{args.calib_split} has fewer than {n_seqs} documents longer than {args.seq_len} tokens")


# -----------------------------------------------------------------------------
# setup

os.makedirs(args.out_dir, exist_ok=True)
out = lambda name: os.path.join(args.out_dir, name)
run_log = open(out("run.log"), "a", encoding="utf-8", buffering=1)
sys.stdout, sys.stderr = Tee(sys.stdout, run_log), Tee(sys.stderr, run_log)
print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.compress_search_mix {' '.join(sys.argv[1:])}", flush=True)
assert torch.cuda.is_available(), "the latency tables and the int4 kernel need a CUDA GPU"
device = torch.device("cuda")
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    git_commit = None
if os.path.exists(out("checkpoint.pkl")) and os.path.exists(out("config.json")):
    old = json.load(open(out("config.json")))["args"]
    for key in FIXED_ARGS:
        assert old.get(key) == getattr(args, key), \
            f"this run was started with --{key.replace('_', '-')} {old.get(key)}, not {getattr(args, key)}: use a new --out-dir"

print(f"Loading {args.model} on {device}", flush=True)
model, tokenizer, _ = compress_models.load(args.model, args.dtype, device)
layout = compress_hw.Layout(model)
print(f"{layout.n_layer} layers, MLP width {layout.intermediate}, {layout.n_head} heads ({layout.n_kv_head} KV), "
      f"lm_head {'tied to the embedding' if layout.tied else 'untied'}, embeddings {layout.embeds or 'none separate'}", flush=True)

# the calibration batch: this run's copy, --calib-from, the shared cache, or a fresh sample (then cached)
pool_meta = {"source": [args.calib_dataset, args.calib_config, args.calib_split], "seed": args.data_seed,
             "seq_len": args.seq_len, "calib_seqs": args.calib_seqs}
pool_cache = compress_eval.cache_path("calib", {**pool_meta, "model": args.model})


def usable_pool(path):
    if not path or not os.path.exists(path):
        return None
    saved = torch.load(path)
    if saved["meta"] != pool_meta or saved["tokens"].size(0) < args.calib_seqs or saved.get("model", args.model) != args.model:
        return None
    return saved["tokens"]


pool = None
for path in (out("calib_pool.pt"), os.path.join(args.calib_from, "calib_pool.pt") if args.calib_from else None, pool_cache):
    pool = usable_pool(path)
    if pool is not None:
        print(f"Calibration batch from {path}", flush=True)
        break
if pool is None:
    print(f"Sampling {args.calib_seqs} x {args.seq_len} calibration tokens from {args.calib_dataset} {args.calib_split}...", flush=True)
    pool = sample_windows(tokenizer, args.calib_seqs)
saved = {"meta": pool_meta, "tokens": pool, "model": args.model}
compress_eval.save_atomic(saved, out("calib_pool.pt"))
if usable_pool(pool_cache) is None:
    compress_eval.save_atomic(saved, pool_cache)
calib = pool[:args.calib_seqs]
eval_loss = lambda tokens: compress_eval.eval_loss(model, tokens, args.eval_batch_size)

t0 = time.time()
baseline_loss = eval_loss(calib)
print(f"Baseline: calib loss {baseline_loss:.4f}, one forward pass over the batch takes {time.time() - t0:.1f}s", flush=True)

# one-time statistics of the uncompressed model, each saved with the settings it was made with
stats_settings = {"model": args.model, "calib": pool_meta}
saved_stats = torch.load(out("act_stats.pt")) if os.path.exists(out("act_stats.pt")) else None
if saved_stats is None or saved_stats["settings"] != stats_settings:
    t0 = time.time()
    act_sq = compress_hw.activation_stats(model, layout, calib, args.eval_batch_size)
    torch.save({"settings": stats_settings, "stats": act_sq}, out("act_stats.pt"))
    print(f"Activation statistics of {len(act_sq)} linear inputs in {time.time() - t0:.0f}s", flush=True)
else:
    act_sq = saved_stats["stats"]

if args.max_skip > 0:
    removal_settings = {**stats_settings, "importance_seqs": args.importance_seqs}
    removal = json.load(open(out("removal.json"))) if os.path.exists(out("removal.json")) else None
    if removal is None or removal["settings"] != removal_settings:
        t0 = time.time()
        print(f"Removal ranking: each of the {2 * layout.n_layer} sublayers removed alone, on {args.importance_seqs} windows", flush=True)
        base, order = compress_hw.removal_ranking(model, layout, calib[:args.importance_seqs], args.eval_batch_size,
                                                  log=lambda m: print(m, flush=True))
        removal = {"settings": removal_settings, "baseline_loss": base, **order}
        atomic_dump(removal, out("removal.json"), json)
        print(f"...done in {time.time() - t0:.0f}s", flush=True)
else:  # nothing is removed: the order does not matter
    removal = {kind: [(i, 0.0) for i in range(layout.n_layer)] for kind in ("attn", "mlp")}

formats = [f for f in args.formats.split(",") if f]
mlp_keep = sorted({float(k) for k in args.mlp_keep.split(",")} | {1.0}, reverse=True)


def latency_table(file, kind, batch, build):
    """A kernel-time table: this run's copy, the per-GPU cache, or measured now (then cached)."""
    settings = {"version": compress_hw.TABLE_VERSION, "kind": kind, "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__), "model": args.model, "batch": batch, "context": args.context,
                "formats": formats, "mlp_keep": mlp_keep}
    cache_file = compress_eval.cache_path("latency", settings)
    table = json.load(open(out(file))) if os.path.exists(out(file)) else None
    if table is None or table.get("settings") != settings:
        if os.path.exists(cache_file):
            table = torch.load(cache_file)
            print(f"{kind} table from {cache_file}", flush=True)
        else:
            t0 = time.time()
            print(f"Measuring the {kind} table on {settings['gpu']}: batch {batch}, {args.context} tokens of context", flush=True)
            table = {"settings": settings, **build(model, layout, formats, mlp_keep, batch, args.context, log=lambda m: print(m, flush=True))}
            compress_eval.save_atomic(table, cache_file)
            print(f"...done in {time.time() - t0:.0f}s", flush=True)
        atomic_dump(table, out(file), json)
    unavailable = sorted(set(formats) - set(table["formats"]))
    assert not unavailable, f"no kernel on this GPU runs {unavailable}: drop them from --formats"
    return table


decode_table = latency_table("latency_table.json", "decode", args.serve_batch, compress_hw.build_latency_table)
prompt_table = latency_table("prompt_table.json", "prompt", args.prompt_batch, compress_mix.build_prompt_table)

compressor = compress_mix.MixCompressor(model, layout, decode_table, prompt_table, act_sq, {k: removal[k] for k in ("attn", "mlp")},
                                        new_tokens=args.new_tokens, gen_tokens=args.gen_tokens, vram_gb=args.vram_gb, reserve=args.reserve,
                                        formats=formats, bands=args.bands, mlp_keep=mlp_keep, max_skip=args.max_skip,
                                        compress_embeddings=not args.no_embeddings,
                                        clip_ratios=[float(r) for r in args.clip_ratios.split(",")], offload=args.offload_originals)
GENE_NAMES = [name for name, _ in compressor.genes]
EVAL_FIELDS = ["gen", "role", "key", "config", *GENE_NAMES, *STATS, "seconds"]
POP_FIELDS = ["gen", "rank", "crowding", "key", "config", *GENE_NAMES, *STATS]
baseline_cfg = compressor.config(compressor.uniform_genes("bf16"))
baseline_cost = {**compressor.cost(baseline_cfg), **compressor.apply(baseline_cfg)}
compressor.restore()
print(f"{len(compressor.genes)} genes ({', '.join(f'{n}:{k}' for n, k in compressor.genes)}), originals "
      f"{'offloaded to CPU' if compressor.offloaded else 'kept on GPU'}", flush=True)
print(f"Uncompressed: an agent's turn takes {baseline_cost['latency_ms']:.0f} ms ({baseline_cost['prompt_us']:.0f} us per prompt token, "
      f"{baseline_cost['decode_us']:.0f} us per generated token), {baseline_cost['size_mb']:.0f} MB stored, "
      f"{baseline_cost['memory_mb']:.0f} MB deployed (KV cache {baseline_cost['kv_mb']:.0f} MB), {baseline_cost['agents_fit']} agents fit", flush=True)
atomic_dump({"args": {**vars(args), "calib_every": 0},  # always one fixed batch (read by compress_compare)
             "git_commit": git_commit, "genes": compressor.genes, "baseline_size_mb": baseline_cost["size_mb"],
             "baseline": {"loss": baseline_loss, "delta_loss": 0.0, **baseline_cost},
             "latency_table": {k: decode_table[k] for k in ("gpu", "torch", "batch", "context", "formats")},
             "layout": {"n_layer": layout.n_layer, "intermediate": layout.intermediate, "tied": layout.tied,
                        "n_target_params": compressor.n_target_params, "rest_params": compressor.rest_params}},
            out("config.json"), json)

# -----------------------------------------------------------------------------
# NSGA-II over integer genes (SBX and polynomial mutation, rounded), with the evaluation in an ask/tell loop

cost_objectives = [COST_STAT[o] for o in args.objectives]
problem = Problem(n_var=len(compressor.genes), n_obj=1 + len(cost_objectives),
                  n_ieq_constr=(args.max_delta_loss is not None) + (args.memory_budget_mb is not None),
                  xl=np.zeros(len(compressor.genes)), xu=np.array([n - 1 for _, n in compressor.genes], dtype=float))


class SeededSampling(Sampling):
    """Random genes, except that each structural gene (MLP width, removed sublayers) starts at 0 (nothing
    pruned) half of the time, and one individual per format is that format everywhere, unpruned: random
    pruning mostly breaks the model, and the uniform ones span the search from safe to cheap."""

    def _do(self, problem, n_samples, *op_args, random_state=None, **kwargs):
        X = random_state.integers(0, problem.xu.astype(int) + 1, size=(n_samples, problem.n_var))
        structural = np.array([n.startswith(("keep_", "skip_")) for n in GENE_NAMES])
        X[:, structural] *= random_state.random((n_samples, int(structural.sum()))) < 0.5
        for i, fmt in enumerate(compressor.formats[:n_samples]):
            X[i] = compressor.uniform_genes(fmt)
        return X.astype(float)


class DecodedDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return compressor.decode(a.X) == compressor.decode(b.X)


if os.path.exists(out("checkpoint.pkl")):
    with open(out("checkpoint.pkl"), "rb") as f:
        algorithm = dill.load(f)
    algorithm.termination = get_termination("n_gen", args.n_gen)
    done_gen = algorithm.n_gen - 1
    algorithm.termination.perc = done_gen / args.n_gen
    print(f"Resumed NSGA-II after generation {done_gen}", flush=True)
else:
    algorithm = NSGA2(pop_size=args.pop_size, sampling=SeededSampling(),
                      crossover=SBX(prob=0.9, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=DecodedDuplicateElimination())
    algorithm.setup(problem, termination=get_termination("n_gen", args.n_gen), seed=args.seed)
    done_gen = 0

key_of = lambda genes: ".".join(map(str, genes))
cache = {}
if os.path.exists(out("evals.csv")):
    with open(out("evals.csv"), newline="") as f:
        for r in csv.DictReader(f):
            cache[r["key"]] = {s: float(r[s]) for s in STATS}
    print(f"Loaded {len(cache)} previous evaluations", flush=True)
evals_log = CsvLog(out("evals.csv"), EVAL_FIELDS)
pop_log = CsvLog(out("population.csv"), POP_FIELDS, keep_upto=done_gen)
front_log = CsvLog(out("front.csv"), POP_FIELDS, keep_upto=done_gen)
gen_log = CsvLog(out("generations.csv"), GEN_FIELDS, keep_upto=done_gen)
norm_path = out("normalization.json")
nadir = np.array(json.load(open(norm_path))["nadir"]) if os.path.exists(norm_path) else None


def evaluate(genes):
    """Calibration loss of the configuration (written into the model, then restored) and its predicted cost."""
    cfg = compressor.config(genes)
    t0 = time.time()
    try:
        stored = compressor.apply(cfg)
        loss = eval_loss(calib)
    finally:
        compressor.restore()
    if not math.isfinite(loss):
        loss = 1e6  # a broken model: dominated by everything, but the run goes on
    return {"loss": loss, "delta_loss": loss - baseline_loss, "ppl": math.exp(min(loss, 700.0)), **compressor.cost(cfg), **stored,
            "seconds": round(time.time() - t0, 2)}


def evaluate_all(individuals, gen, role):
    """Objectives and constraints of each individual (from the cache when seen before). Returns (F, G, n_reused)."""
    F, G, n_reused = [], [], 0
    for i, ind in enumerate(individuals):
        genes = compressor.decode(ind.X)
        key = key_of(genes)
        if key in cache:
            n_reused += 1
        else:
            cache[key] = r = evaluate(genes)
            cfg = compressor.config(genes)
            evals_log.write([{"gen": gen, "role": role, "key": key, "config": compressor.describe(cfg),
                              **dict(zip(GENE_NAMES, genes)), **r}])
            print(f"gen {gen} {role} [{i + 1}/{len(individuals)}] {compressor.describe(cfg)} | delta {r['delta_loss']:+.4f} "
                  f"turn {r['latency_ms']:.0f} ms size {r['size_mb']:.0f} MB memory {r['memory_mb']:.0f} MB ({r['seconds']:.1f}s)", flush=True)
        r = cache[key]
        F.append([r["delta_loss"]] + [r[c] for c in cost_objectives])
        G.append(([r["delta_loss"] - args.max_delta_loss] if args.max_delta_loss is not None else [])
                 + ([r["memory_mb"] - args.memory_budget_mb] if args.memory_budget_mb is not None else []))
    return np.array(F), np.array(G), n_reused


def rows(individuals, gen):
    out_rows = []
    for ind in individuals:
        genes = compressor.decode(ind.X)
        out_rows.append({"gen": gen, "rank": ind.get("rank"), "crowding": ind.get("crowding"), "key": key_of(genes),
                         "config": compressor.describe(compressor.config(genes)), **dict(zip(GENE_NAMES, genes)),
                         **cache[key_of(genes)]})
    return out_rows


# -----------------------------------------------------------------------------
# search

while algorithm.has_next():
    t_gen = time.time()
    torch.cuda.reset_peak_memory_stats()
    gen = 1 if algorithm.n_gen is None else algorithm.n_gen  # pymoo advances the counter inside ask()
    offspring = algorithm.ask()
    F, G, n_reused = evaluate_all(offspring, gen, "initial" if gen == 1 else "offspring")
    Evaluator().eval(StaticProblem(problem, F=F, **({"G": G} if problem.n_ieq_constr else {})), offspring)
    algorithm.tell(infills=offspring)

    opt = algorithm.opt[algorithm.opt.get("feas")]
    front_F = opt.get("F")
    if nadir is None and len(front_F):  # fixed once, from the first front with a feasible solution
        nadir = front_F.max(axis=0)
        if nadir[0] <= 0:  # the whole front is at least as good as the original: use the population's worst
            nadir[0] = max(algorithm.pop.get("F")[:, 0].max(), args.max_delta_loss or 1.0)
        atomic_dump({"nadir": nadir.tolist(), "objectives": ["delta_loss", *cost_objectives], "gen": gen}, norm_path, json)
    hv = float(HV(ref_point=np.ones(problem.n_obj))(front_F / nadir)) if len(front_F) else 0.0
    best = lambda col: min(cache[key_of(compressor.decode(x))][col] for x in opt.get("X")) if len(opt) else None
    summary = {"gen": gen, "hypervolume": hv, "front_size": len(opt), "n_feasible": int(algorithm.pop.get("feas").sum()),
               "n_offspring": len(offspring), "n_reused": n_reused, "best_delta_loss": best("delta_loss"),
               "min_latency_ms": best("latency_ms"), "min_size_mb": best("size_mb"), "min_memory_mb": best("memory_mb"),
               "seconds": round(time.time() - t_gen, 1), "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6)}
    pop_log.write(rows(algorithm.pop, gen))
    front_log.write(rows(opt, gen))
    gen_log.write([summary])
    atomic_dump(algorithm, out("checkpoint.pkl"), dill)  # last: a crash before this re-runs the generation from the cache
    print(f"=== generation {gen}/{args.n_gen} done in {summary['seconds']:.0f}s ({n_reused} reused), HV {hv:.5f}, "
          f"front {len(opt)} points, {summary['n_feasible']}/{len(algorithm.pop)} feasible" +
          (f", best delta loss {summary['best_delta_loss']:+.4f}, fastest turn {summary['min_latency_ms']:.0f} ms, "
           f"smallest {summary['min_size_mb']:.0f} MB stored" if len(opt) else ", no feasible solution yet"), flush=True)

# -----------------------------------------------------------------------------
# the final front: summary, then test-set scores (resumable: test_scores.json is written after each solution)

front = sorted(rows(algorithm.opt[algorithm.opt.get("feas")], algorithm.n_gen - 1), key=lambda r: r[cost_objectives[0]])
print(f"\nFinal front ({len(front)} points, sorted by {cost_objectives[0]}); uncompressed: turn {baseline_cost['latency_ms']:.0f} ms, "
      f"{baseline_cost['size_mb']:.0f} MB stored, {baseline_cost['memory_mb']:.0f} MB deployed", flush=True)
print(f"{'delta':>8} {'turn':>9} {'stored':>7} {'deployed':>8} {'agents':>6}  configuration", flush=True)
for r in front:
    print(f"{r['delta_loss']:+8.4f} {r['latency_ms']:7.0f}ms {r['size_mb']:5.0f}MB {r['memory_mb']:6.0f}MB {r['agents_fit']:6.0f}  {r['config']}", flush=True)

if args.wikitext_seqs == 0 and args.c4_test_seqs == 0:
    print("Test scoring skipped (no test windows)", flush=True)
    raise SystemExit
scored = [r for r in front if r["delta_loss"] <= args.test_max_delta_loss]
if args.test_max_points is not None and len(scored) > args.test_max_points:
    scored = [scored[j] for j in np.linspace(0, len(scored) - 1, args.test_max_points).round().astype(int)]
settings = {"seq_len": args.test_seq_len, "wikitext_seqs": args.wikitext_seqs, "c4_test_seqs": args.c4_test_seqs}
results = json.load(open(out("test_scores.json"))) if os.path.exists(out("test_scores.json")) else None
if results is None or results["settings"] != settings:
    results = {"settings": settings, "baseline": None, "pareto": []}  # the layout of compress_score's
done = {p["key"] for p in results["pareto"]}
test_sets = compress_eval.load_test_sets(tokenizer, args.test_seq_len, args.wikitext_seqs, args.c4_test_seqs, cache_key=args.model)
print(f"Scoring {len(scored)} front solutions on " + ", ".join(f"{n} ({t.size(0)} windows)" for n, t in test_sets.items()), flush=True)
if results["baseline"] is None:
    results["baseline"] = {k: baseline_cost[k] for k in ("latency_ms", "size_mb", "memory_mb")}
    for name, tokens in test_sets.items():
        results["baseline"][f"{name}_loss"] = loss = eval_loss(tokens)
        results["baseline"][f"{name}_ppl"] = math.exp(loss)
    atomic_dump(results, out("test_scores.json"), json)
for r in scored:
    if r["key"] in done:
        continue
    t0 = time.time()
    entry = {k: r[k] for k in ("key", "config", *GENE_NAMES, *STATS)}
    try:
        compressor.apply(compressor.config(compressor.decode([r[n] for n in GENE_NAMES])))
        for name, tokens in test_sets.items():
            entry[f"{name}_loss"] = loss = eval_loss(tokens)
            entry[f"{name}_ppl"] = math.exp(min(loss, 700.0))
    finally:
        compressor.restore()
    results["pareto"].append(entry)
    atomic_dump(results, out("test_scores.json"), json)
    print(f"  {r['latency_ms']:.0f} ms {r['size_mb']:.0f} MB: " + ", ".join(
        f"{n} ppl {entry[f'{n}_ppl']:.2f} (base {results['baseline'][f'{n}_ppl']:.2f})" for n in test_sets)
        + f" ({time.time() - t0:.0f}s)", flush=True)
print(f"Done. Test scores in {out('test_scores.json')}", flush=True)
