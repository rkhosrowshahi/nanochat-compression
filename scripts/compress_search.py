"""
Evolutionary multi-objective search (NSGA-II, pymoo) over GLOBAL compression
genes for a pretrained HuggingFace causal LM. See nanochat/compress.py for what
the genes (k, c, alpha, beta) do.

Objectives (both minimized):
    1) delta_loss: mean next-token cross-entropy of the compressed model on the
       current calibration batch (C4 train) minus that of the uncompressed model
       on the same batch (0 = as good as the original; lower is better). On one batch this ranks exactly like the raw loss, but
       it removes the batch difficulty, so values are comparable across batches.
    2) encoded model size in MB

Constraint (--max-delta-loss, optional): solutions with a larger delta_loss are infeasible.
NSGA-II then ranks every feasible solution ahead of every infeasible one, and infeasible
ones by how far they exceed the limit, so the population is spent on usable models instead
of broken ones. The front and the hypervolume only count feasible solutions.

Calibration batch (--calib-every):
    0   one fixed random batch for the whole run (the GPTQ/AWQ recipe): every solution
        is evaluated once, parents keep their scores, and the search is noise-free.
    N   a fresh batch every N generations (1 = every generation). When the batch
        changes, the parents are re-scored on it and re-ranked before the offspring are
        scored, so selection and survival always compare everyone on the same data.

Outputs in <out-dir>:
    evals.csv        every evaluation (generation, parent/offspring, genes, loss, size, ...)
    batch_baselines.csv  loss of the uncompressed model on each batch (gen = the first generation using it)
    population.csv   the surviving population after every generation (rank 0 = Pareto front)
    front.csv        the Pareto front after every generation
    generations.csv  one summary row per generation: hypervolume, timing, GPU memory, ...
    normalization.json  the nadir point used to normalize the hypervolume
    calib_pool.pt    all calibration batches of the run
    pareto.json      the final Pareto set (delta_loss <= --test-max-delta-loss) scored on
                     WikiText-2 test and C4 validation with 2048-token windows, with the
                     Huffman-coded storage size of each solution (reported, never optimized)
Every row carries the genes, both objectives, perplexity, size and pruning counts.

Hypervolume: each objective is divided by the nadir point (worst value on the
Pareto front) of the initial population, then measured against reference (1, 1).

Designed to survive interruptions on a long server run: the full NSGA-II state
(incl. RNG) is saved to <out-dir>/checkpoint.pkl after every generation; rerunning
the same command resumes from it, and evaluations of a half-finished generation
are read back from evals.csv, not recomputed.

Examples:

    # quick local smoke test (small model, tiny population)
    python -m scripts.compress_search --pop-size 6 --n-gen 2 --out-dir compress_runs/smoke

    # full run on the server
    python -m scripts.compress_search --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-batch-size 8 --out-dir compress_runs/qwen7b
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time

import dill
import numpy as np
import torch
from datasets import load_dataset
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems.static import StaticProblem
from pymoo.termination import get_termination
from nanochat import compress_eval, compress_models
from nanochat.compress import Candidate, GlobalCompressor

parser = argparse.ArgumentParser(description="NSGA-II search over global quantization + pruning genes")
# model
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HuggingFace model id or local path, or a nanochat checkpoint nanochat:<base|sft|rl>:<tag>[:<step>], "
                         "e.g. nanochat:base:d34 (see nanochat/compress_models.py)")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
parser.add_argument("--include-lm-head", action="store_true", help="also compress the output projection")
parser.add_argument("--include-embeddings", action="store_true",
                    help="also compress the embedding tables (nn.Embedding, one row per token); for nanochat the token "
                         "embedding and the value embeddings (37%% of d34's parameters)")
parser.add_argument("--offload-originals", type=str, default="auto", choices=["auto", "yes", "no"], help="keep original weights in CPU memory")
# data, following GPTQ/AWQ: the search only ever sees C4 train (a new calibration batch
# every generation); the final Pareto set is scored on WikiText-2 test and C4 validation
# (C4 has no test split, its validation split is the reported one)
parser.add_argument("--calib-dataset", type=str, default="allenai/c4")
parser.add_argument("--calib-config", type=str, default="en")
parser.add_argument("--calib-split", type=str, default="train", help="split for the search objective")
parser.add_argument("--calib-seqs", type=int, default=8, help="windows per calibration batch")
parser.add_argument("--calib-every", type=int, default=1,
                    help="draw a new calibration batch every N generations; 0 = one fixed batch for the whole run "
                         "(parents are only re-evaluated when the batch changes)")
parser.add_argument("--wikitext-seqs", type=int, default=-1, help="WikiText-2 test windows for the final set: -1 = all, 0 = skip")
parser.add_argument("--c4-test-seqs", type=int, default=256, help="C4 validation windows for the final set (GPTQ uses 256), 0 = skip")
parser.add_argument("--test-seq-len", type=int, default=2048, help="window length for the test sets (2048 as in GPTQ/AWQ)")
parser.add_argument("--test-max-delta-loss", type=float, default=1.0,
                    help="only score final solutions whose delta_loss on the last batch is at most this (broken models are skipped)")
parser.add_argument("--text-field", type=str, default="text")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--data-seed", type=int, default=0, help="which random documents/windows are sampled")
parser.add_argument("--eval-batch-size", type=int, default=1)
# genes (search bounds)
parser.add_argument("--k-min", type=int, default=3)
parser.add_argument("--k-max", type=int, default=256)
parser.add_argument("--k-space", type=str, default="log2", choices=["log2", "linear"],
                    help="gene for k: log2 searches b = log2(k) with k = round(2^b), so crossover/mutation steps "
                         "mean about the same number of bits at every k and each bit width gets an equal share "
                         "of the random initial population; linear searches k itself")
parser.add_argument("--c-min", type=float, default=0.7)
parser.add_argument("--c-max", type=float, default=1.0)
parser.add_argument("--alpha-min", type=float, default=-1.0, help="alpha in [alpha-min, 0]")
parser.add_argument("--beta-max", type=float, default=1.0, help="beta in [0, beta-max]")
parser.add_argument("--no-pruning", action="store_true",
                    help="quantization only: drop the alpha/beta genes and never prune (logged as alpha = beta = 0)")
parser.add_argument("--max-delta-loss", type=float, default=None,
                    help="constraint: delta_loss above this is infeasible (default: unconstrained)")
parser.add_argument("--size-mode", type=str, default="fixed", choices=["fixed", "entropy"])
parser.add_argument("--formats", type=str, default="dense,bitmap", help="storage formats for --size-mode fixed, from dense,bitmap,csr; each matrix uses the smallest")
parser.add_argument("--reconstruction", type=str, default="grid", choices=["grid", "centroid"], help="value of each bin: its center, or the mean of its weights")
# search
parser.add_argument("--pop-size", type=int, default=100)
parser.add_argument("--n-gen", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", type=str, default="compress_runs/default")
args = parser.parse_args()

GENES = ["b", "k_init", "c", "alpha", "beta"]  # b: the k gene in bits, k_init = round(2^b): the grid size
STATS = ["loss", "delta_loss", "ppl", "size_mb", "size_ratio", "target_bits_per_weight",
         "n_pruned", "n_rounded_zero", "n_nonzero", "pruned_pct", "sparsity_pct",
         "n_dense_matrices", "n_bitmap_matrices", "n_csr_matrices",
         "k_used_mean", "k_used_min", "k_used_max", "k_used_row_mean"]
EVAL_FIELDS = ["gen", "role", *GENES, *STATS, "seconds"]
POP_FIELDS = ["gen", "rank", "crowding", *GENES, *STATS, "f1_norm", "f2_norm"]
GEN_FIELDS = ["gen", "batch_baseline_loss", "hypervolume", "front_size", "n_feasible", "n_parents", "n_offspring", "n_reused",
              "best_delta_loss", "min_size_mb", "seconds", "peak_gpu_mb"]


def sample_windows(tokenizer, dataset, config, split, n_seqs, seed):
    """n_seqs windows of seq_len tokens, each at a random offset of a different random
    document that is longer than seq_len (the GPTQ calibration recipe)."""
    ds = load_dataset(dataset, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    rng = np.random.default_rng(seed)
    windows = []
    for example in ds:
        ids = tokenizer(example[args.text_field], add_special_tokens=False)["input_ids"]
        if len(ids) <= args.seq_len:
            continue
        start = int(rng.integers(0, len(ids) - args.seq_len + 1))
        windows.append(ids[start:start + args.seq_len])
        if len(windows) == n_seqs:
            return torch.tensor(windows, dtype=torch.long)
    raise ValueError(f"{dataset}/{split} has fewer than {n_seqs} documents longer than {args.seq_len} tokens")


def eval_loss(model, tokens):
    return compress_eval.eval_loss(model, tokens, args.eval_batch_size)


class SymmetricPruneSampling(FloatRandomSampling):
    """
    Uniform initial population, except the pruning interval starts symmetric
    (alpha = -beta): asymmetric intervals shift every row's output and wreck the
    model, so a random (alpha, beta) start wastes the first generations. Crossover
    and mutation still move alpha and beta independently afterwards.
    """

    def _do(self, problem, n_samples, *op_args, random_state=None, **kwargs):
        X = super()._do(problem, n_samples, *op_args, random_state=random_state, **kwargs)
        t = random_state.random(n_samples) * min(-args.alpha_min, args.beta_max)
        if not args.no_pruning:
            X[:, 2], X[:, 3] = -t, t
        return X


def decode(x):
    k_gene = 2.0 ** float(x[0]) if args.k_space == "log2" else float(x[0])
    k = int(np.clip(round(k_gene), args.k_min, args.k_max))
    if args.no_pruning:
        return Candidate(k=k, c=float(x[1]), alpha=0.0, beta=0.0)
    return Candidate(k=k, c=float(x[1]), alpha=float(x[2]), beta=float(x[3]))


def gene_fields(cand, x):
    """The genes as logged: b in bits (log2 of the raw k gene), the decoded grid size k_init, c, alpha, beta."""
    b = float(x[0]) if args.k_space == "log2" else float(np.log2(x[0]))
    return {"b": b, "k_init": cand.k, "c": cand.c, "alpha": cand.alpha, "beta": cand.beta}


def k_to_gene(k):
    return float(np.log2(k)) if args.k_space == "log2" else float(k)


def cache_key(cand):
    return (cand.k, round(cand.c, 9), round(cand.alpha, 9), round(cand.beta, 9))


class DecodedDuplicateElimination(ElementwiseDuplicateElimination):
    """Two individuals are duplicates when they decode to the same candidate (k is rounded)."""

    def is_equal(self, a, b):
        return cache_key(decode(a.X)) == cache_key(decode(b.X))


RENAMED_COLUMNS = {"loss_increase": "delta_loss", "best_loss_increase": "best_delta_loss"}


def upgrade_row(row):
    """Old column names -> current ones, so runs started by an older version can be resumed."""
    return {RENAMED_COLUMNS.get(k, k): v for k, v in row.items()}


class CsvLog:
    """Append-only CSV. On resume, rows from generations after `keep_upto` are dropped,
    and a file with older column names is rewritten with the current ones."""

    def __init__(self, path, fields, keep_upto=None):
        self.path, self.fields = path, fields
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                header, rows = reader.fieldnames, [upgrade_row(r) for r in reader]
            if keep_upto is not None or header != fields:
                rows = [r for r in rows if keep_upto is None or int(r["gen"]) <= keep_upto]
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


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [upgrade_row(r) for r in csv.DictReader(f)]


def evaluate(cand, tokens, huffman=False):
    """Apply cand, measure loss, restore the original weights. Returns (loss, size stats)."""
    try:
        stats = compressor.apply(cand, huffman=huffman)
        loss = eval_loss(model, tokens)
    finally:
        compressor.restore()
    if not math.isfinite(loss):
        loss = 1e6  # a broken model: dominated by everything, but keeps the run going
    stats["ppl"] = math.exp(min(loss, 700.0))
    return loss, stats


def atomic_dump(obj, path, dumper):
    tmp = path + ".tmp"
    with open(tmp, "wb" if dumper is dill else "w") as f:
        dumper.dump(obj, f) if dumper is dill else json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# setup

os.makedirs(args.out_dir, exist_ok=True)
out = lambda name: os.path.join(args.out_dir, name)
ckpt_path, norm_path, pool_path = out("checkpoint.pkl"), out("normalization.json"), out("calib_pool.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    git_commit = None

print(f"Loading {args.model} on {device}", flush=True)
model, tokenizer, exclude = compress_models.load(args.model, args.dtype, device)
compressor = GlobalCompressor(model, args.include_lm_head, args.size_mode, args.offload_originals, args.reconstruction,
                              formats=args.formats.split(","), prune=not args.no_pruning, exclude=exclude,
                              include_embeddings=args.include_embeddings)
print(f"Compressing {len(compressor.targets)} matrices, {compressor.n_target_params / 1e6:.1f}M weights "
      f"({compressor.n_target_params / (compressor.n_target_params + compressor.rest_params):.1%} of params), "
      f"originals {'offloaded to CPU' if compressor.offloaded else 'kept on GPU'}", flush=True)

# the calibration batches, all drawn up front from a single shuffled pass over the documents
# (no document is used twice) and saved, so a resumed run sees exactly the same batches.
# A longer run with the same seed draws the same first batches, then more.
assert args.calib_every >= 0, "--calib-every must be 0 (fixed batch) or a positive number of generations"
batch_index = lambda gen: 0 if args.calib_every == 0 else (gen - 1) // args.calib_every  # which batch gen uses
n_batches = batch_index(args.n_gen) + 1
pool_meta = {"source": [args.calib_dataset, args.calib_config, args.calib_split],
             "seed": args.data_seed, "seq_len": args.seq_len, "calib_seqs": args.calib_seqs}
pool = None
if os.path.exists(pool_path):
    saved = torch.load(pool_path)
    if saved["meta"] == pool_meta and saved["tokens"].size(0) >= n_batches * args.calib_seqs:
        pool = saved["tokens"]
if pool is None:
    t0 = time.time()
    print(f"Sampling {n_batches} calibration batch(es) of {args.calib_seqs} x {args.seq_len} tokens from "
          f"{args.calib_dataset} {args.calib_split}...", flush=True)
    pool = sample_windows(tokenizer, args.calib_dataset, args.calib_config, args.calib_split,
                          n_batches * args.calib_seqs, args.data_seed)
    torch.save({"meta": pool_meta, "tokens": pool}, pool_path)
    print(f"...done in {time.time() - t0:.0f}s", flush=True)
calib_batch = lambda b: pool[b * args.calib_seqs:(b + 1) * args.calib_seqs]
print("Calibration: " + ("one fixed batch for the whole run" if args.calib_every == 0 else
                         f"a new batch every {args.calib_every} generation(s)"), flush=True)

t0 = time.time()
baseline_loss = eval_loss(model, calib_batch(0))
print(f"Baseline: calib loss {baseline_loss:.4f} on the first batch, size {compressor.baseline_bits / 8 / 1e6:.1f} MB, "
      f"one forward pass over a batch takes {time.time() - t0:.1f}s", flush=True)
if os.path.exists(ckpt_path) and os.path.exists(out("config.json")):
    # the checkpoint stores raw genes: resuming with another encoding or other bounds would misread them
    old = json.load(open(out("config.json")))["args"]
    for key, default in (("k_space", "linear"), ("k_min", 3), ("k_max", 256), ("c_min", 0.2), ("c_max", 1.0),
                         ("alpha_min", -1.0), ("beta_max", 1.0), ("no_pruning", False), ("calib_every", 1), ("max_delta_loss", None),
                         ("model", "Qwen/Qwen2.5-0.5B-Instruct"), ("include_lm_head", False), ("include_embeddings", False),
                         ("calib_seqs", 8), ("seq_len", 1024)):
        assert old.get(key, default) == getattr(args, key), \
            f"this run was started with --{key.replace('_', '-')} {old.get(key, default)}, not {getattr(args, key)}"
atomic_dump({"args": vars(args), "git_commit": git_commit, "baseline_size_mb": compressor.baseline_bits / 8 / 1e6,
             "n_target_params": compressor.n_target_params, "n_other_params": compressor.rest_params},
            out("config.json"), json)

# evaluations are only valid on the batch they were made on: key = (batch index, candidate), so
# with a fixed batch every candidate is evaluated once for the whole run.
# Reloaded on resume, so a half-finished generation does not repeat work.
cache = {}
row_key = lambda r: cache_key(Candidate(int(r["k_init"]), float(r["c"]), float(r["alpha"]), float(r["beta"])))
for r in read_rows(out("evals.csv")):
    assert "k_used_mean" in r, "evals.csv is from an older version of this script, use a new --out-dir"
    cache[(batch_index(int(r["gen"])), row_key(r))] = {f: float(r[f]) for f in STATS}
# loss of the uncompressed model per batch index (logged with the first generation using the batch)
batch_baselines = {batch_index(int(r["gen"])): float(r["baseline_loss"]) for r in read_rows(out("batch_baselines.csv"))}
if cache:
    print(f"Loaded {len(cache)} previous evaluations", flush=True)

# the problem only carries the bounds; evaluation happens in the ask/tell loop below,
# which keeps the model out of the pickled algorithm state
problem = Problem(
    n_var=2 if args.no_pruning else 4, n_obj=2, n_ieq_constr=0 if args.max_delta_loss is None else 1,
    # the k gene covers [k_min - 0.49, k_max + 0.49] so the end values get their full rounding interval
    xl=np.array([k_to_gene(args.k_min - 0.49), args.c_min, args.alpha_min, 0.0][:2 if args.no_pruning else 4]),
    xu=np.array([k_to_gene(args.k_max + 0.49), args.c_max, 0.0, args.beta_max][:2 if args.no_pruning else 4]),
)
if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        algorithm = dill.load(f)
    algorithm.termination = get_termination("n_gen", args.n_gen)
    done_gen = algorithm.n_gen - 1
    print(f"Resumed NSGA-II from {ckpt_path} after generation {done_gen}", flush=True)
else:
    algorithm = NSGA2(pop_size=args.pop_size, sampling=SymmetricPruneSampling(),
                      eliminate_duplicates=DecodedDuplicateElimination())
    algorithm.setup(problem, termination=get_termination("n_gen", args.n_gen), seed=args.seed)
    done_gen = 0

evals_log = CsvLog(out("evals.csv"), EVAL_FIELDS)
baseline_log = CsvLog(out("batch_baselines.csv"), ["gen", "baseline_loss"])
# drop rows of a generation that was logged but not checkpointed before an interruption
pop_log = CsvLog(out("population.csv"), POP_FIELDS, keep_upto=done_gen)
front_log = CsvLog(out("front.csv"), POP_FIELDS, keep_upto=done_gen)
gen_log = CsvLog(out("generations.csv"), GEN_FIELDS, keep_upto=done_gen)
nadir = np.array(json.load(open(norm_path))["nadir"]) if os.path.exists(norm_path) else None


def evaluate_on_batch(individuals, gen, role):
    """Calibration loss and size of each individual on the batch of this generation. Returns (F, n_reused)."""
    b = batch_index(gen)
    tokens = calib_batch(b)
    F_out, n_reused = np.zeros((len(individuals), 2)), 0
    for i, ind in enumerate(individuals):
        cand = decode(ind.X)
        key = (b, cache_key(cand))
        if key in cache:
            n_reused += 1
        else:
            t0 = time.time()
            loss, stats = evaluate(cand, tokens)
            cache[key] = {"loss": loss, **{f: stats[f] for f in STATS if f not in ("loss", "delta_loss")},
                          "delta_loss": loss - batch_baselines[b]}
            evals_log.write([{"gen": gen, "role": role, **gene_fields(cand, ind.X), **cache[key],
                              "seconds": round(time.time() - t0, 3)}])
            print(f"gen {gen} {role} [{i + 1}/{len(individuals)}] k={cand.k:3d} (used {stats['k_used_mean']:5.1f}) "
                  f"c={cand.c:.3f} alpha={cand.alpha:+.3f} "
                  f"beta={cand.beta:+.3f} | loss {loss:.4f} (delta {cache[key]['delta_loss']:+.4f}) "
                  f"size {stats['size_mb']:.1f} MB sparsity {stats['sparsity_pct']:.1f}%", flush=True)
        F_out[i] = cache[key]["delta_loss"], cache[key]["size_mb"]
    return F_out, n_reused


def outputs(F):
    """Objectives, plus the constraint delta_loss - max_delta_loss <= 0 when there is one."""
    return {"F": F} if args.max_delta_loss is None else {"F": F, "G": F[:, :1] - args.max_delta_loss}


def population_rows(individuals, gen):
    rows = []
    for ind in individuals:
        cand = decode(ind.X)
        r = cache[(batch_index(gen), cache_key(cand))]
        rows.append({"gen": gen, "rank": ind.get("rank"), "crowding": ind.get("crowding"), **gene_fields(cand, ind.X), **r,
                     "f1_norm": r["delta_loss"] / nadir[0], "f2_norm": r["size_mb"] / nadir[1]})
    return rows


# -----------------------------------------------------------------------------
# search: parents and offspring are always compared on the same batch

while algorithm.has_next():
    t_gen = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = 1 if algorithm.n_gen is None else algorithm.n_gen  # pymoo advances the counter inside ask()
    b = batch_index(gen)
    if b not in batch_baselines:
        batch_baselines[b] = eval_loss(model, calib_batch(b))
        baseline_log.write([{"gen": gen, "baseline_loss": batch_baselines[b]}])

    n_parents, n_reused = 0, 0
    if gen > 1 and b != batch_index(gen - 1):
        # the batch changed: re-score the parents on it and re-rank them, so mating selection
        # and survival both compare everyone on the same data (with an unchanged batch the
        # parents keep the scores they already have)
        parents = algorithm.pop
        F_parents, n_reused = evaluate_on_batch(parents, gen, "parent")
        Evaluator().eval(StaticProblem(problem, **outputs(F_parents)), parents, skip_already_evaluated=False, count_evals=False)
        algorithm.pop = algorithm.survival.do(problem, parents, n_survive=len(parents),
                                              algorithm=algorithm, random_state=algorithm.random_state)
        n_parents = len(parents)

    offspring = algorithm.ask()
    F_off, n_reused_off = evaluate_on_batch(offspring, gen, "initial" if gen == 1 else "offspring")
    Evaluator().eval(StaticProblem(problem, **outputs(F_off)), offspring)
    algorithm.tell(infills=offspring)
    n_reused += n_reused_off

    # with a constraint and no feasible solution yet, pymoo's opt holds the least infeasible ones
    front_F = algorithm.opt.get("F")[algorithm.opt.get("feas")]
    if nadir is None and len(front_F):  # fixed once, from the (feasible) Pareto front of the initial population
        nadir = front_F.max(axis=0)
        if nadir[0] <= 0:  # the whole initial front is at least as good as the original: use the population's worst
            nadir[0] = algorithm.pop.get("F")[:, 0].max()
        if nadir[0] <= 0:  # every candidate beats the original (only seen with random test models)
            nadir[0] = args.max_delta_loss or 1.0
        assert (nadir > 0).all(), f"nadir {nadir} must be positive to normalize the hypervolume"
        atomic_dump({"nadir": nadir.tolist(), "objectives": ["delta_loss", "size_mb"], "gen": gen}, norm_path, json)
    hv = float(HV(ref_point=np.ones(2))(front_F / nadir)) if len(front_F) else 0.0
    summary = {"gen": gen, "batch_baseline_loss": batch_baselines[b], "hypervolume": hv, "front_size": len(front_F),
               "n_feasible": int(algorithm.pop.get("feas").sum()),
               "n_parents": n_parents, "n_offspring": len(offspring), "n_reused": n_reused,
               "best_delta_loss": front_F[:, 0].min() if len(front_F) else None,
               "min_size_mb": front_F[:, 1].min() if len(front_F) else None, "seconds": round(time.time() - t_gen, 1),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None}
    pop_log.write(population_rows(algorithm.pop, gen))
    front_log.write(population_rows(algorithm.opt[algorithm.opt.get("feas")], gen))
    gen_log.write([summary])
    atomic_dump(algorithm, ckpt_path, dill)  # last: a crash before this re-runs the generation from the cache
    print(f"=== generation {gen}/{args.n_gen} done in {summary['seconds']:.0f}s ({n_reused} reused), "
          f"batch baseline {batch_baselines[b]:.4f}, HV {hv:.5f}, front {len(front_F)} points, "
          f"{summary['n_feasible']}/{len(algorithm.pop)} feasible, " +
          (f"best delta loss {summary['best_delta_loss']:+.4f}, smallest {summary['min_size_mb']:.1f} MB" if len(front_F)
           else "no feasible solution yet"), flush=True)

# -----------------------------------------------------------------------------
# score the final Pareto set on the test sets (never seen by the search)

opt = algorithm.opt
last_gen = algorithm.n_gen - 1
order = [i for i in np.argsort(opt.get("F")[:, 1]) if opt.get("F")[i, 0] <= args.test_max_delta_loss]
print(f"Scoring {len(order)} of {len(opt)} final solutions (delta_loss <= {args.test_max_delta_loss}) "
      f"with {args.test_seq_len}-token windows", flush=True)
test_sets = compress_eval.load_test_sets(tokenizer, args.test_seq_len, args.wikitext_seqs, args.c4_test_seqs)
for name, tokens in test_sets.items():
    print(f"Test set {name}: {tokens.size(0)} windows, {tokens.numel()} tokens", flush=True)
results = {"baseline": {"last_batch_loss": batch_baselines[batch_index(last_gen)], "size_mb": compressor.baseline_bits / 8 / 1e6},
           "test_seq_len": args.test_seq_len, "max_delta_loss": args.test_max_delta_loss, "pareto": []}
for name, tokens in test_sets.items():
    results["baseline"][f"{name}_loss"] = loss = eval_loss(model, tokens)
    results["baseline"][f"{name}_ppl"] = math.exp(loss)
for idx in order:
    cand = decode(opt[idx].get("X"))
    r = cache[(batch_index(last_gen), cache_key(cand))]
    entry = {**gene_fields(cand, opt[idx].get("X")), **{f"last_batch_{f}" if f in ("loss", "ppl", "delta_loss") else f: v for f, v in r.items()}}
    stats = compressor.apply(cand, huffman=True)  # storage only, never used by the search
    compressor.restore()
    entry["huffman_size_mb"], entry["huffman_bits_per_weight"] = stats["huffman_size_mb"], stats["huffman_bits_per_weight"]
    for name, tokens in test_sets.items():
        entry[f"{name}_loss"], stats = evaluate(cand, tokens)
        entry[f"{name}_ppl"] = stats["ppl"]
    results["pareto"].append(entry)
    print(json.dumps(entry), flush=True)
atomic_dump(results, out("pareto.json"), json)
print(f"Done. Pareto set ({len(order)} points) written to {out('pareto.json')}", flush=True)
