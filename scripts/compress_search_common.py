"""
What the turn-time searches share (scripts/compress_search_kv.py, scripts/compress_search_mix_simple.py): their
arguments, the setup of a run (log, model, calibration batch, statistics, latency tables), NSGA-II over integer genes with
evaluations in an ask/tell loop, resuming from checkpoint.pkl and evals.csv, and the test scores of the final front.
The outputs have the layout of compress_search_mix.py, so compress_plot and compress_compare read them.

A search script builds a compressor (apply, restore, cost, decode, config, describe, uniform_genes, formats, genes) and
calls Search.prepare_tables, Search.start, Search.run and Search.score. Its compressor must decode as
HWCompressor.decode does: the checkpoint cannot hold a model, so the operators below only get the gene sizes.
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

from nanochat import compress_eval, compress_hw, compress_models

COST_STAT = {"time": "latency_ms", "size": "size_mb", "memory": "memory_mb"}
GEN_FIELDS = ["gen", "hypervolume", "front_size", "n_feasible", "n_offspring", "n_reused", "best_delta_loss",
              "min_latency_ms", "min_size_mb", "min_memory_mb", "seconds", "peak_gpu_mb"]
# arguments that change what a gene vector means or how it is scored: a resumed run must keep them
FIXED_ARGS = ("model", "objectives", "max_delta_loss", "memory_budget_mb", "serve_batch", "prompt_batch", "context", "new_tokens",
              "gen_tokens", "formats", "bands", "mlp_keep", "max_skip", "no_embeddings", "clip_ratios", "calib_dataset",
              "calib_config", "calib_split", "calib_seqs", "seq_len", "data_seed", "importance_seqs", "pop_size")
STRUCTURAL = ("keep_", "skip_", "window_")  # random values here mostly break the model


def add_arguments(parser, objectives, out_dir):
    # model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model id or local path")
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
    parser.add_argument("--objectives", nargs="+", default=objectives, choices=list(COST_STAT), help="besides delta_loss")
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
    parser.add_argument("--formats", type=str, default=",".join(compress_hw.FORMATS), help=f"kernels, from {','.join(compress_hw.FORMATS)}")
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
    parser.add_argument("--out-dir", type=str, default=out_dir)


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


def decode_genes(x, genes):
    """As HWCompressor.decode, from the gene sizes alone."""
    return tuple(int(min(max(round(float(v)), 0), n - 1)) for v, (_, n) in zip(x, genes))


class SeededSampling(Sampling):
    """Random genes, except that each structural gene starts at 0 (nothing pruned, the whole context) half of the time, and
    the first individuals are the given seeds: random pruning mostly breaks the model, and the uniform ones span the search
    from safe to cheap. Plain data only: it goes into the checkpoint."""

    def __init__(self, names, seeds):
        super().__init__()
        self.names, self.seeds = list(names), [list(s) for s in seeds]

    def _do(self, problem, n_samples, *op_args, random_state=None, **kwargs):
        X = random_state.integers(0, problem.xu.astype(int) + 1, size=(n_samples, problem.n_var))
        structural = np.array([n.startswith(STRUCTURAL) for n in self.names])
        X[:, structural] *= random_state.random((n_samples, int(structural.sum()))) < 0.5
        for i, seed in enumerate(self.seeds[:n_samples]):
            X[i] = seed
        return X.astype(float)


class DecodedDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, genes):
        super().__init__()
        self.gene_sizes = list(genes)

    def is_equal(self, a, b):
        return decode_genes(a.X, self.gene_sizes) == decode_genes(b.X, self.gene_sizes)


class Search:
    """One run of a turn-time search in args.out_dir. Rerunning the same command continues it."""

    def __init__(self, args, script):
        self.args = args
        os.makedirs(args.out_dir, exist_ok=True)
        run_log = open(self.out("run.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, run_log), Tee(sys.stderr, run_log)
        print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.{script} {' '.join(sys.argv[1:])}", flush=True)
        assert torch.cuda.is_available(), "the latency tables and the int4 kernel need a CUDA GPU"
        self.device = torch.device("cuda")
        try:
            self.git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            self.git_commit = None
        if os.path.exists(self.out("checkpoint.pkl")) and os.path.exists(self.out("config.json")):
            old = json.load(open(self.out("config.json")))["args"]
            for key in FIXED_ARGS + getattr(args, "fixed_extra", ()):
                assert old.get(key) == getattr(args, key), \
                    f"this run was started with --{key.replace('_', '-')} {old.get(key)}, not {getattr(args, key)}: use a new --out-dir"

        print(f"Loading {args.model} on {self.device}", flush=True)
        self.model, self.tokenizer, _ = compress_models.load(args.model, args.dtype, self.device)
        self.layout = L = compress_hw.Layout(self.model)
        print(f"{L.n_layer} layers, MLP width {L.intermediate}, {L.n_head} heads ({L.n_kv_head} KV), "
              f"lm_head {'tied to the embedding' if L.tied else 'untied'}, embeddings {L.embeds or 'none separate'}", flush=True)
        self.calib = self._calibration_batch()
        t0 = time.time()
        self.baseline_loss = self.eval_loss(self.calib)
        print(f"Baseline: calib loss {self.baseline_loss:.4f}, one forward pass over the batch takes {time.time() - t0:.1f}s", flush=True)
        self.act_sq, self.removal = self._statistics()
        self.formats = [f for f in args.formats.split(",") if f]
        self.mlp_keep = sorted({float(k) for k in args.mlp_keep.split(",")} | {1.0}, reverse=True)

    def out(self, name):
        return os.path.join(self.args.out_dir, name)

    def eval_loss(self, tokens):
        return compress_eval.eval_loss(self.model, tokens, self.args.eval_batch_size)

    # --- setup

    def _sample_windows(self):
        """calib_seqs windows of seq_len tokens, each at a random offset of a different random document longer than
        seq_len (the same sampling as scripts.compress_search)."""
        a = self.args
        ds = load_dataset(a.calib_dataset, a.calib_config, split=a.calib_split, streaming=True).shuffle(seed=a.data_seed, buffer_size=10_000)
        rng = np.random.default_rng(a.data_seed)
        windows = []
        for example in ds:
            ids = self.tokenizer(example[a.text_field], add_special_tokens=False)["input_ids"]
            if len(ids) <= a.seq_len:
                continue
            start = int(rng.integers(0, len(ids) - a.seq_len + 1))
            windows.append(ids[start:start + a.seq_len])
            if len(windows) == a.calib_seqs:
                return torch.tensor(windows, dtype=torch.long)
        raise ValueError(f"{a.calib_dataset}/{a.calib_split} has fewer than {a.calib_seqs} documents longer than {a.seq_len} tokens")

    def _calibration_batch(self):
        """This run's copy, --calib-from, the shared cache, or a fresh sample (then cached)."""
        a = self.args
        meta = {"source": [a.calib_dataset, a.calib_config, a.calib_split], "seed": a.data_seed, "seq_len": a.seq_len, "calib_seqs": a.calib_seqs}
        cache = compress_eval.cache_path("calib", {**meta, "model": a.model})

        def usable(path):
            if not path or not os.path.exists(path):
                return None
            saved = torch.load(path)
            if saved["meta"] != meta or saved["tokens"].size(0) < a.calib_seqs or saved.get("model", a.model) != a.model:
                return None
            return saved["tokens"]

        pool = None
        for path in (self.out("calib_pool.pt"), os.path.join(a.calib_from, "calib_pool.pt") if a.calib_from else None, cache):
            pool = usable(path)
            if pool is not None:
                print(f"Calibration batch from {path}", flush=True)
                break
        if pool is None:
            print(f"Sampling {a.calib_seqs} x {a.seq_len} calibration tokens from {a.calib_dataset} {a.calib_split}...", flush=True)
            pool = self._sample_windows()
        saved = {"meta": meta, "tokens": pool, "model": a.model}
        compress_eval.save_atomic(saved, self.out("calib_pool.pt"))
        if usable(cache) is None:
            compress_eval.save_atomic(saved, cache)
        self.calib_meta = meta
        return pool[:a.calib_seqs]

    def _statistics(self):
        """One-time statistics of the uncompressed model, each saved with the settings it was made with."""
        a = self.args
        settings = {"model": a.model, "calib": self.calib_meta}
        saved = torch.load(self.out("act_stats.pt")) if os.path.exists(self.out("act_stats.pt")) else None
        if saved is None or saved["settings"] != settings:
            t0 = time.time()
            act_sq = compress_hw.activation_stats(self.model, self.layout, self.calib, a.eval_batch_size)
            torch.save({"settings": settings, "stats": act_sq}, self.out("act_stats.pt"))
            print(f"Activation statistics of {len(act_sq)} linear inputs in {time.time() - t0:.0f}s", flush=True)
        else:
            act_sq = saved["stats"]
        if a.max_skip <= 0:  # nothing is removed: the order does not matter
            return act_sq, {kind: [(i, 0.0) for i in range(self.layout.n_layer)] for kind in ("attn", "mlp")}
        settings = {**settings, "importance_seqs": a.importance_seqs}
        removal = json.load(open(self.out("removal.json"))) if os.path.exists(self.out("removal.json")) else None
        if removal is None or removal["settings"] != settings:
            t0 = time.time()
            print(f"Removal ranking: each of the {2 * self.layout.n_layer} sublayers removed alone, on {a.importance_seqs} windows", flush=True)
            base, order = compress_hw.removal_ranking(self.model, self.layout, self.calib[:a.importance_seqs], a.eval_batch_size,
                                                      log=lambda m: print(m, flush=True))
            removal = {"settings": settings, "baseline_loss": base, **order}
            atomic_dump(removal, self.out("removal.json"), json)
            print(f"...done in {time.time() - t0:.0f}s", flush=True)
        return act_sq, {k: removal[k] for k in ("attn", "mlp")}

    def table(self, file, kind, build, batch=None, extra=None):
        """A kernel-time table: this run's copy, the per-GPU cache, or measured now by build() (then cached)."""
        a = self.args
        settings = {"version": compress_hw.TABLE_VERSION, "kind": kind, "gpu": torch.cuda.get_device_name(self.device),
                    "torch": str(torch.__version__), "model": a.model, "batch": batch, "context": a.context,
                    "formats": self.formats, "mlp_keep": self.mlp_keep, **(extra or {})}
        cache_file = compress_eval.cache_path("latency", settings)
        table = json.load(open(self.out(file))) if os.path.exists(self.out(file)) else None
        if table is None or table.get("settings") != settings:
            if os.path.exists(cache_file):
                table = torch.load(cache_file)
                print(f"{kind} table from {cache_file}", flush=True)
            else:
                t0 = time.time()
                print(f"Measuring the {kind} table on {settings['gpu']}: batch {batch}, {a.context} tokens of context", flush=True)
                table = {"settings": settings, **build()}
                compress_eval.save_atomic(table, cache_file)
                print(f"...done in {time.time() - t0:.0f}s", flush=True)
            atomic_dump(table, self.out(file), json)
        return table

    def turn_tables(self, build_prompt):
        """The decode and prompt tables of the turn (the caches of compress_search_mix's, on the same settings)."""
        a, log = self.args, lambda m: print(m, flush=True)
        decode = self.table("latency_table.json", "decode", lambda: compress_hw.build_latency_table(
            self.model, self.layout, self.formats, self.mlp_keep, a.serve_batch, a.context, log=log), batch=a.serve_batch)
        prompt = self.table("prompt_table.json", "prompt", lambda: build_prompt(
            self.model, self.layout, self.formats, self.mlp_keep, a.prompt_batch, a.context, log=log), batch=a.prompt_batch)
        for t in (decode, prompt):
            unavailable = sorted(set(self.formats) - set(t["formats"]))
            assert not unavailable, f"no kernel on this GPU runs {unavailable}: drop them from --formats"
        return decode, prompt

    def compressor_options(self):
        """The options every turn-time compressor takes."""
        a = self.args
        return dict(new_tokens=a.new_tokens, gen_tokens=a.gen_tokens, vram_gb=a.vram_gb, reserve=a.reserve, formats=self.formats,
                    bands=a.bands, mlp_keep=self.mlp_keep, max_skip=a.max_skip, compress_embeddings=not a.no_embeddings,
                    clip_ratios=[float(r) for r in a.clip_ratios.split(",")], offload=a.offload_originals)

    def start(self, compressor, decode_table, stats, seeds):
        """The baseline, config.json, and the algorithm (new or from the checkpoint). stats: the columns logged per solution."""
        a, L = self.args, self.layout
        self.compressor, self.stats = compressor, stats
        self.gene_names = [name for name, _ in compressor.genes]
        self.eval_fields = ["gen", "role", "key", "config", *self.gene_names, *stats, "seconds"]
        self.pop_fields = ["gen", "rank", "crowding", "key", "config", *self.gene_names, *stats]
        cfg = compressor.config(compressor.uniform_genes("bf16"))
        self.baseline = {**compressor.cost(cfg), **compressor.apply(cfg)}
        compressor.restore()
        b = self.baseline
        print(f"{len(compressor.genes)} genes ({', '.join(f'{n}:{k}' for n, k in compressor.genes)}), originals "
              f"{'offloaded to CPU' if compressor.offloaded else 'kept on GPU'}", flush=True)
        print(f"Uncompressed: an agent's turn takes {b['latency_ms']:.0f} ms ({b['prompt_us']:.0f} us per prompt token, "
              f"{b['decode_us']:.0f} us per generated token), {b['size_mb']:.0f} MB stored, "
              f"{b['memory_mb']:.0f} MB deployed (KV cache {b['kv_mb']:.0f} MB), {b['agents_fit']} agents fit", flush=True)
        atomic_dump({"args": {**{k: v for k, v in vars(a).items() if k != "fixed_extra"}, "calib_every": 0},  # read by compress_compare
                     "git_commit": self.git_commit, "genes": compressor.genes, "baseline_size_mb": b["size_mb"],
                     "baseline": {"loss": self.baseline_loss, "delta_loss": 0.0, **b},
                     "latency_table": {k: decode_table[k] for k in ("gpu", "torch", "batch", "context", "formats")},
                     "layout": {"n_layer": L.n_layer, "intermediate": L.intermediate, "tied": L.tied,
                                "n_target_params": compressor.n_target_params, "rest_params": compressor.rest_params}},
                    self.out("config.json"), json)

        self.cost_objectives = [COST_STAT[o] for o in a.objectives]
        self.problem = Problem(n_var=len(compressor.genes), n_obj=1 + len(self.cost_objectives),
                               n_ieq_constr=(a.max_delta_loss is not None) + (a.memory_budget_mb is not None),
                               xl=np.zeros(len(compressor.genes)), xu=np.array([n - 1 for _, n in compressor.genes], dtype=float))
        if os.path.exists(self.out("checkpoint.pkl")):
            with open(self.out("checkpoint.pkl"), "rb") as f:
                self.algorithm = dill.load(f)
            self.algorithm.termination = get_termination("n_gen", a.n_gen)
            done_gen = self.algorithm.n_gen - 1
            self.algorithm.termination.perc = done_gen / a.n_gen
            print(f"Resumed NSGA-II after generation {done_gen}", flush=True)
        else:
            self.algorithm = NSGA2(pop_size=a.pop_size, sampling=SeededSampling(self.gene_names, seeds),
                                   crossover=SBX(prob=0.9, eta=3.0, vtype=float, repair=RoundingRepair()),
                                   mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                                   eliminate_duplicates=DecodedDuplicateElimination(compressor.genes))
            self.algorithm.setup(self.problem, termination=get_termination("n_gen", a.n_gen), seed=a.seed)
            done_gen = 0
        self.cache = {}
        if os.path.exists(self.out("evals.csv")):
            with open(self.out("evals.csv"), newline="") as f:
                for r in csv.DictReader(f):
                    self.cache[r["key"]] = {s: float(r[s]) for s in stats}
            print(f"Loaded {len(self.cache)} previous evaluations", flush=True)
        self.evals_log = CsvLog(self.out("evals.csv"), self.eval_fields)
        self.pop_log = CsvLog(self.out("population.csv"), self.pop_fields, keep_upto=done_gen)
        self.front_log = CsvLog(self.out("front.csv"), self.pop_fields, keep_upto=done_gen)
        self.gen_log = CsvLog(self.out("generations.csv"), GEN_FIELDS, keep_upto=done_gen)
        norm = self.out("normalization.json")
        self.nadir = np.array(json.load(open(norm))["nadir"]) if os.path.exists(norm) else None

    # --- search

    @staticmethod
    def key_of(genes):
        return ".".join(map(str, genes))

    def _evaluate(self, genes):
        """Calibration loss of the configuration (written into the model, then restored) and its predicted cost."""
        c = self.compressor
        cfg = c.config(genes)
        t0 = time.time()
        try:
            stored = c.apply(cfg)
            loss = self.eval_loss(self.calib)
        finally:
            c.restore()
        if not math.isfinite(loss):
            loss = 1e6  # a broken model: dominated by everything, but the run goes on
        return {"loss": loss, "delta_loss": loss - self.baseline_loss, "ppl": math.exp(min(loss, 700.0)), **c.cost(cfg), **stored,
                "seconds": round(time.time() - t0, 2)}

    def _evaluate_all(self, individuals, gen, role):
        """Objectives and constraints of each individual (from the cache when seen before). Returns (F, G, n_reused)."""
        a, c = self.args, self.compressor
        F, G, n_reused = [], [], 0
        for i, ind in enumerate(individuals):
            genes = c.decode(ind.X)
            key = self.key_of(genes)
            if key in self.cache:
                n_reused += 1
            else:
                self.cache[key] = r = self._evaluate(genes)
                text = c.describe(c.config(genes))
                self.evals_log.write([{"gen": gen, "role": role, "key": key, "config": text, **dict(zip(self.gene_names, genes)), **r}])
                print(f"gen {gen} {role} [{i + 1}/{len(individuals)}] {text} | delta {r['delta_loss']:+.4f} turn {r['latency_ms']:.0f} ms "
                      f"size {r['size_mb']:.0f} MB memory {r['memory_mb']:.0f} MB ({r['seconds']:.1f}s)", flush=True)
            r = self.cache[key]
            F.append([r["delta_loss"]] + [r[s] for s in self.cost_objectives])
            G.append(([r["delta_loss"] - a.max_delta_loss] if a.max_delta_loss is not None else [])
                     + ([r["memory_mb"] - a.memory_budget_mb] if a.memory_budget_mb is not None else []))
        return np.array(F), np.array(G), n_reused

    def _rows(self, individuals, gen):
        c, out = self.compressor, []
        for ind in individuals:
            genes = c.decode(ind.X)
            out.append({"gen": gen, "rank": ind.get("rank"), "crowding": ind.get("crowding"), "key": self.key_of(genes),
                        "config": c.describe(c.config(genes)), **dict(zip(self.gene_names, genes)), **self.cache[self.key_of(genes)]})
        return out

    def run(self):
        a, alg, c = self.args, self.algorithm, self.compressor
        while alg.has_next():
            t_gen = time.time()
            torch.cuda.reset_peak_memory_stats()
            gen = 1 if alg.n_gen is None else alg.n_gen  # pymoo advances the counter inside ask()
            offspring = alg.ask()
            F, G, n_reused = self._evaluate_all(offspring, gen, "initial" if gen == 1 else "offspring")
            Evaluator().eval(StaticProblem(self.problem, F=F, **({"G": G} if self.problem.n_ieq_constr else {})), offspring)
            alg.tell(infills=offspring)

            opt = alg.opt[alg.opt.get("feas")]
            front_F = opt.get("F")
            if self.nadir is None and len(front_F):  # fixed once, from the first front with a feasible solution
                self.nadir = front_F.max(axis=0)
                if self.nadir[0] <= 0:  # the whole front is at least as good as the original: use the population's worst
                    self.nadir[0] = max(alg.pop.get("F")[:, 0].max(), a.max_delta_loss or 1.0)
                atomic_dump({"nadir": self.nadir.tolist(), "objectives": ["delta_loss", *self.cost_objectives], "gen": gen},
                            self.out("normalization.json"), json)
            hv = float(HV(ref_point=np.ones(self.problem.n_obj))(front_F / self.nadir)) if len(front_F) else 0.0
            best = lambda col: min(self.cache[self.key_of(c.decode(x))][col] for x in opt.get("X")) if len(opt) else None
            summary = {"gen": gen, "hypervolume": hv, "front_size": len(opt), "n_feasible": int(alg.pop.get("feas").sum()),
                       "n_offspring": len(offspring), "n_reused": n_reused, "best_delta_loss": best("delta_loss"),
                       "min_latency_ms": best("latency_ms"), "min_size_mb": best("size_mb"), "min_memory_mb": best("memory_mb"),
                       "seconds": round(time.time() - t_gen, 1), "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6)}
            self.pop_log.write(self._rows(alg.pop, gen))
            self.front_log.write(self._rows(opt, gen))
            self.gen_log.write([summary])
            atomic_dump(alg, self.out("checkpoint.pkl"), dill)  # last: a crash before this re-runs the generation from the cache
            print(f"=== generation {gen}/{a.n_gen} done in {summary['seconds']:.0f}s ({n_reused} reused), HV {hv:.5f}, "
                  f"front {len(opt)} points, {summary['n_feasible']}/{len(alg.pop)} feasible" +
                  (f", best delta loss {summary['best_delta_loss']:+.4f}, fastest turn {summary['min_latency_ms']:.0f} ms, "
                   f"smallest {summary['min_size_mb']:.0f} MB stored, {summary['min_memory_mb']:.0f} MB deployed" if len(opt)
                   else ", no feasible solution yet"), flush=True)

    # --- the final front: summary, then test-set scores (resumable: test_scores.json is written after each solution)

    def score(self):
        a, b, alg, c = self.args, self.baseline, self.algorithm, self.compressor
        first = self.cost_objectives[0]
        front = sorted(self._rows(alg.opt[alg.opt.get("feas")], alg.n_gen - 1), key=lambda r: r[first])
        print(f"\nFinal front ({len(front)} points, sorted by {first}); uncompressed: turn {b['latency_ms']:.0f} ms, "
              f"{b['size_mb']:.0f} MB stored, {b['memory_mb']:.0f} MB deployed", flush=True)
        print(f"{'delta':>8} {'turn':>9} {'stored':>7} {'deployed':>8} {'agents':>6}  configuration", flush=True)
        for r in front:
            print(f"{r['delta_loss']:+8.4f} {r['latency_ms']:7.0f}ms {r['size_mb']:5.0f}MB {r['memory_mb']:6.0f}MB {r['agents_fit']:6.0f}  {r['config']}", flush=True)
        if a.wikitext_seqs == 0 and a.c4_test_seqs == 0:
            print("Test scoring skipped (no test windows)", flush=True)
            return
        scored = [r for r in front if r["delta_loss"] <= a.test_max_delta_loss]
        if a.test_max_points is not None and len(scored) > a.test_max_points:
            scored = [scored[j] for j in np.linspace(0, len(scored) - 1, a.test_max_points).round().astype(int)]
        settings = {"seq_len": a.test_seq_len, "wikitext_seqs": a.wikitext_seqs, "c4_test_seqs": a.c4_test_seqs}
        path = self.out("test_scores.json")
        results = json.load(open(path)) if os.path.exists(path) else None
        if results is None or results["settings"] != settings:
            results = {"settings": settings, "baseline": None, "pareto": []}  # the layout of compress_score's
        done = {p["key"] for p in results["pareto"]}
        test_sets = compress_eval.load_test_sets(self.tokenizer, a.test_seq_len, a.wikitext_seqs, a.c4_test_seqs, cache_key=a.model)
        print(f"Scoring {len(scored)} front solutions on " + ", ".join(f"{n} ({t.size(0)} windows)" for n, t in test_sets.items()), flush=True)
        if results["baseline"] is None:
            results["baseline"] = {k: b[k] for k in ("latency_ms", "size_mb", "memory_mb")}
            for name, tokens in test_sets.items():
                results["baseline"][f"{name}_loss"] = loss = self.eval_loss(tokens)
                results["baseline"][f"{name}_ppl"] = math.exp(loss)
            atomic_dump(results, path, json)
        for r in scored:
            if r["key"] in done:
                continue
            t0 = time.time()
            entry = {k: r[k] for k in ("key", "config", *self.gene_names, *self.stats)}
            try:
                c.apply(c.config(c.decode([r[n] for n in self.gene_names])))
                for name, tokens in test_sets.items():
                    entry[f"{name}_loss"] = loss = self.eval_loss(tokens)
                    entry[f"{name}_ppl"] = math.exp(min(loss, 700.0))
            finally:
                c.restore()
            results["pareto"].append(entry)
            atomic_dump(results, path, json)
            print(f"  {r['latency_ms']:.0f} ms {r['memory_mb']:.0f} MB: " + ", ".join(
                f"{n} ppl {entry[f'{n}_ppl']:.2f} (base {results['baseline'][f'{n}_ppl']:.2f})" for n in test_sets)
                + f" ({time.time() - t0:.0f}s)", flush=True)
        print(f"Done. Test scores in {path}", flush=True)


def parser(description, objectives, out_dir):
    p = argparse.ArgumentParser(description=description)
    add_arguments(p, objectives, out_dir)
    return p
