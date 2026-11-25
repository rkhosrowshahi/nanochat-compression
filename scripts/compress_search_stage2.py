"""
Stage 2 of the two-stage search: NSGA-II over weight sharing + pruning (UBQ-ECSQ) on the Pareto front of a finished hardware-aware
search (stage 1, --stage1-dir: a run of scripts/compress_search_tpot.py, or of compress_search_hw.py). See nanochat/compress_stage2.py
for the genes. Each solution is a stage-1 solution (its kernels, so its TPOT and GPU memory) with some of its bf16 matrices stored as
shared codes: stage 2 trades quality against the STORED size (download, disk, load time), and keeps time and memory.

Objectives (all minimized):
    kld     mean KL divergence from the uncompressed model, on the stage-1 run's calibration windows (code conversations)
    time    the base's TPOT (latency_ms = tpot_ms, from stage 1)
    memory  the base's deployed weights (memory_mb, from stage 1)
    size    stored size (size_mb), entropy coded
(--objectives picks the costs; with size alone NSGA-II ignores how the bases differ in speed and memory.)
Constraint (optional): --max-kld.

The bases: the front of ALL stage-1 evaluations (evals.csv: NSGA-II's final population drops some of it) on stage 1's own
objectives, with quality up to --base-max-kld, at most --max-bases of them spread along TPOT, and always the uncompressed model.
They are saved to <out-dir>/bases.csv the first time and read back from it on a resume, so the base gene keeps its meaning.
The model, the calibration windows (calib_pool.pt), the activation statistics (act_stats.pt: the int4 clipping), the search
space and the evaluation batch size (bf16 forwards in batches of another size shift the KLD by ~0.001-0.003) are stage 1's:
those arguments are taken from its config.json, whatever is given here.

Outputs in <out-dir>: as scripts/compress_search_tpot.py (evals.csv: every evaluation; front.csv, population.csv, generations.csv,
config.json, checkpoint.pkl: rerunning the same command resumes), plus bases.csv, front_all.csv (the front of every evaluation,
which the test scores use) and test_scores.json (code KLD, top-1 agreement and perplexity on held-out text).

    python -m scripts.compress_search_stage2 --stage1-dir compress_runs/coder15_tpot_a100 --out-dir compress_runs/coder15_stage2

    # quick local smoke test on a finished stage-1 smoke run
    python -m scripts.compress_search_stage2 --stage1-dir compress_runs/tpot_smoke --pop-size 8 --n-gen 2 --code-test-seqs 1 \
        --wikitext-seqs 1 --test-max-points 2 --out-dir compress_runs/stage2_smoke
"""

import csv
import json
import os
import shutil
import sys

import torch

from nanochat import compress_stage2
from nanochat.compress_mix import SHARED
from scripts.compress_search_common import COST_STAT, GiB_PER_MB, KLDSearch, parser

STATS = ["kld", "kld_median", "kld_p99", "top1_agree", "loss", "delta_loss", "base", "base_quality", "tpot_ms", "latency_ms",
         "tok_s", "memory_mb", "weights_gib", "size_mb", "stored_bits_per_weight", "deployed_bits_per_weight", "shared_pct",
         "k_merged", "int4_pct"]
# taken from the stage-1 run: what the model, its data and the deployed formats are, and how its quality was measured
# (eval_batch_size too: bf16 forwards in batches of another size shift the KLD by ~0.001-0.003)
INHERITED = ("model", "dtype", "eval_batch_size", "calib_dataset", "calib_config", "calib_split", "calib_seqs", "seq_len",
             "data_seed", "text_field", "importance_seqs", "context", "formats", "bands", "mlp_keep", "max_skip", "no_embeddings", "clip_ratios")
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))  # the configuration strings of per-layer genes are long

p = parser("Stage 2: NSGA-II over weight sharing + pruning on the front of a hardware-aware search, for KL divergence and stored "
           "size at the stage-1 TPOT and memory", ["time", "memory", "size"], "compress_runs/stage2_default")
p.set_defaults(model=None, max_delta_loss=None, c4_test_seqs=0, wikitext_seqs=32, test_max_points=8)
p.add_argument("--stage1-dir", type=str, required=True, help="the finished stage-1 run")
p.add_argument("--max-bases", type=int, default=32, help="stage-1 front solutions to start from, spread along TPOT")
p.add_argument("--base-max-kld", type=float, default=None, help="only stage-1 solutions with quality up to this")
p.add_argument("--share-bands", type=int, default=4, help="depth bands of the per-role sharing genes")
p.add_argument("--share-k", type=str, default=",".join(map(str, compress_stage2.SHARE_K)), help="grid levels")
p.add_argument("--share-c", type=str, default=",".join(map(str, compress_stage2.SHARE_C)), help="clipping of each row's range")
p.add_argument("--share-prune", type=str, default=",".join(map(str, compress_stage2.SHARE_PRUNE)),
               help="pruning interval: weights within +-beta row stds become zero")
p.add_argument("--share-rho", type=str, default=",".join(map(str, compress_stage2.SHARE_RHO)), help="merge: fraction of levels kept")
p.add_argument("--max-kld", type=float, default=None, help="constraint: kld above this is infeasible")
p.add_argument("--test-max-kld", type=float, default=1.0, help="only score front solutions with kld up to this")
p.add_argument("--code-test-seqs", type=int, default=64, help="held-out code windows for the final front (0 = skip)")
args = p.parse_args()

stage1_path = lambda name: os.path.join(args.stage1_dir, name)
assert os.path.exists(stage1_path("config.json")) and os.path.exists(stage1_path("evals.csv")), \
    f"{args.stage1_dir} is not a search run (no config.json / evals.csv)"
stage1 = json.load(open(stage1_path("config.json")))
run1 = stage1["args"]
if args.model is not None and args.model != run1["model"]:
    sys.exit(f"--model {args.model}: stage 1 ({args.stage1_dir}) compressed {run1['model']}")
for key in INHERITED:
    setattr(args, key, run1[key])
args.calib_from = args.stage1_dir
args.fixed_extra = ("stage1_dir", "max_bases", "base_max_kld", "share_bands", "share_k", "share_c", "share_prune", "share_rho",
                    "max_kld")
os.makedirs(args.out_dir, exist_ok=True)
for name in ("act_stats.pt", "removal.json"):  # the same statistics as stage 1: its int4 clipping and its removal order
    if os.path.exists(stage1_path(name)) and not os.path.exists(os.path.join(args.out_dir, name)):
        shutil.copy(stage1_path(name), os.path.join(args.out_dir, name))
floats = lambda text: [float(v) for v in text.split(",") if v]


class Stage2Search(KLDSearch):
    def result_text(self, r):
        return (f"KLD {r['kld']:.4f} (base {r['base_quality']:.4f}) stored {r['size_mb'] * GiB_PER_MB:.3f} GiB "
                f"({r['stored_bits_per_weight']:.2f} bits/w) | TPOT {r['tpot_ms']:.2f} ms weights {r['memory_mb'] * GiB_PER_MB:.3f} GiB")

    def baseline_text(self, b):
        return (f"Uncompressed: TPOT {b['tpot_ms']:.2f} ms ({b['tok_s']:.0f} tok/s), weights {b['memory_mb'] * GiB_PER_MB:.3f} GiB "
                f"deployed, {b['size_mb'] * GiB_PER_MB:.3f} GiB stored")

    def generation_text(self, s):
        return (f", best KLD {s['best_kld']:.4f}, fastest {s['min_latency_ms']:.2f} ms/token, "
                f"smallest stored {s['min_size_mb'] * GiB_PER_MB:.3f} GiB")

    def front_text(self, r):
        if r is None:
            return f"{'KLD':>8} {'base KLD':>8} {'stored':>9} {'TPOT':>9} {'weights':>9}  configuration"
        return (f"{r['kld']:8.4f} {r['base_quality']:8.4f} {r['size_mb'] * GiB_PER_MB:6.3f}GiB {r['tpot_ms']:7.2f}ms "
                f"{r['memory_mb'] * GiB_PER_MB:6.3f}GiB  {r['config']}")

    def note(self, r):
        """With nothing shared a solution is its base: its KLD must be the stage-1 value (same windows, same weights)."""
        if quality1 == "kld" and r["shared_pct"] == 0 and abs(r["kld"] - r["base_quality"]) > 1e-4:
            return f" [WARNING: stage 1 measured KLD {r['base_quality']:.4f} for this model]"
        return ""

    def final_front(self):
        """The feasible front of every evaluation (the final population drops some of it), also saved as front_all.csv."""
        c = self.compressor
        rows = []
        for key, r in self.cache.items():
            if self.quality_limit is not None and r[self.quality] > self.quality_limit:
                continue
            genes = tuple(int(v) for v in key.split("."))
            rows.append({"gen": self.algorithm.n_gen - 1, "rank": 0, "crowding": float("nan"), "key": key,
                         "config": c.describe(c.config(genes)), **dict(zip(self.gene_names, genes)), **r})
        front = compress_stage2.pareto_front(rows, [self.quality, *self.cost_objectives])
        with open(self.out("front_all.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.pop_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(front, key=lambda r: r[self.cost_objectives[0]]))
        return front


def load_bases(search):
    """The stage-1 solutions of the base gene: from this run's bases.csv, or chosen now (then saved)."""
    gene_names = [name for name, _ in stage1["genes"]]
    path = search.out("bases.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            bases = [{k: (v if k in ("key", "config") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        print(f"{len(bases)} bases from {path}", flush=True)
        return bases
    with open(stage1_path("evals.csv"), newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [{k: (v if k in ("role", "key", "config") else float(v)) for k, v in r.items()} for r in reader]
    quality = "kld" if "kld" in fields else "delta_loss"
    objectives = [COST_STAT[o] for o in run1["objectives"]]
    uncompressed = ".".join(map(str, uncompressed_genes(gene_names)))
    bases = compress_stage2.select_bases(rows, quality, objectives, args.base_max_kld, args.max_bases,
                                         is_uncompressed=lambda r: r["key"] == uncompressed)
    keep = ["key", "config", *gene_names, quality, *(s for s in ("tpot_ms", "latency_ms", "memory_mb", "bits_per_weight",
                                                                   "int4_pct") if s in fields)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(keep)), extrasaction="ignore")
        w.writeheader()
        w.writerows(bases)
    print(f"{len(bases)} bases: the front of {len(rows)} stage-1 evaluations on ({quality}, {', '.join(objectives)}), "
          f"saved to {path}", flush=True)
    return bases


def uncompressed_genes(gene_names):
    """Stage-1 genes of the uncompressed model: every option 0 (bf16 is the first format, keep 1, nothing removed)."""
    assert compress_stage2.stage1_formats(run1)[0] == "bf16", "stage 1 had no bf16 option: no uncompressed base"
    return [0] * len(gene_names)


search = Stage2Search(args, "compress_search_stage2")
stage1_calib = torch.load(stage1_path("calib_pool.pt"))["tokens"][:run1["calib_seqs"]]
assert torch.equal(search.calib, stage1_calib), "the calibration windows are not stage 1's: the KLDs would not be comparable"
bases = load_bases(search)
quality1 = "kld" if "kld" in bases[0] else "delta_loss"
table = json.load(open(stage1_path("latency_table.json")))  # which kernels run which shapes; the costs come from the bases
compressor = compress_stage2.Stage2Compressor(search.model, search.layout, table, search.act_sq, search.removal, run1, bases,
                                              quality=quality1, share_bands=args.share_bands, share_k=[int(k) for k in floats(args.share_k)],
                                              share_c=floats(args.share_c), share_prune=floats(args.share_prune),
                                              share_rho=floats(args.share_rho), offload=args.offload_originals)
print(f"Stage 1: {args.stage1_dir} ({run1['model']}, latency table of {table.get('gpu')}, batch {table.get('batch')}, "
      f"context {table.get('context')})", flush=True)
# seeds: every base as it is (stage 1's front, so stage 2's front can only improve on it), then bases with all their bf16
# matrices shared on the mildest candidate, from the slowest base down
seeds = [compressor.no_sharing_genes(b) for b in range(len(bases))]
seeds += [g for b in reversed(range(len(bases))) for g in [compressor.all_shared_genes(b, share_k=len(compressor.options["share_k"]) - 1)]
          if compressor.shared_any(compressor.config(g))]
seeds = list(dict.fromkeys(seeds))[:args.pop_size]
search.start(compressor, table, STATS, seeds)
search.run()
search.score()
n_shared = sum(1 for r in search.final_front() if r["shared_pct"] > 0)
print(f"Front of every evaluation: {search.out('front_all.csv')} ({n_shared} with shared matrices). Shared matrices run as bf16 "
      f"with the codebook's values ({SHARED} format): TPOT and memory are their base's.", flush=True)
