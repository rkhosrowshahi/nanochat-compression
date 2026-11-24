"""
Multi-objective search (NSGA-II, no backpropagation) over the weight format of every matrix of a causal LM, for a Pareto front
of models that trade quality against decode speed and GPU memory. See nanochat/compress_tpot.py (TPOTCompressor) for the
decision variables and the costs, and nanochat/compress_eval.py for the quality measure and the data.

Decision variables: the format of each role (qkv, o, up, down) in each of --bands depth bands (--bands = number of layers:
one gene per matrix role and layer), and of the output head; formats among bf16 and int4 with groups of 32/64/128 (PyTorch's
tinygemm kernel). MLP pruning and sublayer removal are off by default (--mlp-keep, --max-skip turn them on).

Objectives (all minimized):
    kld     mean KL divergence from the uncompressed model's next-token distribution over the whole vocabulary, on
            --calib-seqs windows of --seq-len tokens of coding conversations (bigcode/self-oss-instruct-sc2-exec-filter-50k
            in the model's chat template; 2,000 documents held out for test)
    time    TPOT: the latency table's decode step at batch 1 over --context tokens (default 640: the mean context of a
            512-token prompt and 256 generated tokens). Logged as tpot_ms (= latency_ms) and tok_s
    memory  deployed weights (MB = 1e6 bytes; weights_gib also logged): int4 codes with their scales, the rest in bf16, a
            tied embedding counted again when the head is int4
Constraint (optional): --max-kld.

The latency table is cached per GPU, so a table measured by scripts/compress_tpot_measure.py on the same settings is reused.
Measure front solutions for real with scripts/compress_tpot_measure.py.

Outputs in <out-dir>: as scripts/compress_search_mix_simple.py (evals.csv: every evaluation; front.csv, population.csv,
generations.csv, config.json, checkpoint.pkl: rerunning the same command resumes), and test_scores.json: KLD, top-1 agreement
and perplexity of the final front on held-out code (--code-test-seqs) and WikiText-2 / C4 test windows.

    python -m scripts.compress_search_tpot --model Qwen/Qwen2.5-Coder-1.5B-Instruct --bands 28 --out-dir compress_runs/coder15_tpot

    # quick local smoke test
    python -m scripts.compress_search_tpot --model Qwen/Qwen2.5-Coder-1.5B-Instruct --pop-size 6 --n-gen 2 --calib-seqs 1 \
        --code-test-seqs 1 --wikitext-seqs 1 --out-dir compress_runs/tpot_smoke
"""

import math
import time

from nanochat import compress_eval, compress_hw, compress_tpot
from scripts.compress_search_common import Search, parser

STATS = ["kld", "kld_median", "kld_p99", "top1_agree", "loss", "delta_loss", "tpot_ms", "latency_ms", "tok_s", "memory_mb",
         "size_mb", "weights_gib", "bits_per_weight", "int4_pct", "mlp_kept_pct", "n_removed_attn", "n_removed_mlp"]
GiB_PER_MB = 1e6 / (1 << 30)

p = parser("NSGA-II search over the weight format of every matrix, for KL divergence, decode TPOT and weight memory",
           ["time", "memory"], "compress_runs/tpot_default")
p.set_defaults(calib_dataset=compress_eval.CODE_DATASET, calib_config="chat", calib_split="calib", context=640,
               mlp_keep="1.0", max_skip=0, max_delta_loss=None, c4_test_seqs=0, wikitext_seqs=32,
               offload_originals="yes")  # an 8 GB GPU holds the model and the reference head, not a second copy of the weights
p.add_argument("--max-kld", type=float, default=None, help="constraint: kld above this is infeasible")
p.add_argument("--test-max-kld", type=float, default=1.0, help="only score front solutions with kld up to this")
p.add_argument("--code-test-seqs", type=int, default=64, help="held-out code windows for the final front (0 = skip)")
args = p.parse_args()
args.fixed_extra = ("max_kld",)
assert set(args.objectives) <= {"time", "memory"}, "--objectives: time and/or memory (size is the same as memory here)"


class TPOTSearch(Search):
    quality = "kld"

    def __init__(self, args, script):
        super().__init__(args, script)
        self.quality_limit, self.test_quality_limit = args.max_kld, args.test_max_kld
        t0 = time.time()
        self.reference = compress_eval.reference_hidden(self.model, self.calib, args.eval_batch_size)
        print(f"Reference for the KL divergence: inputs to the output head at {self.reference['hidden'].shape[:2].numel()} "
              f"positions ({self.reference['hidden'].numel() * 2 / 1e6:.0f} MB) in {time.time() - t0:.1f}s", flush=True)

    def _sample_windows(self):
        a = self.args
        return compress_eval.code_instruct_windows(self.tokenizer, a.seq_len, a.calib_seqs, "calib", seed=a.data_seed)

    def quality_scores(self):
        r = compress_eval.eval_kld(self.model, self.calib, self.reference, self.args.eval_batch_size)
        if not math.isfinite(r["kld"]):
            r = {**r, "kld": 1e6}  # a broken model: dominated by everything, but the run goes on
        return {**r, "delta_loss": r["loss"] - self.baseline_loss}

    def result_text(self, r):
        return (f"KLD {r['kld']:.4f} TPOT {r['tpot_ms']:.2f} ms ({r['tok_s']:.0f} tok/s) "
                f"weights {r['memory_mb'] * GiB_PER_MB:.3f} GiB")

    def baseline_text(self, b):
        return (f"Uncompressed: TPOT {b['tpot_ms']:.2f} ms ({b['tok_s']:.0f} tok/s) at batch 1 over {self.args.context} tokens "
                f"of context, weights {b['memory_mb'] * GiB_PER_MB:.3f} GiB")

    def generation_text(self, s):
        return (f", best KLD {s['best_kld']:.4f}, fastest {s['min_latency_ms']:.2f} ms/token, "
                f"smallest {s['min_memory_mb'] * GiB_PER_MB:.3f} GiB")

    def front_text(self, r):
        if r is None:
            return f"{'KLD':>8} {'TPOT':>9} {'tok/s':>6} {'weights':>10}  configuration"
        return (f"{r['kld']:8.4f} {r['tpot_ms']:7.2f}ms {r['tok_s']:6.0f} {r['memory_mb'] * GiB_PER_MB:7.3f}GiB  "
                f"{r['config']}")

    def test_sets(self):
        a = self.args
        sets = {}
        if a.code_test_seqs != 0:
            sets["code"] = compress_eval.code_instruct_windows(self.tokenizer, a.test_seq_len, a.code_test_seqs, "test")
        return {**sets, **super().test_sets()}

    def prepare_test(self, test_sets):
        self.test_refs = {name: compress_eval.reference_hidden(self.model, tokens, self.args.eval_batch_size)
                          for name, tokens in test_sets.items()}

    def test_scores(self, name, tokens):
        r = compress_eval.eval_kld(self.model, tokens, self.test_refs[name], self.args.eval_batch_size)
        return {f"{name}_kld": r["kld"], f"{name}_top1": r["top1_agree"], f"{name}_loss": r["loss"],
                f"{name}_ppl": math.exp(min(r["loss"], 700.0))}

    def test_text(self, name, entry, baseline):
        return (f"{name} KLD {entry[f'{name}_kld']:.4f} top-1 {100 * entry[f'{name}_top1']:.1f}% "
                f"ppl {entry[f'{name}_ppl']:.3f} (base {baseline[f'{name}_ppl']:.3f})")


search = TPOTSearch(args, "compress_search_tpot")
table = search.table("latency_table.json", "decode", lambda: compress_hw.build_latency_table(
    search.model, search.layout, search.formats, search.mlp_keep, 1, args.context, log=lambda m: print(m, flush=True)), batch=1)
unavailable = sorted(set(search.formats) - set(table["formats"]))
assert not unavailable, f"no kernel on this GPU runs {unavailable}: drop them from --formats"
o = search.compressor_options()
compressor = compress_tpot.TPOTCompressor(search.model, search.layout, table, search.act_sq, search.removal, formats=o["formats"],
                                          bands=o["bands"], mlp_keep=o["mlp_keep"], max_skip=o["max_skip"],
                                          compress_embeddings=o["compress_embeddings"], clip_ratios=o["clip_ratios"],
                                          offload=o["offload"])
uniform = [compressor.uniform_genes(f) for f in compressor.formats]
head = [name for name, _ in compressor.genes].index("lm_head")
blocks_only = [g[:head] + (0,) + g[head + 1:] for g in uniform[1:]]  # int4 blocks, bf16 head: the GPTQ/AWQ setting
search.start(compressor, table, STATS, uniform + blocks_only)
search.run()
search.score()
