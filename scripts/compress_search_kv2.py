"""
The KV search, second version: multi-objective search (NSGA-II) over the KV cache and the weight formats, with the cache
genes the KV probe (compress_runs/kv_probe) found worth searching: keys with one scale per channel and group of tokens,
values per token, windows that keep the first token, and planted-fact recall as a constraint. See nanochat/compress_kv2.py.
Needs Triton (the decode kernel that reads the quantized cache in place).

Objectives (all minimized; delta_loss always, --objectives adds the others):
    delta_loss  next-token cross-entropy on the calibration batch minus the uncompressed model's
    time        predicted GPU time (ms) of one agent turn on this GPU (as scripts/compress_search_kv.py). Logged as latency_ms
    memory      deployed weights plus the KV cache of --serve-batch agents (MB). Logged as memory_mb
    size        stored size (MB). Logged as size_mb
Constraints: delta_loss <= --max-delta-loss, and recall_acc >= --min-recall: the fraction of --recall-seqs facts, each
planted at a random depth of a --seq-len window of calibration text, that the configuration still gives back exactly when
asked at the end of the window (recall_ce: cross-entropy of the answers). The uncompressed model recalls all of them on Qwen2.5-0.5B.

Outputs in <out-dir>: as scripts/compress_search_kv.py, with kv2_table.json in place of kv_table.json (also cached per GPU
in $COMPRESS_CACHE) and recall_acc, recall_ce in evals.csv. Rerunning the same command resumes. The GPU should otherwise be
idle while the tables are measured.

    python -m scripts.compress_search_kv2 --out-dir compress_runs/qwen_kv2
"""

from nanochat import compress_kv2, compress_mix
from scripts.compress_search_common import Search, parser

STATS = ["loss", "delta_loss", "ppl", "recall_acc", "recall_ce", "latency_ms", "decode_us", "prompt_us", "size_mb", "memory_mb",
         "weights_mb", "kv_mb", "kv_pct", "window_pct", "agents_fit", "bits_per_weight", "int4_pct", "shared_pct", "k_merged",
         "mlp_kept_pct", "n_removed_attn", "n_removed_mlp"]

p = parser("NSGA-II search over the KV cache (per-channel keys, recall-checked windows) and the weight formats",
           ["time", "memory"], "compress_runs/kv2_default")
p.add_argument("--kv-keys", type=str, default=",".join(compress_kv2.KEY_KINDS), help="precisions of the cached keys (bf16 first)")
p.add_argument("--kv-values", type=str, default=",".join(compress_kv2.VALUE_KINDS), help="precisions of the cached values (bf16 first)")
p.add_argument("--kv-windows", type=str, default=",".join(map(str, compress_kv2.KV2_WINDOWS)), help="fractions of the context a band can keep (1 is always included)")
p.add_argument("--kv-group", type=int, default=compress_kv2.GROUP, help="tokens per key scale")
p.add_argument("--recall-seqs", type=int, default=64, help="planted facts, one per window")
p.add_argument("--min-recall", type=float, default=0.95, help="constraint: fraction of the facts recalled exactly")
p.add_argument("--recall-batch", type=int, default=8)
args = p.parse_args()
args.fixed_extra = ("kv_keys", "kv_values", "kv_windows", "kv_group", "recall_seqs", "min_recall")


class RecallSearch(Search):
    extra_constraints = (lambda r: args.min_recall - r["recall_acc"],)

    def extra_scores(self):
        ce, acc = compress_kv2.recall_score(self.model, self.recall_tokens, self.recall_answers, args.recall_batch)
        return {"recall_acc": acc, "recall_ce": ce}

    def note(self, r):
        return f" recall {r['recall_acc']:.0%}"


search = RecallSearch(args, "compress_search_kv2")
filler = [t for row in search.calib.tolist() for t in row]
search.recall_tokens, search.recall_answers = compress_kv2.recall_set(search.tokenizer, filler, args.recall_seqs, args.seq_len, args.data_seed)
decode_table, prompt_table = search.turn_tables(compress_mix.build_prompt_table)
keys = [x for x in args.kv_keys.split(",") if x]
values = [x for x in args.kv_values.split(",") if x]
assert keys[0] == "bf16" and values[0] == "bf16", "the first precision is the uncompressed one: bf16"
windows = sorted({float(w) for w in args.kv_windows.split(",")} | {1.0}, reverse=True)
import triton  # noqa: E402  (a clear error before the table, not inside it)
kv2_table = search.table("kv2_table.json", "KV2", lambda: compress_kv2.build_kv2_table(
    search.layout, keys, values, windows, args.serve_batch, args.prompt_batch, args.context, search.device, args.kv_group,
    log=lambda m: print(m, flush=True)), batch=args.serve_batch,
    extra={"prompt_batch": args.prompt_batch, "keys": keys, "values": values, "windows": windows, "group": args.kv_group,
           "sinks": compress_kv2.SINKS, "triton": triton.__version__, "kernel_version": compress_kv2.KERNEL_VERSION})

compressor = compress_kv2.KV2Compressor(search.model, search.layout, decode_table, prompt_table, kv2_table, search.act_sq, search.removal,
                                        context=args.context, **search.compressor_options())
seeds = [compressor.uniform_genes(f) for f in compressor.formats] + compressor.extra_seeds()
search.start(compressor, decode_table, STATS, seeds)
if search.baseline["recall_acc"] < args.min_recall:
    print(f"WARNING: the uncompressed model recalls only {search.baseline['recall_acc']:.0%} of the facts, below --min-recall "
          f"{args.min_recall}: nothing can be feasible", flush=True)
print(f"Uncompressed: recall {search.baseline['recall_acc']:.0%} of {args.recall_seqs} facts (answer cross-entropy "
      f"{search.baseline['recall_ce']:.3f})", flush=True)
search.run()
search.score()
