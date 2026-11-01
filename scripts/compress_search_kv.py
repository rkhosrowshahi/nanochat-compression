"""
Step 1 after the mix run: multi-objective search (NSGA-II) in which the KV cache is a decision, next to the weight formats
and the MLP width. See nanochat/compress_kv.py for the method and for what the window genes can and cannot tell. No weight
sharing here: it only changes the stored size (scripts/compress_search_mix_simple.py has that).

Objectives (all minimized; delta_loss always, --objectives adds the others):
    delta_loss  next-token cross-entropy on the calibration batch (C4 train, one fixed batch) minus the uncompressed model's
    time        predicted GPU time (ms) of one agent turn on this GPU: --new-tokens prompt tokens in chunks of --prompt-batch,
                then --gen-tokens among --serve-batch sequences, over --context tokens of cache. Logged as latency_ms
    memory      deployed weights plus the KV cache of --serve-batch agents (MB). Logged as memory_mb
    size        stored size (MB), entropy coded. Logged as size_mb
Every solution also logs the agents that fit on a --vram-gb GPU (agents_fit), the decode and prompt times per token, kv_pct
(the cache against bf16 at the whole context) and window_pct (the mean window).

Outputs in <out-dir>: as scripts/compress_search_mix.py, plus kv_table.json (read and attention times per precision and
window, also cached per GPU in $COMPRESS_CACHE). Rerunning the same command resumes. The GPU should otherwise be idle while
the tables are measured.

    python -m scripts.compress_search_kv --out-dir compress_runs/qwen_kv

    # quick local smoke test
    python -m scripts.compress_search_kv --pop-size 12 --n-gen 2 --calib-seqs 2 --wikitext-seqs 0 --c4-test-seqs 0 --out-dir compress_runs/kv_smoke
"""

from nanochat import compress_kv, compress_mix
from scripts.compress_search_common import Search, parser

STATS = ["loss", "delta_loss", "ppl", "latency_ms", "decode_us", "prompt_us", "size_mb", "memory_mb", "weights_mb", "kv_mb", "kv_pct",
         "window_pct", "agents_fit", "bits_per_weight", "int4_pct", "shared_pct", "k_merged", "mlp_kept_pct", "n_removed_attn", "n_removed_mlp"]

p = parser("NSGA-II search over the KV cache and the weight formats, for the GPU time and the memory of an agent's work",
           ["time", "memory"], "compress_runs/kv_default")
p.add_argument("--kv-precisions", type=str, default=",".join(compress_kv.KV_PRECISIONS), help="precisions of the cached keys and values")
p.add_argument("--kv-windows", type=str, default=",".join(map(str, compress_kv.KV_WINDOWS)), help="fractions of the context a band can keep (1 is always included)")
args = p.parse_args()
args.fixed_extra = ("kv_precisions", "kv_windows")

search = Search(args, "compress_search_kv")
decode_table, prompt_table = search.turn_tables(compress_mix.build_prompt_table)
precisions = [x for x in args.kv_precisions.split(",") if x]
assert precisions[0] == "bf16", "the first precision is the uncompressed one: bf16"
windows = sorted({float(w) for w in args.kv_windows.split(",")} | {1.0}, reverse=True)
kv_table = search.table("kv_table.json", "KV", lambda: compress_kv.build_kv_table(
    search.layout, precisions, windows, args.serve_batch, args.prompt_batch, args.context, decode_table["attn_impl"], search.device,
    log=lambda m: print(m, flush=True)), batch=args.serve_batch, extra={"prompt_batch": args.prompt_batch, "precisions": precisions, "windows": windows})

compressor = compress_kv.KVCompressor(search.model, search.layout, decode_table, prompt_table, kv_table, search.act_sq, search.removal,
                                      context=args.context, **search.compressor_options())
seeds = [compressor.uniform_genes(f) for f in compressor.formats] + compressor.extra_seeds()
search.start(compressor, decode_table, STATS, seeds)
search.run()
search.score()
