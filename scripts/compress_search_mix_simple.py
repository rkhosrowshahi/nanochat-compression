"""
Step 3 after the mix run: the search half of the simplified mix (see nanochat/compress_mix_simple.py). One format per matrix group
among the deployed ones (bf16 or int4 with a group size) and the MLP width per band, with no sharing, pruning, merge or clipping genes.
scripts/compress_storage.py is the optional storage step on its final front.

Objectives (all minimized; delta_loss always, --objectives adds the others):
    delta_loss  next-token cross-entropy on the calibration batch (C4 train, one fixed batch) minus the uncompressed model's
    time        predicted GPU time (ms) of one agent turn on this GPU: --new-tokens prompt tokens in chunks of --prompt-batch,
                then --gen-tokens among --serve-batch sequences, over --context tokens of cache. Logged as latency_ms
    memory      deployed weights plus the KV cache of --serve-batch agents (MB). Logged as memory_mb
    size        stored size (MB), entropy coded. Logged as size_mb for every solution
Outputs in <out-dir>: as scripts/compress_search_mix.py. Rerunning the same command resumes. The GPU should otherwise be idle while the
tables are measured.

    python -m scripts.compress_search_mix_simple --out-dir compress_runs/qwen_mix_simple

    # quick local smoke test
    python -m scripts.compress_search_mix_simple --pop-size 8 --n-gen 2 --calib-seqs 2 --wikitext-seqs 0 --c4-test-seqs 0 --out-dir compress_runs/mix_simple_smoke
"""

from nanochat import compress_mix
from scripts.compress_search_common import Search, parser

STATS = ["loss", "delta_loss", "ppl", "latency_ms", "decode_us", "prompt_us", "size_mb", "memory_mb", "weights_mb", "kv_mb",
         "agents_fit", "bits_per_weight", "int4_pct", "shared_pct", "k_merged", "mlp_kept_pct", "n_removed_attn", "n_removed_mlp"]

args = parser("NSGA-II search over the deployed formats and the MLP width, for the GPU time and the memory of an agent's work",
              ["time", "memory"], "compress_runs/mix_simple_default").parse_args()

search = Search(args, "compress_search_mix_simple")
decode_table, prompt_table = search.turn_tables(compress_mix.build_prompt_table)
compressor = compress_mix.MixCompressor(search.model, search.layout, decode_table, prompt_table, search.act_sq, search.removal, sharing=False,
                                        **search.compressor_options())
search.start(compressor, decode_table, STATS, [compressor.uniform_genes(f) for f in compressor.formats])
search.run()
search.score()
