#!/bin/bash
# NSGA-II compression search on the published nanochat d34 base model (karpathy/nanochat-d34, Nov 2025:
# 2.22B parameters, vocab 65536, CORE 0.338) on one GPU: global uniform quantization + pruning of its
# 204 transformer matrices (1.93B weights, 87% of the parameters); the token embedding and lm_head
# stay 16-bit, as in the Qwen runs. The checkpoint predates nanochat's current architecture, so it is
# built with nanochat/gpt_legacy.py (verified to give bit-identical logits to the code it was trained with).
#
# Needs: uv sync --extra gpu --group compress && source .venv/bin/activate
#        internet access to Hugging Face (the checkpoint, ~8.6 GB, is downloaded on the first run;
#        C4 train for calibration; WikiText-2 test and C4 validation for the final test scores)
#
# Usage (in screen/tmux; every step resumes where it stopped, so just re-run after an interruption):
#   SMOKE=1 bash runs/compress_d34.sh   # quick check: download, load, 2 tiny generations; prints the
#                                       # time of one forward pass over 8 calibration windows
#   bash runs/compress_d34.sh           # the real run
# Settings to override with environment variables:
#   CALIB_SEQS  calibration windows of 2048 tokens (default 128, the GPTQ size). The time per
#               generation is ~100 x (forward time over CALIB_SEQS windows + ~1 s compression).
#   EVAL_BATCH  windows per forward pass (default 2: fits a 24 GB L4; 8 on an 80 GB H100)
#   N_GEN, POP  generations (default 10) and population size (default 100)
#   C_MIN       lowest clipping fraction searched (default 0.9)
#   PRUNE_MAX   widest pruning bound searched, in row std units: alpha >= -PRUNE_MAX, beta <= PRUNE_MAX (0.3)
#   MAX_DELTA   constraint: candidates with a larger calibration delta_loss are infeasible (default 1.0)
#   EXTRA       any further compress_search options, e.g. EXTRA="--warmup-max-error 0.002"
#
# Search: fixed calibration batch of random C4-train windows; log2 K with its range [K_min, K_max] found
# by the warm-up (--auto-k-bounds: uniform quantization only, K_min = largest K with >50% perplexity
# increase, K_max = smallest K with <=0.5%, both snapped outward to powers of two); c in [C_MIN, 1] and
# pruning within +-PRUNE_MAX sigma, narrowed for d34, which the first 25 random candidates of a wide
# search showed breaks at c < 0.9 or wide pruning; constraint delta_loss <= 1.0. The final front is
# scored on WikiText-2 test and C4 validation with 2048-token windows (pareto.json), then plotted.

set -euo pipefail
cd "$(dirname "$0")/.."

# a base dir of its own, so the d34 tokenizer (vocab 65536) never replaces another nanochat tokenizer
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-d34}
CKPT_DIR=$NANOCHAT_BASE_DIR/base_checkpoints/d34
if [ ! -f "$CKPT_DIR/meta_169150.json" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Downloading karpathy/nanochat-d34 into $NANOCHAT_BASE_DIR"
    python - <<EOF
from huggingface_hub import hf_hub_download
for name, sub in [("tokenizer.pkl", "tokenizer"), ("token_bytes.pt", "tokenizer"),
                  ("meta_169150.json", "base_checkpoints/d34"), ("model_169150.pt", "base_checkpoints/d34")]:
    print(hf_hub_download("karpathy/nanochat-d34", name, local_dir="$NANOCHAT_BASE_DIR/" + sub), flush=True)
EOF
fi

MODEL=nanochat:base:d34
CALIB_SEQS=${CALIB_SEQS:-128}
EVAL_BATCH=${EVAL_BATCH:-2}
N_GEN=${N_GEN:-10}
C_MIN=${C_MIN:-0.9}
PRUNE_MAX=${PRUNE_MAX:-0.3}
MAX_DELTA=${MAX_DELTA:-1.0}
if [ "${SMOKE:-0}" = "1" ]; then
    OUT=${OUT:-compress_runs/d34_smoke}
    ARGS="--pop-size 4 --n-gen 2 --calib-seqs 8 --wikitext-seqs 4 --c4-test-seqs 4"
else
    OUT=${OUT:-compress_runs/d34_${N_GEN}_log2_autok_c${C_MIN}_prune${PRUNE_MAX}_calib_fixed_${CALIB_SEQS}x2048_maxdelta${MAX_DELTA}}
    ARGS="--pop-size ${POP:-100} --n-gen $N_GEN --calib-seqs $CALIB_SEQS"
fi
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$OUT/run.log"
python -u -m scripts.compress_search --model "$MODEL" $ARGS \
    --seq-len 2048 --calib-every 0 --k-space log2 --auto-k-bounds \
    --c-min "$C_MIN" --alpha-min "-$PRUNE_MAX" --beta-max "$PRUNE_MAX" --max-delta-loss "$MAX_DELTA" \
    --eval-batch-size "$EVAL_BATCH" --out-dir "$OUT" ${EXTRA:-} 2>&1 | tee -a "$OUT/run.log"
python -m scripts.compress_plot --run-dir "$OUT" --all 2>&1 | tee -a "$OUT/run.log"
echo "done: $OUT (pareto.json = test scores of the final front)" | tee -a "$OUT/run.log"
