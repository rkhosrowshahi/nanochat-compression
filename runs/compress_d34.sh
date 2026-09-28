#!/bin/bash
# NSGA-II compression search on the nanochat d34 base model (3.29B parameters) on one GPU (H100 80GB):
# global uniform quantization + pruning of the 204 transformer matrices (1.93B weights, 59% of the
# parameters); embeddings, lm_head, gates and scalars stay 16-bit, as in the Qwen runs.
#
# Needs:
#   - the checkpoint in $NANOCHAT_BASE_DIR/base_checkpoints/d34/ (model_<step>.pt + meta_<step>.json)
#     and the tokenizer in $NANOCHAT_BASE_DIR/tokenizer/ (default NANOCHAT_BASE_DIR=~/.cache/nanochat)
#   - the environment: uv sync --extra gpu --group compress && source .venv/bin/activate
#   - internet access to Hugging Face datasets (C4 train for calibration; WikiText-2 test and
#     C4 validation for the final test scores), or those datasets in the HF cache
#
# Usage (in screen/tmux; every step resumes where it stopped, so just re-run after an interruption):
#   SMOKE=1 bash runs/compress_d34.sh   # ~10 min check: loads d34, 2 tiny generations, timing + memory
#   bash runs/compress_d34.sh           # the real run
#
# Settings: fixed GPTQ-size calibration batch (128 random C4-train windows of 2048 tokens), log2 K
# with K <= 256, constraint delta_loss <= 1.0, population 100, 50 generations. The log prints how
# long one forward pass over the calibration batch takes; each generation is ~100 of those plus
# the compression itself. The final front (all solutions have delta_loss <= 1) is scored on
# WikiText-2 test and C4 validation with 2048-token windows, then every plot is made.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-nanochat:base:d34}  # nanochat:<base|sft|rl>:<tag>[:<step>], latest step if omitted
if [ "${SMOKE:-0}" = "1" ]; then
    OUT=${OUT:-compress_runs/d34_smoke}
    ARGS="--pop-size 4 --n-gen 2 --calib-seqs 8 --wikitext-seqs 4 --c4-test-seqs 4"
else
    OUT=${OUT:-compress_runs/d34_50_log2_k256_calib_fixed_128x2048_maxdelta1}
    ARGS="--pop-size 100 --n-gen 50 --calib-seqs 128"
fi
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$OUT/run.log"
python -u -m scripts.compress_search --model "$MODEL" $ARGS \
    --seq-len 2048 --calib-every 0 --k-space log2 --k-max 256 --max-delta-loss 1.0 \
    --eval-batch-size 8 --out-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
python -m scripts.compress_plot --run-dir "$OUT" --all 2>&1 | tee -a "$OUT/run.log"
echo "done: $OUT (pareto.json = test scores of the final front)" | tee -a "$OUT/run.log"
