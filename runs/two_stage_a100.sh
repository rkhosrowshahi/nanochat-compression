#!/bin/bash
# Two-stage compression search on one A100 (Qwen2.5-Coder-1.5B-Instruct):
#   stage 1  hardware-aware formats (bf16 / int4 tinygemm per matrix and layer) for KL divergence, TPOT and weight memory
#            configs/compress/coder15_tpot_a100.yaml -> compress_runs/coder15_tpot_a100
#   stage 2  weight sharing + pruning (UBQ-ECSQ) of the bf16 matrices of the stage-1 front, for KL divergence and stored size
#            at the stage-1 TPOT and memory: configs/compress/coder15_stage2.yaml -> compress_runs/coder15_stage2
# Every step resumes: rerun the same command after an interruption. The GPU should otherwise be idle while stage 1 measures
# its latency table (the first minutes).
#
#   screen -L -Logfile runs/two_stage_a100.log -S two_stage bash runs/two_stage_a100.sh
#   STAGES="2" bash runs/two_stage_a100.sh        # stage 2 only (stage 1 finished)
#   SMOKE=1 bash runs/two_stage_a100.sh           # tiny runs into *_smoke dirs, to check the setup first (~10 min)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES="${STAGES:-1 2}"
SMOKE_FLAG=""
STAGE1_DIR="compress_runs/coder15_tpot_a100"
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke"
    STAGE1_DIR="compress_runs/coder15_tpot_a100_smoke"
fi

if [ -d .venv ]; then
    source .venv/bin/activate
fi
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU"
name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
print(f"GPU {name} (sm{cap[0]}{cap[1]}), torch {torch.__version__}")
assert cap >= (8, 0), "the int4 tinygemm kernel needs sm80+ (A100, H100, L4)"
EOF

# 1) the tests, on this GPU (the int4 kernel, CUDA-graph decoding, KL divergence, both stages)
python -m pytest -q tests/test_compress_hw.py tests/test_compress_tpot.py tests/test_compress_kld.py \
    tests/test_compress_mix.py tests/test_compress_mix_simple.py tests/test_compress_stage2.py

# 2) stage 1: the hardware-aware front (50 x 100 evaluations)
if [[ " $STAGES " == *" 1 "* ]]; then
    python -m scripts.compress_run configs/compress/coder15_tpot_a100.yaml $SMOKE_FLAG
fi

# 3) stage 2 on it (96 x 40 evaluations)
if [[ " $STAGES " == *" 2 "* ]]; then
    [ -f "$STAGE1_DIR/evals.csv" ] || { echo "no stage-1 run in $STAGE1_DIR: run stage 1 first"; exit 1; }
    python -m scripts.compress_run configs/compress/coder15_stage2.yaml $SMOKE_FLAG --set "search.stage1_dir=$STAGE1_DIR"
fi
echo "Done. Stage 1: $STAGE1_DIR, stage 2: compress_runs/coder15_stage2${SMOKE_FLAG:+_smoke} (front_all.csv, test_scores.json)"
