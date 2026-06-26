#!/bin/bash
# Run GRPO v2 training with execution reward on MIMIC-IV-Demo.
#
# Usage:
#   bash scripts/run_grpo.sh [--adapter PATH] [--output DIR] [--epochs N]
#                            [--num-generations K] [--temperature T]
#                            [--max-examples N] [--resume-from-checkpoint PATH]
#
# Defaults: adapter=checkpoints/orpo_v3/adapter_final, output=checkpoints/grpo_v2,
#           epochs=1, K=4, temperature=1.5, answerable-only (no --max-examples limit)
#
# Smoke test (2000 examples, ~30-40 min):
#   bash scripts/run_grpo.sh --max-examples 2000 --output checkpoints/grpo_v2_smoke
#
# If OOM at K=4, rerun with --num-generations 2.

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# bitsandbytes: add CUDA 13 lib path before Python starts
_NV_CU13=$(python3 -c "
import sys, os
sp = next(p for p in sys.path if 'site-packages' in p)
print(os.path.join(sp, 'nvidia', 'cu13', 'lib'))
" 2>/dev/null || echo "")
if [ -n "$_NV_CU13" ] && [ -d "$_NV_CU13" ]; then
    export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH set: $_NV_CU13"
fi

# Check for zombie GPU processes before training
echo "GPU memory before training:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true

ADAPTER="checkpoints/orpo_v3/adapter_final"
OUTPUT="checkpoints/grpo_v2"
DATA="data/ehrsql/sft_train_v2.jsonl"
EPOCHS=1
NUM_GEN=4
TEMPERATURE="1.5"
MAX_EXAMPLES=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter) ADAPTER="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --num-generations) NUM_GEN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
        --resume-from-checkpoint) RESUME="--resume-from-checkpoint $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs "$OUTPUT"

LOGFILE="logs/grpo_train_$(date +%Y%m%d_%H%M%S).log"

MAX_EX_LABEL="${MAX_EXAMPLES:-all}"
echo "============================================================"
echo " GRPO v2 Training"
echo " Adapter     : $ADAPTER"
echo " Data        : $DATA"
echo " Output      : $OUTPUT"
echo " Epochs      : $EPOCHS  |  K rollouts: $NUM_GEN  |  Temp: $TEMPERATURE"
echo " Max examples: $MAX_EX_LABEL"
echo " Log         : $LOGFILE"
echo " $(date)"
echo "============================================================"

MAX_EX_FLAG=""
if [ -n "$MAX_EXAMPLES" ]; then
    MAX_EX_FLAG="--max-examples $MAX_EXAMPLES"
fi

python3 -m ehrcopilot.finetune.grpo_train \
    --data "$DATA" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --num-generations "$NUM_GEN" \
    --temperature "$TEMPERATURE" \
    $MAX_EX_FLAG \
    $RESUME \
    2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo " GRPO training complete. Adapter: $OUTPUT/adapter_final"
echo " $(date)"
echo "============================================================"

echo ""
echo "Run eval with:"
echo "  bash scripts/run_sft_eval.sh --adapter $OUTPUT/adapter_final --output tests/evalgen/grpo_v2_results.json --repair --few-shot"
