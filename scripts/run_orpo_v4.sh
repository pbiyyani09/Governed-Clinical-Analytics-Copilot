#!/bin/bash
# Run ORPO v4 training — comprehensive SQL quality fine-tuning.
#
# Usage:
#   bash scripts/run_orpo_v4.sh [--pairs PATH] [--adapter PATH]
#                               [--output DIR] [--epochs N] [--lr FLOAT]
#
# Defaults: pairs=data/ehrsql/orpo_v4_pairs.jsonl,
#           adapter=checkpoints/orpo_v3/adapter_final,
#           output=checkpoints/orpo_v4, epochs=1, lr=5e-6
#
# ORPO v4 vs ORPO v3:
#   - 7x more answerable pairs (~3500-4500 vs 503)
#   - Full 9318-question training coverage (vs 503 sampled)
#   - 1 epoch (not 2) since dataset is 7x larger → similar total gradient steps
#   - Lower LR (5e-6) since continuing from a well-trained checkpoint
#
# After training, eval with:
#   bash scripts/run_sft_eval.sh --adapter checkpoints/orpo_v4/adapter_final
#                                --output tests/evalgen/orpo_v4_results.json
#                                --repair --few-shot

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

_NV_CU13=$(python3 -c "
import sys, os
sp = next(p for p in sys.path if 'site-packages' in p)
print(os.path.join(sp, 'nvidia', 'cu13', 'lib'))
" 2>/dev/null || echo "")
if [ -n "$_NV_CU13" ] && [ -d "$_NV_CU13" ]; then
    export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH set: $_NV_CU13"
fi

echo "GPU memory before training:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true

PAIRS="data/ehrsql/orpo_v4_pairs.jsonl"
ADAPTER="checkpoints/orpo_v3/adapter_final"
OUTPUT="checkpoints/orpo_v4"
EPOCHS=1
LR="5e-6"
MAX_LENGTH="1536"
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs) PAIRS="$2"; shift 2 ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --resume-from-checkpoint) RESUME="--resume-from-checkpoint $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs "$OUTPUT"

LOGFILE="logs/orpo_v4_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " ORPO v4 Training"
echo " Pairs   : $PAIRS"
echo " Adapter : $ADAPTER"
echo " Output  : $OUTPUT"
echo " Epochs  : $EPOCHS  |  LR: $LR"
echo " Log     : $LOGFILE"
echo " $(date)"
echo "============================================================"

python3 -m ehrcopilot.finetune.abstention_dpo \
    --pairs "$PAIRS" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAX_LENGTH" \
    $RESUME \
    2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo " ORPO v4 training complete. Adapter: $OUTPUT/adapter_final"
echo " $(date)"
echo "============================================================"
echo ""
echo "Run eval with:"
echo "  bash scripts/run_sft_eval.sh --adapter $OUTPUT/adapter_final --output tests/evalgen/orpo_v4_results.json --repair --few-shot"
