#!/bin/bash
# Run ORPO v5 training — targeted abstention + fresh valid-set SQL quality pairs.
#
# What's new vs v4:
#   - Starts from fully-trained ORPO v4 Colab adapter (327/327 steps)
#   - Uses unified system prompt (training now matches eval)
#   - Unanswerable rejected = model's own inference output (not random SQL)
#   - Fresh data: 760 valid answerable + all train failures (verify-execution)
#   - Lower LR (2e-6) since already well-converged
#   - 1 epoch
#
# Run pair generation first:
#   bash scripts/run_build_pairs_v5.sh
#
# Then train:
#   bash scripts/run_orpo_v5.sh

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

_NV_CU13=$(python3 -c "
import sys, os
sp = next((p for p in sys.path if 'site-packages' in p), '')
lib = os.path.join(sp, 'nvidia', 'cu13', 'lib')
print(lib)
" 2>/dev/null || echo "")
if [ -n "$_NV_CU13" ] && [ -d "$_NV_CU13" ]; then
    export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH set: $_NV_CU13"
fi

PAIRS="data/pairs/orpo_v5_pairs.jsonl"
ADAPTER="checkpoints/orpo_v4_colab/adapter_final"
OUTPUT="checkpoints/orpo_v5"
EPOCHS=1
LR="2e-6"
MAX_LENGTH="1536"
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs)   PAIRS="$2";   shift 2 ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --epochs)  EPOCHS="$2";  shift 2 ;;
        --lr)      LR="$2";      shift 2 ;;
        --resume-from-checkpoint) RESUME="--resume-from-checkpoint $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$PAIRS" ]; then
    echo "ERROR: Pairs file not found: $PAIRS"
    echo "Run: bash scripts/run_build_pairs_v5.sh"
    exit 1
fi

mkdir -p logs "$OUTPUT"

N_PAIRS=$(wc -l < "$PAIRS")
LOGFILE="logs/orpo_v5_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " ORPO v5 Training"
echo " Pairs   : $PAIRS  ($N_PAIRS pairs)"
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
echo " ORPO v5 training complete. Adapter: $OUTPUT/adapter_final"
echo " $(date)"
echo "============================================================"
echo ""
echo "Run eval with:"
echo "  bash scripts/run_sft_eval.sh \\"
echo "    --adapter $OUTPUT/adapter_final \\"
echo "    --output tests/evalgen/orpo_v5_results.json \\"
echo "    --repair --few-shot --retrieval-mode template \\"
echo "    --classifier-cache data/ehrsql2024/template_classifier.pkl"
