#!/bin/bash
# Build ORPO v4 preference pairs from the full 9318-question training set.
#
# Key difference vs ORPO v3: uses all training questions (not just 503).
# ORPO v3 left ~8800 questions on the table. This run captures every
# question ORPO v3 still gets wrong (~3500-4500 pairs expected) to give
# 7x more gradient steps targeting the hard cluster.
#
# Pair format:
#   unanswerable: chosen=[ABSTAIN], rejected=random gold SQL (same as ORPO v3)
#   answerable:   chosen=gold_sql,  rejected=model's first wrong rollout
#                 --verify-execution: skip if model already executes correctly
#
# Expected runtime: ~8-10 hours on RTX 4080 Super
# Output: data/ehrsql/orpo_v4_pairs.jsonl

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
fi

ADAPTER="${1:-checkpoints/orpo_v3/adapter_final}"
OUTPUT="${2:-data/ehrsql/orpo_v4_pairs.jsonl}"

mkdir -p logs

LOGFILE="logs/build_orpo_v4_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " ORPO v4 Pair Building"
echo " Adapter : $ADAPTER"
echo " Output  : $OUTPUT"
echo " Training: data/ehrsql/ehrsql/mimic_iii/train.json (9318 Qs)"
echo " Valid   : data/ehrsql/ehrsql/mimic_iii/valid.json (unanswerable)"
echo " Flags   : --verify-execution --num-samples 2 --max-answerable 9318"
echo " Log     : $LOGFILE"
echo " $(date)"
echo "============================================================"

python3 -m ehrcopilot.finetune.build_pairs \
    --train data/ehrsql/ehrsql/mimic_iii/train.json \
    --valid data/ehrsql/ehrsql/mimic_iii/valid.json \
    --adapter "$ADAPTER" \
    --output "$OUTPUT" \
    --max-answerable 9318 \
    --verify-execution \
    --num-samples 2 \
    2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo " Pair building complete: $OUTPUT"
echo " $(date)"
echo "============================================================"
echo ""
echo "Run ORPO v4 training with:"
echo "  bash scripts/run_orpo_v4.sh --pairs $OUTPUT --adapter $ADAPTER"
