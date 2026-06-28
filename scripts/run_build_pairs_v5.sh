#!/bin/bash
# Build ORPO v5 preference pairs using the fully-trained ORPO v4 Colab adapter.
#
# What's new vs v4:
#   - Uses unified config.SYSTEM_PROMPT (training prompt now matches eval prompt)
#   - Unanswerable rejected: model's own inference output, not random gold SQL
#     (stronger signal — model learns to prefer [ABSTAIN] over its own hallucinations)
#   - Answerable source: 760 valid answerable questions (fresh, not in v4)
#   - verify_execution: skip pairs where model already executes correctly
#
# Data composition (target):
#   362 valid unanswerable  → abstention pairs (inference-rejected)
#   760 valid answerable    → SQL quality pairs (verify-execution filtered)
#   ~4500 train answerable  → SQL quality pairs (re-run with new model's failures)
#
# Usage:
#   bash scripts/run_build_pairs_v5.sh              # full v5 build
#   bash scripts/run_build_pairs_v5.sh fast         # abstention + 500 valid only (test run)

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
fi

MODE="${1:-full}"
ADAPTER="checkpoints/orpo_v4_colab/adapter_final"
TRAIN="data/ehrsql/ehrsql/mimic_iii/train.json"
VALID="data/ehrsql/ehrsql/mimic_iii/valid.json"
OUTPUT_ABSTAIN="data/ehrsql/orpo_v5_abstention_pairs.jsonl"
OUTPUT_VALID="data/ehrsql/orpo_v5_valid_pairs.jsonl"
OUTPUT_TRAIN="data/ehrsql/orpo_v5_train_pairs.jsonl"
OUTPUT_FINAL="data/ehrsql/orpo_v5_pairs.jsonl"

mkdir -p data/ehrsql logs

echo "============================================================"
echo " ORPO v5 Pair Generation"
echo " Adapter: $ADAPTER"
echo " Mode: $MODE"
echo " $(date)"
echo "============================================================"

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: Adapter not found at $ADAPTER"
    echo "Extract the Colab zip first: unzip models/adapter_final-*.zip -d checkpoints/orpo_v4_colab/"
    exit 1
fi

# ── Step 1: Abstention pairs from valid unanswerable (inference-rejected) ──
echo ""
echo "Step 1/3: Abstention pairs (362 valid unanswerable, inference-rejected) ..."
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$TRAIN" \
    --valid "$VALID" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_ABSTAIN" \
    --unanswerable-only \
    --inference-rejected \
    2>&1 | tee logs/v5_abstention.log

echo "  Abstention pairs: $(wc -l < "$OUTPUT_ABSTAIN")"

# ── Step 2: SQL quality pairs from valid answerable (760 questions) ──
MAX_VALID=760
if [ "$MODE" = "fast" ]; then MAX_VALID=200; fi

echo ""
echo "Step 2/3: SQL quality pairs from valid answerable (max=$MAX_VALID, verify-execution) ..."
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$VALID" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_VALID" \
    --max-answerable "$MAX_VALID" \
    --verify-execution \
    --num-samples 2 \
    2>&1 | tee logs/v5_valid.log

echo "  Valid answerable pairs: $(wc -l < "$OUTPUT_VALID")"

# ── Step 3: SQL quality pairs from train set (model's current failures) ──
MAX_TRAIN=5000
if [ "$MODE" = "fast" ]; then
    echo "Step 3/3: Skipping train pairs (fast mode)"
    cat "$OUTPUT_ABSTAIN" "$OUTPUT_VALID" > "$OUTPUT_FINAL"
else
    echo ""
    echo "Step 3/3: SQL quality pairs from train (max=$MAX_TRAIN, verify-execution) ..."
    python3 -m ehrcopilot.finetune.build_pairs \
        --train "$TRAIN" \
        --adapter "$ADAPTER" \
        --output "$OUTPUT_TRAIN" \
        --max-answerable "$MAX_TRAIN" \
        --verify-execution \
        --num-samples 2 \
        2>&1 | tee logs/v5_train.log

    echo "  Train pairs: $(wc -l < "$OUTPUT_TRAIN")"
    cat "$OUTPUT_ABSTAIN" "$OUTPUT_VALID" "$OUTPUT_TRAIN" > "$OUTPUT_FINAL"
fi

echo ""
echo "============================================================"
echo " ORPO v5 pairs written to: $OUTPUT_FINAL"
N_TOTAL=$(wc -l < "$OUTPUT_FINAL")
N_UNANS=$(python3 -c "
import json
n = sum(1 for l in open('$OUTPUT_FINAL') if not json.loads(l).get('is_answerable', True))
print(n)
")
N_ANS=$((N_TOTAL - N_UNANS))
echo " Total pairs: $N_TOTAL  (answerable: $N_ANS, unanswerable: $N_UNANS)"
echo " $(date)"
echo "============================================================"
