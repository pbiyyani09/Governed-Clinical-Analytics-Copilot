#!/bin/bash
# Build ORPO v5 preference pairs — using the EHRSQL 2024 MIMIC-IV dataset.
#
# Data: EHRSQL 2024 (data/ehrsql2024/mimic_iv/)
#   - train:     5124 Qs (4674 answerable, 450 unanswerable)
#   - train_aug: 38689 Qs (35356 answerable, 3333 unanswerable)
#   - valid:     1163 Qs (931 answerable, 232 unanswerable)
#   - test:      1167 Qs (934 answerable, 233 unanswerable)
#
# Gold SQL execution on EHRSQL 2024 DB (no remaps needed — correct schema):
#   - 89-91% of answerable gold SQL executes with real data
#   - 9-10% returns empty (patient not in 94-patient subset)
#   - 0% errors (all tables/columns present)
#
# Pair strategy:
#   Step 1: Abstention pairs from valid+train unanswerable (682 total)
#           rejected = model's own inference output (stronger signal)
#   Step 2: SQL quality pairs from valid answerable (max=800)
#           only where gold executes with data (verify-execution)
#   Step 3: SQL quality pairs from train answerable (max=5000)
#           only where gold executes with data (verify-execution)
#
# Usage:
#   bash scripts/run_build_pairs_v5.sh              # full build
#   bash scripts/run_build_pairs_v5.sh fast         # abstention + 200 valid only

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
BASE="data/ehrsql2024/mimic_iv"
TRAIN_DIR="$BASE/train"
VALID_DIR="$BASE/valid"
OUTPUT_ABSTAIN="data/pairs/orpo_v5_abstention_pairs.jsonl"
OUTPUT_VALID="data/pairs/orpo_v5_valid_pairs.jsonl"
OUTPUT_TRAIN="data/pairs/orpo_v5_train_pairs.jsonl"
OUTPUT_FINAL="data/pairs/orpo_v5_pairs.jsonl"

mkdir -p data/pairs logs

echo "============================================================"
echo " ORPO v5 Pair Generation (EHRSQL 2024 MIMIC-IV)"
echo " Adapter: $ADAPTER"
echo " Mode: $MODE"
echo " $(date)"
echo "============================================================"

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: Adapter not found at $ADAPTER"
    echo "Extract the Colab zip first."
    exit 1
fi

# ── Step 1: Abstention pairs (valid unanswerable=232 + train unanswerable=450) ──
echo ""
echo "Step 1/3: Abstention pairs (valid+train unanswerable, model inference as rejected) ..."
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$TRAIN_DIR" \
    --valid "$VALID_DIR" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_ABSTAIN" \
    --unanswerable-only \
    --inference-rejected \
    2>&1 | tee logs/v5_abstention.log
echo "  Abstention pairs: $(wc -l < "$OUTPUT_ABSTAIN")"

# ── Step 2: SQL quality pairs from valid set (only questions where gold has data) ──
MAX_VALID=800
if [ "$MODE" = "fast" ]; then MAX_VALID=200; fi

echo ""
echo "Step 2/3: SQL quality pairs from valid set (max=$MAX_VALID, verify-execution) ..."
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$VALID_DIR" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_VALID" \
    --max-answerable "$MAX_VALID" \
    --verify-execution \
    --num-samples 2 \
    2>&1 | tee logs/v5_valid.log
echo "  Valid answerable pairs: $(wc -l < "$OUTPUT_VALID")"

# ── Step 3: SQL quality pairs from train set (only questions where gold has data) ──
if [ "$MODE" = "fast" ]; then
    echo "Step 3/3: Skipping train pairs (fast mode)"
    cat "$OUTPUT_ABSTAIN" "$OUTPUT_VALID" > "$OUTPUT_FINAL"
else
    MAX_TRAIN=5000
    echo ""
    echo "Step 3/3: SQL quality pairs from train set (max=$MAX_TRAIN, verify-execution) ..."
    echo "  ~4185/4674 train answerable questions have valid gold SQL"
    python3 -m ehrcopilot.finetune.build_pairs \
        --train "$TRAIN_DIR" \
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
echo ""
echo "Next: bash scripts/run_orpo_v5.sh"
