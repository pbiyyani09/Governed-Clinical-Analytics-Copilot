#!/bin/bash
# Build ORPO v5 preference pairs — clean rebuild from scratch.
#
# What's correct this time:
#   - Gold SQL canonicalized (MIMIC-III→MIMIC-IV schema remaps applied)
#   - Gold SQL verified: skip if errors OR returns empty on our DB
#   - Chosen = canonicalized gold SQL (not raw MIMIC-III SQL)
#   - Rejected = model's own output (stronger signal for abstention pairs)
#   - Only valid-gold questions produce pairs (no backwards training signal)
#
# Data composition:
#   362 unanswerable  → abstention pairs (model's actual output as rejected)
#   up to 2784 train answerable where gold executes with real data
#   ~800 valid answerable where gold executes with real data
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
TRAIN="data/ehrsql/ehrsql/mimic_iii/train.json"
VALID="data/ehrsql/ehrsql/mimic_iii/valid.json"
OUTPUT_ABSTAIN="data/ehrsql/orpo_v5_abstention_pairs.jsonl"
OUTPUT_VALID="data/ehrsql/orpo_v5_valid_pairs.jsonl"
OUTPUT_TRAIN="data/ehrsql/orpo_v5_train_pairs.jsonl"
OUTPUT_FINAL="data/ehrsql/orpo_v5_pairs.jsonl"

mkdir -p data/ehrsql logs

echo "============================================================"
echo " ORPO v5 Pair Generation (clean rebuild)"
echo " Adapter: $ADAPTER"
echo " Mode: $MODE"
echo " $(date)"
echo "============================================================"

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: Adapter not found at $ADAPTER"
    echo "Extract the Colab zip first."
    exit 1
fi

# ── Step 1: Abstention pairs (all 362 valid unanswerable, inference-rejected) ──
echo ""
echo "Step 1/3: Abstention pairs (362 unanswerable, model inference as rejected) ..."
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$TRAIN" \
    --valid "$VALID" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_ABSTAIN" \
    --unanswerable-only \
    --inference-rejected \
    2>&1 | tee logs/v5_abstention.log
echo "  Abstention pairs: $(wc -l < "$OUTPUT_ABSTAIN")"

# ── Step 2: SQL quality pairs from valid set (only valid-gold questions) ──
MAX_VALID=800
if [ "$MODE" = "fast" ]; then MAX_VALID=200; fi

echo ""
echo "Step 2/3: SQL quality pairs from valid set (max=$MAX_VALID, verify-execution) ..."
echo "  Note: skips questions where gold SQL errors or returns empty (MIMIC-IV-Demo mismatch)"
python3 -m ehrcopilot.finetune.build_pairs \
    --train "$VALID" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT_VALID" \
    --max-answerable "$MAX_VALID" \
    --verify-execution \
    --num-samples 2 \
    2>&1 | tee logs/v5_valid.log
echo "  Valid answerable pairs: $(wc -l < "$OUTPUT_VALID")"

# ── Step 3: SQL quality pairs from train set (only valid-gold questions) ──
if [ "$MODE" = "fast" ]; then
    echo "Step 3/3: Skipping train pairs (fast mode)"
    cat "$OUTPUT_ABSTAIN" "$OUTPUT_VALID" > "$OUTPUT_FINAL"
else
    MAX_TRAIN=5000
    echo ""
    echo "Step 3/3: SQL quality pairs from train set (max=$MAX_TRAIN, verify-execution) ..."
    echo "  Note: only ~2784/9318 train questions have valid gold SQL → expect ~2000-2500 pairs"
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
echo ""
echo "Next: bash scripts/run_orpo_v5.sh"
