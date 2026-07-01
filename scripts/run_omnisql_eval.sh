#!/bin/bash
# OmniSQL-7B zero-shot baseline evaluation on EHRSQL 2024 MIMIC-IV
# Uses K=10 hybrid retrieval from combined train+aug corpus (40K examples).
#
# Usage:
#   bash scripts/run_omnisql_eval.sh [OPTIONS]
#
# Options:
#   --output PATH          JSON metrics output (default: tests/evalgen/omnisql_7b_results.json)
#   --preds PATH           Per-prediction JSONL output (default: tests/evalgen/omnisql_7b_preds.jsonl)
#   --repair               Enable execution-guided repair loop (up to 3 retries)
#   --num-samples N        Self-consistency voting over N completions
#   --few-shot-k K         Number of retrieval examples (default: 10)
#   --model MODEL          HuggingFace model ID (default: seeklhy/OmniSQL-7B)

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# bitsandbytes fix: conda ships libnvJitLink.so.13 in its nvidia/cu13/lib directory
_NV_CU13=$(python3 -c "
import sys, os
sp = next(p for p in sys.path if 'site-packages' in p)
lib = os.path.join(sp, 'nvidia', 'cu13', 'lib')
print(lib)
" 2>/dev/null || echo "")
if [ -n "$_NV_CU13" ] && [ -d "$_NV_CU13" ]; then
    export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH set: $_NV_CU13"
fi

OUTPUT="tests/evalgen/omnisql_7b_results.json"
PREDS="tests/evalgen/omnisql_7b_preds.jsonl"
MODEL="seeklhy/OmniSQL-7B"
REPAIR_FLAG=""
NUM_SAMPLES_FLAG=""
FEW_SHOT_K="10"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT="$2"; shift 2 ;;
        --preds)  PREDS="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --repair) REPAIR_FLAG="--repair"; shift ;;
        --num-samples) NUM_SAMPLES_FLAG="--num-samples $2"; shift 2 ;;
        --few-shot-k) FEW_SHOT_K="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$PREDS")"

echo "============================================================"
echo " OmniSQL Evaluation"
echo " Model   : $MODEL"
echo " Output  : $OUTPUT"
echo " Retrieval: Hybrid K=$FEW_SHOT_K (train+aug, 40K corpus)"
echo " $(date)"
echo "============================================================"

python3 -m ehrcopilot.eval.harness \
    data/ehrsql2024/mimic_iv/test \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --save-predictions "$PREDS" \
    --few-shot data/ehrsql2024/mimic_iv/train \
    --retrieval-mode hybrid \
    --retrieval-aug data/ehrsql2024/mimic_iv/train_aug \
    --few-shot-k "$FEW_SHOT_K" \
    --embed-cache data/ehrsql2024/mimic_iv/train_combined_embeddings_bge_large.npy \
    $REPAIR_FLAG \
    $NUM_SAMPLES_FLAG \
    2>&1 | tee logs/omnisql_eval.log

echo ""
echo "============================================================"
echo " Evaluation complete: $OUTPUT"
echo " $(date)"
echo "============================================================"

# Compare against SFT and baseline results
python3 -c "
import json, sys, os

result_file = '$OUTPUT'
if not os.path.exists(result_file):
    print('Result file not found:', result_file)
    sys.exit(0)

s = json.load(open(result_file))
print()
print('  OmniSQL Results')
print('  ---------------')
for k in ['EX', 'RS(0)', 'RS(5)', 'RS(10)']:
    print(f'  {k:11s}: {s.get(k, \"N/A\")}')
print()
for k in ['correct_answers', 'wrong_abstentions', 'wrong_answers_on_unanswerable', 'correct_abstentions']:
    print(f'  {k}: {s.get(k, \"N/A\")}')

# Side-by-side with SFT if available
sft_file = 'tests/evalgen/sft_results.json'
if os.path.exists(sft_file):
    b = json.load(open(sft_file))
    print()
    print('  Metric      | SFT       | OmniSQL-7B | Delta')
    print('  ------------|-----------|------------|----------')
    for k in ['EX', 'RS(0)', 'RS(5)', 'RS(10)']:
        bv = b.get(k, 0.0); sv = s.get(k, 0.0); d = sv - bv
        sign = '+' if d >= 0 else ''
        print(f'  {k:11s} | {bv:9.4f} | {sv:10.4f} | {sign}{d:.4f}')
    print()
    for k in ['correct_answers', 'wrong_abstentions', 'wrong_answers_on_unanswerable', 'correct_abstentions']:
        bv = b.get(k, 0); sv = s.get(k, 0)
        print(f'  {k:36s}: SFT={bv}  OmniSQL={sv}  delta={sv-bv:+d}')
"
