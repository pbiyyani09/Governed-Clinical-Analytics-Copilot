#!/bin/bash
# Run post-SFT evaluation using the SFT LoRA adapter via Unsloth.
# LD_LIBRARY_PATH must be set here (before Python starts) so that bitsandbytes
# can find libnvJitLink.so.13 needed for 4-bit forward passes.
#
# Usage:
#   bash scripts/run_sft_eval.sh [--output PATH] [--split PATH] [--adapter PATH]
#                                [--repair] [--few-shot] [--retrieval-mode MODE]
#                                [--retrieval-aug PATH]
#
# --repair           enables execution-guided repair loop (up to 3 retries per failed SQL)
# --few-shot         enables RAG few-shot retrieval (uses --retrieval-mode; default: hybrid)
# --retrieval-mode   hybrid (default) | bm25 | embed | template
# --retrieval-aug    path to train_aug split; adds ~35K examples to retrieval corpus
# Default output: tests/evalgen/sft_results.json

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

OUTPUT="tests/evalgen/sft_results.json"
SPLIT="data/ehrsql2024/mimic_iv/test"
ADAPTER="checkpoints/sft/adapter_final"
TRAIN="data/ehrsql2024/mimic_iv/train"
REPAIR_FLAG=""
FEW_SHOT_FLAG=""
NUM_SAMPLES_FLAG=""
RETRIEVAL_MODE_FLAG=""
RETRIEVAL_AUG_FLAG=""
CLASSIFIER_CACHE_FLAG=""
ENTROPY_FLAG=""
ABSTAIN_EMPTY_FLAG=""
ABSTAIN_ERROR_FLAG=""
SAVE_PREDS_FLAG=""
FEW_SHOT_K_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        --repair) REPAIR_FLAG="--repair"; shift ;;
        --few-shot) FEW_SHOT_FLAG="--few-shot $TRAIN"; shift ;;
        --few-shot-k) FEW_SHOT_K_FLAG="--few-shot-k $2"; shift 2 ;;
        --num-samples) NUM_SAMPLES_FLAG="--num-samples $2"; shift 2 ;;
        --retrieval-mode) RETRIEVAL_MODE_FLAG="--retrieval-mode $2"; shift 2 ;;
        --retrieval-aug) RETRIEVAL_AUG_FLAG="--retrieval-aug $2"; shift 2 ;;
        --classifier-cache) CLASSIFIER_CACHE_FLAG="--classifier-cache $2"; shift 2 ;;
        --entropy-threshold) ENTROPY_FLAG="--entropy-threshold $2"; shift 2 ;;
        --abstain-on-empty) ABSTAIN_EMPTY_FLAG="--abstain-on-empty"; shift ;;
        --abstain-on-error) ABSTAIN_ERROR_FLAG="--abstain-on-error"; shift ;;
        --save-predictions) SAVE_PREDS_FLAG="--save-predictions $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs "$(dirname "$OUTPUT")"

echo "============================================================"
echo " Post-SFT Evaluation"
echo " Adapter : $ADAPTER"
echo " Split   : $SPLIT"
echo " Output  : $OUTPUT"
echo " $(date)"
echo "============================================================"

python3 -m ehrcopilot.eval.harness \
    "$SPLIT" \
    --model "$ADAPTER" \
    --output "$OUTPUT" \
    $REPAIR_FLAG \
    $FEW_SHOT_FLAG \
    $FEW_SHOT_K_FLAG \
    $RETRIEVAL_MODE_FLAG \
    $RETRIEVAL_AUG_FLAG \
    $CLASSIFIER_CACHE_FLAG \
    $NUM_SAMPLES_FLAG \
    $ENTROPY_FLAG \
    $ABSTAIN_EMPTY_FLAG \
    $ABSTAIN_ERROR_FLAG \
    $SAVE_PREDS_FLAG \
    2>&1 | tee logs/sft_eval.log

echo ""
echo "============================================================"
echo " Evaluation complete: $OUTPUT"
echo " $(date)"
echo "============================================================"

# Print comparison with baseline
if [ -f tests/evalgen/baselines.json ]; then
    python3 -c "
import json
b = json.load(open('tests/evalgen/baselines.json'))
s = json.load(open('$OUTPUT'))
print()
print('  Metric      | Baseline  | Post-SFT  | Delta')
print('  ------------|-----------|-----------|----------')
for k in ['EX', 'RS(0)', 'RS(5)', 'RS(10)']:
    bv = b.get(k, 0.0)
    sv = s.get(k, 0.0)
    d = sv - bv
    sign = '+' if d >= 0 else ''
    print(f'  {k:11s} | {bv:9.4f} | {sv:9.4f} | {sign}{d:.4f}')
print()
for k in ['correct_answers', 'wrong_abstentions', 'wrong_answers_on_unanswerable', 'correct_abstentions']:
    bv = b.get(k, 0)
    sv = s.get(k, 0)
    print(f'  {k:36s}: baseline={bv}  sft={sv}  Δ={sv-bv:+d}')
"
fi
