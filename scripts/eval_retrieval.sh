#!/bin/bash
# Measure RAG retrieval quality (RAGAS-style) for BM25, embedding, and hybrid modes.
#
# Uses the EHRSQL `tag` field as ground-truth relevance signal:
#   A retrieved training example is "relevant" if its abstract template (`tag`)
#   matches the test question's template — no LLM-as-judge needed.
#
# Metrics computed:
#   - Context Recall@K  (Hit Rate@K): did we retrieve any relevant example in top-K?
#   - Context Precision@K: what fraction of top-K retrieved examples are relevant?
#   - MRR: mean reciprocal rank of first relevant result
#
# Usage:
#   bash scripts/eval_retrieval.sh              # runs all three modes (bm25, embed, hybrid)
#   bash scripts/eval_retrieval.sh bm25         # BM25 baseline only
#   bash scripts/eval_retrieval.sh hybrid       # hybrid only
#   bash scripts/eval_retrieval.sh all          # explicit all (default)
#
# Output is written to tests/evalgen/rag_eval_results.json

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

MODE="${1:-all}"
TRAIN="data/ehrsql/ehrsql/mimic_iii/train.json"
TEST="data/ehrsql/ehrsql/mimic_iii/test.json"
OUTPUT="tests/evalgen/rag_eval_results.json"
EMBED_CACHE="data/ehrsql/train_embeddings_bge_large.npy"

mkdir -p tests/evalgen logs

echo "============================================================"
echo " RAG Retrieval Quality Evaluation"
echo " Mode    : $MODE"
echo " Train   : $TRAIN"
echo " Test    : $TEST"
echo " Output  : $OUTPUT"
echo " $(date)"
echo "============================================================"

python3 -m ehrcopilot.eval.rag_eval \
    --train "$TRAIN" \
    --test  "$TEST" \
    --mode  "$MODE" \
    --embed-cache "$EMBED_CACHE" \
    --output "$OUTPUT" \
    --k 1,2,3,5,10 \
    2>&1 | tee logs/rag_eval.log

echo ""
echo "============================================================"
echo " Retrieval eval complete: $OUTPUT"
echo " $(date)"
echo "============================================================"
