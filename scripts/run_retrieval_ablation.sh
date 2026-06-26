#!/bin/bash
# Isolated retrieval embedding-model ablation.
#
# Each embedding model is embedded in its OWN python process (--embed-only) so a
# model whose custom CUDA kernel throws a device-side assert (e.g. gte's RoPE on
# transformers 5.5) cannot poison the CUDA context for the rest of the ablation.
# A final scoring pass reads the cached embeddings (CPU only) and reports metrics.
#
# Usage: bash scripts/run_retrieval_ablation.sh [repr1 repr2 ...]   (default: q q_sql)

set -uo pipefail
cd "$(dirname "$0")/.."

REPRS="${*:-q q_sql}"
PYBIN=.venv/bin/python
CACHE=data/ehrsql/embed_cache
OUT=tests/evalgen/retrieval_bench_full.json
KVALS="1 2 3 5 10 20 50 100 500 1000"

export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(grep -E "^HF_TOKEN" .env 2>/dev/null | sed -E 's/^HF_TOKEN[[:space:]]*=[[:space:]]*//' | tr -d '"' | tr -d ' ')

# Models that load cleanly under transformers 5.5 (+ embeddinggemma, attempted).
# Skipped (custom code incompatible with transformers 5.5): gte-large, gte-qwen2-1.5b, sfr-code-2b.
MODELS="bge-large mxbai arctic-l e5-large nomic qwen3-0.6b qwen3-4b embeddinggemma"

echo "=== Isolated embedding pass (reprs: $REPRS) — $(date) ==="
for m in $MODELS; do
  echo ">>> embed $m"
  $PYBIN -m ehrcopilot.eval.retrieval_bench --embed-only "$m" --reprs $REPRS --cache-dir "$CACHE" \
    2>&1 | grep -vE "Batches|it/s\]|LOAD REPORT|position_ids|UNEXPECTED|can be ignored|^Key |^---|^Notes" | tail -4
  echo ">>> $m exit=${PIPESTATUS[0]}"
done

echo ""
echo "=== Scoring pass (reads caches; CPU) — $(date) ==="
$PYBIN -m ehrcopilot.eval.retrieval_bench \
  --models bm25 $MODELS --reprs $REPRS --index flat \
  --k-values $KVALS --cache-dir "$CACHE" --output "$OUT"
echo "=== DONE $(date) ==="
