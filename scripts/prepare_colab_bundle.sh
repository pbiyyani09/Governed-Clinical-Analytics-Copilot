#!/bin/bash
# Package the code + data needed for heavy fine-tuning on Colab (RTX 6000 Pro 95GB)
# into a single tarball you upload to Google Drive. The Colab notebook
# (notebooks/gemma_finetune_colab.ipynb) copies it from Drive to the local Colab
# disk and runs the SFT + abstention-ORPO there; the resulting adapter is small
# and runs inference back on the local RTX 3090.
#
# Usage:
#   bash scripts/prepare_colab_bundle.sh
#   -> creates colab_bundle.tar.gz  (upload to  MyDrive/ehrcopilot/colab_bundle.tar.gz)

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=colab_bundle.tar.gz
echo "Packaging code + data into $OUT ..."

# Build the SFT data + DB if missing so the bundle is self-contained.
[ -f data/mimic_iv_demo.db ] || { echo "WARN: data/mimic_iv_demo.db missing — build it first (db/build_sqlite.sh)"; }
[ -f data/ehrsql/sft_train_v2.jsonl ] || echo "WARN: sft_train_v2.jsonl missing — notebook will re-prepare it"

tar -czf "$OUT" \
  src pyproject.toml \
  scripts/run_gemma_finetune.sh scripts/run_sft_eval.sh \
  src/ehrcopilot/db/build_sqlite.py \
  data/ehrsql/ehrsql/mimic_iii/train.json \
  data/ehrsql/ehrsql/mimic_iii/valid.json \
  data/ehrsql/ehrsql/mimic_iii/test.json \
  data/ehrsql/ehrsql/mimic_iii/test_cmp75.json \
  $( [ -f data/ehrsql/sft_train_v2.jsonl ] && echo data/ehrsql/sft_train_v2.jsonl ) \
  $( [ -f data/mimic_iv_demo.db ] && echo data/mimic_iv_demo.db )

echo ""
echo "Done: $(du -sh "$OUT" | cut -f1)  -> $OUT"
echo "Next:"
echo "  1) Upload $OUT to Google Drive at:  MyDrive/ehrcopilot/colab_bundle.tar.gz"
echo "  2) In Colab, add a secret named HF_TOKEN (your HF token)."
echo "  3) Open notebooks/gemma_finetune_colab.ipynb in Colab and Run All."
echo "  4) Download the produced checkpoints/orpo_gemma/adapter_final from Drive"
echo "     and run inference locally on the RTX 3090."
