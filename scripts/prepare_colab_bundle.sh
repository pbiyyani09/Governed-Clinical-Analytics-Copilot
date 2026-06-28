#!/bin/bash
# Package code + data into a ZIP for the Colab Gemma-4 fine-tune+eval notebook.
#
# Drive flow (per project setup):
#   1. This makes ehrsql_gemma_bundle.zip (code + EHRSQL data + built mimic_iv_demo.db).
#   2. Upload it to a folder in your MAIN (5 TB) Google Drive.
#   3. Share that folder as "EHRSQL_GEMMA" to the Colab Pro+ account.
#   4. In the Colab account's Drive, folder "ehrsql/" holds a SHORTCUT to EHRSQL_GEMMA.
#   5. Colab notebook reads  /content/drive/MyDrive/ehrsql/EHRSQL_GEMMA/ehrsql_gemma_bundle.zip
#      copies it to the local Colab disk, trains + evaluates, and writes the
#      (small ~0.5-1 GB LoRA) adapters back to that shared folder.
#
# Usage: bash scripts/prepare_colab_bundle.sh

set -euo pipefail
cd "$(dirname "$0")/.."
OUT=ehrsql_gemma_bundle.zip
rm -f "$OUT"

[ -f data/mimic_iv_demo.db ] || echo "WARN: data/mimic_iv_demo.db missing (notebook will rebuild from PhysioNet)"
[ -f data/ehrsql/sft_train_v2.jsonl ] || echo "WARN: sft_train_v2.jsonl missing (notebook will re-prepare)"

FILES=(
  src pyproject.toml
  scripts/run_gemma_finetune.sh
  data/ehrsql/ehrsql/mimic_iii/train.json
  data/ehrsql/ehrsql/mimic_iii/valid.json
  data/ehrsql/ehrsql/mimic_iii/test.json
  data/ehrsql/ehrsql/mimic_iii/test_cmp75.json
)
[ -f data/ehrsql/sft_train_v2.jsonl ] && FILES+=(data/ehrsql/sft_train_v2.jsonl)
[ -f data/mimic_iv_demo.db ] && FILES+=(data/mimic_iv_demo.db)

# exclude caches / pyc
zip -r -q "$OUT" "${FILES[@]}" -x '*/__pycache__/*' '*.pyc'
echo ""
echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
echo "Upload to your 5TB Drive, share folder as EHRSQL_GEMMA to the Colab account,"
echo "then run notebooks/gemma4_finetune_eval_colab.ipynb."
