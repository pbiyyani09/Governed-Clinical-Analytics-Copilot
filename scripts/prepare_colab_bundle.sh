#!/bin/bash
# Package code + EHRSQL-2024 (MIMIC-IV) data into a ZIP for the Colab fine-tune+eval notebook.
#
# Drive flow (unchanged from before):
#   1. This builds ehrsql_gemma_bundle.zip (code + 2024 data + augmented train + sqlite).
#   2. Upload it to your MAIN (5 TB) Google Drive folder shared as "EHRSQL_GEMMA".
#   3. The Colab account's Drive folder "ehrsql/" holds a SHORTCUT to EHRSQL_GEMMA.
#   4. notebooks/gemma4_finetune_eval_colab.ipynb reads
#      /content/drive/MyDrive/ehrsql/EHRSQL_GEMMA/ehrsql_gemma_bundle.zip,
#      unzips to local Colab disk, trains + evaluates, writes adapters + RS back.
#
# We ship train_aug/{data.json,label.json} (NOT the big annotated.json) and regenerate
# the ~137 MB SFT JSONL inside Colab (prepare_sft, ~30 s) to keep the bundle small.
#
# Usage: bash scripts/prepare_colab_bundle.sh

set -euo pipefail
cd "$(dirname "$0")/.."
OUT=ehrsql_gemma_bundle.zip
rm -f "$OUT"

D=data/ehrsql2024/mimic_iv
[ -f "$D/mimic_iv.sqlite" ] || { echo "ERROR: $D/mimic_iv.sqlite missing (vendor the ehrsql-2024 repo)"; exit 1; }
[ -f "$D/train_aug/label.json" ] || { echo "ERROR: $D/train_aug missing — run: python -m ehrcopilot.finetune.augment_ehrsql2024"; exit 1; }

FILES=(
  src pyproject.toml
  scripts/run_ehrsql2024_finetune.sh
  "$D/mimic_iv.sqlite" "$D/tables.json"
  "$D/train_aug/data.json" "$D/train_aug/label.json"   # augmented SFT/ORPO source
  "$D/train/data.json"     "$D/train/label.json"        # official train (few-shot index)
  "$D/valid/data.json"     "$D/valid/label.json"        # ORPO unanswerable source
  "$D/test/data.json"      "$D/test/label.json"         # eval target (RS)
)
zip -r -q "$OUT" "${FILES[@]}" -x '*/__pycache__/*' '*.pyc'
echo ""
echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
echo "Upload to your 5TB Drive (folder shared as EHRSQL_GEMMA), then run"
echo "notebooks/gemma4_finetune_eval_colab.ipynb."
