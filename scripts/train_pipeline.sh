#!/bin/bash
# Full training pipeline: SFT → DPO pairs → Abstention-DPO → merge/export
# Run after baseline eval completes.
#
# Usage:
#   bash scripts/train_pipeline.sh [--skip-sft] [--skip-dpo] [--gguf]
#
# Outputs:
#   checkpoints/sft/adapter_final   — QLoRA SFT adapter
#   data/ehrsql/dpo_pairs.jsonl     — Abstention-DPO preference pairs
#   checkpoints/dpo/adapter_final   — Abstention-DPO adapter
#   models/merged/                  — Merged bf16 model (for vLLM)
#   models/gguf/                    — GGUF Q4_K_M (if --gguf is passed)

set -euo pipefail

SKIP_SFT=0
SKIP_DPO=0
GGUF_FLAG=""

for arg in "$@"; do
    case $arg in
        --skip-sft) SKIP_SFT=1 ;;
        --skip-dpo) SKIP_DPO=1 ;;
        --gguf) GGUF_FLAG="--gguf" ;;
    esac
done

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs checkpoints/sft checkpoints/dpo models

echo "============================================================"
echo " Governed Clinical-Analytics Copilot — Training Pipeline"
echo " $(date)"
echo "============================================================"

# ── Step 1: QLoRA SFT ────────────────────────────────────────────
if [ "$SKIP_SFT" -eq 0 ]; then
    echo ""
    echo "[ STEP 1 ] QLoRA SFT fine-tuning..."
    python3 -m ehrcopilot.finetune.qlora_sft \
        --data data/ehrsql/sft_train.jsonl \
        --output checkpoints/sft \
        --epochs 3 \
        2>&1 | tee logs/sft_train.log
    echo "[ STEP 1 ] SFT complete → checkpoints/sft/adapter_final"
else
    echo "[ STEP 1 ] Skipping SFT (--skip-sft)"
fi

# ── Step 2: Build DPO preference pairs ───────────────────────────
if [ "$SKIP_DPO" -eq 0 ]; then
    echo ""
    echo "[ STEP 2 ] Building Abstention-DPO preference pairs..."
    python3 -m ehrcopilot.finetune.build_pairs \
        --train data/ehrsql/ehrsql/mimic_iii/train.json \
        --adapter checkpoints/sft/adapter_final \
        --output data/ehrsql/dpo_pairs.jsonl \
        --n-candidates 8 \
        2>&1 | tee logs/dpo_pairs.log
    echo "[ STEP 2 ] Pairs written → data/ehrsql/dpo_pairs.jsonl"

    # ── Step 3: Abstention-DPO training ──────────────────────────
    echo ""
    echo "[ STEP 3 ] Abstention-DPO fine-tuning..."
    python3 -m ehrcopilot.finetune.abstention_dpo \
        --pairs data/ehrsql/dpo_pairs.jsonl \
        --adapter checkpoints/sft/adapter_final \
        --output checkpoints/dpo \
        --epochs 2 \
        --beta 0.1 \
        2>&1 | tee logs/dpo_train.log
    echo "[ STEP 3 ] DPO complete → checkpoints/dpo/adapter_final"
else
    echo "[ STEP 2/3 ] Skipping DPO (--skip-dpo)"
fi

# ── Step 4: Merge adapter + export ───────────────────────────────
echo ""
echo "[ STEP 4 ] Merging adapter and exporting model..."
python3 -m ehrcopilot.serve.quantize \
    --adapter checkpoints/dpo/adapter_final \
    --output models \
    $GGUF_FLAG \
    2>&1 | tee logs/quantize.log
echo "[ STEP 4 ] Model exported → models/merged/"

# ── Step 5: Post-training eval ───────────────────────────────────
echo ""
echo "[ STEP 5 ] Running post-SFT+DPO evaluation on EHRSQL test set..."
python3 -m ehrcopilot.eval.harness \
    data/ehrsql/ehrsql/mimic_iii/test.json \
    --output tests/evalgen/dpo_results.json \
    2>&1 | tee logs/dpo_eval.log
echo "[ STEP 5 ] Eval complete → tests/evalgen/dpo_results.json"

echo ""
echo "============================================================"
echo " Pipeline complete! $(date)"
echo "============================================================"

# Print comparison if baseline results exist
if [ -f tests/evalgen/baseline_results.json ] && [ -f tests/evalgen/dpo_results.json ]; then
    echo ""
    echo "Results comparison:"
    python3 -c "
import json
b = json.load(open('tests/evalgen/baseline_results.json'))
d = json.load(open('tests/evalgen/dpo_results.json'))
print(f'  Metric      | Baseline | Post-DPO | Delta')
print(f'  ------------|----------|----------|------')
for k in ['EX', 'RS(0)', 'RS(5)', 'RS(10)']:
    bv = b.get(k, 0); dv = d.get(k, 0)
    delta = dv - bv
    sign = '+' if delta >= 0 else ''
    print(f'  {k:11s} | {bv:8.4f} | {dv:8.4f} | {sign}{delta:.4f}')
"
fi
