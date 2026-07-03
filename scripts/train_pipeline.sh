#!/bin/bash
# Full training pipeline: data prep → SFT → ORPO pairs → Abstention-ORPO → merge/export
# Run after baseline eval completes.
#
# Usage:
#   bash scripts/train_pipeline.sh [--skip-prep] [--skip-sft] [--skip-sft-eval] [--skip-dpo] [--gguf]
#
# Outputs:
#   data/ehrsql/sft_train_v2.jsonl  — Cleaned SFT training data (MIMIC-IV-compatible)
#   checkpoints/sft/adapter_final   — QLoRA SFT adapter
#   data/ehrsql/dpo_pairs.jsonl     — Abstention-ORPO preference pairs
#   checkpoints/dpo/adapter_final   — Abstention-ORPO adapter
#   models/merged/                  — Merged bf16 model (for vLLM)
#   models/gguf/                    — GGUF Q4_K_M (if --gguf is passed)

set -euo pipefail

SKIP_PREP=0
SKIP_SFT=0
SKIP_SFT_EVAL=0
SKIP_DPO=0
GGUF_FLAG=""

for arg in "$@"; do
    case $arg in
        --skip-prep)     SKIP_PREP=1 ;;
        --skip-sft)      SKIP_SFT=1 ;;
        --skip-sft-eval) SKIP_SFT_EVAL=1 ;;
        --skip-dpo)      SKIP_DPO=1 ;;
        --gguf)          GGUF_FLAG="--gguf" ;;
    esac
done

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# bitsandbytes needs libnvJitLink.so.13 from the bundled nvidia-cu13 package
_NV_CU13=$(python3 -c "import os,sys; print(os.path.join(os.path.dirname(sys.modules['os'].__file__), 'site-packages','nvidia','cu13','lib'))" 2>/dev/null)
if [ -d "$_NV_CU13" ]; then
  export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
fi

mkdir -p logs checkpoints/sft checkpoints/dpo models

echo "============================================================"
echo " Governed Clinical-Analytics Copilot — Training Pipeline v2"
echo " $(date)"
echo "============================================================"

# ── Step 0: Prepare SFT training data ────────────────────────────
if [ "$SKIP_PREP" -eq 0 ]; then
    echo ""
    echo "[ STEP 0 ] Preparing SFT training data (MIMIC-IV-compatible)..."
    python3 -m ehrcopilot.finetune.prepare_sft \
        --train data/ehrsql2024/mimic_iv/train \
        --valid data/ehrsql2024/mimic_iv/valid \
        --output data/ehrsql/sft_train_v2.jsonl \
        2>&1 | tee logs/prepare_sft.log
    echo "[ STEP 0 ] Data prep complete → data/ehrsql/sft_train_v2.jsonl"
else
    echo "[ STEP 0 ] Skipping data prep (--skip-prep)"
fi

# ── Step 1: QLoRA SFT ────────────────────────────────────────────
if [ "$SKIP_SFT" -eq 0 ]; then
    echo ""
    echo "[ STEP 1 ] QLoRA SFT fine-tuning..."
    python3 -m ehrcopilot.finetune.qlora_sft \
        --data data/ehrsql/sft_train_v2.jsonl \
        --output checkpoints/sft \
        --epochs 3 \
        2>&1 | tee logs/sft_train.log
    echo "[ STEP 1 ] SFT complete → checkpoints/sft/adapter_final"
else
    echo "[ STEP 1 ] Skipping SFT (--skip-sft)"
fi

# ── Step 1.5: Post-SFT evaluation ────────────────────────────────
if [ "$SKIP_SFT_EVAL" -eq 0 ]; then
    echo ""
    echo "[ STEP 1.5 ] Running post-SFT evaluation..."
    bash scripts/run_sft_eval.sh 2>&1 | tee logs/sft_eval.log
    echo "[ STEP 1.5 ] SFT eval complete → tests/evalgen/sft_results.json"
else
    echo "[ STEP 1.5 ] Skipping SFT eval (--skip-sft-eval)"
fi

# ── Step 2: Build DPO preference pairs ───────────────────────────
if [ "$SKIP_DPO" -eq 0 ]; then
    echo ""
    echo "[ STEP 2 ] Building Abstention-DPO preference pairs..."
    python3 -m ehrcopilot.finetune.build_pairs \
        --train data/ehrsql2024/mimic_iv/train \
        --valid data/ehrsql2024/mimic_iv/valid \
        --adapter checkpoints/sft/adapter_final \
        --output data/ehrsql/dpo_pairs.jsonl \
        --max-answerable 500 \
        2>&1 | tee logs/dpo_pairs.log
    echo "[ STEP 2 ] Pairs written → data/ehrsql/dpo_pairs.jsonl"

    # ── Step 3: Abstention-DPO training ──────────────────────────
    echo ""
    echo "[ STEP 3 ] Abstention-ORPO fine-tuning..."
    python3 -m ehrcopilot.finetune.abstention_dpo \
        --pairs data/ehrsql/dpo_pairs.jsonl \
        --adapter checkpoints/sft/adapter_final \
        --output checkpoints/dpo \
        --epochs 2 \
        --orpo-lambda 0.1 \
        2>&1 | tee logs/dpo_train.log
    echo "[ STEP 3 ] ORPO complete → checkpoints/dpo/adapter_final"
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
echo "[ STEP 5 ] Running post-DPO evaluation on EHRSQL test set..."
python3 -m ehrcopilot.eval.harness \
    data/ehrsql2024/mimic_iv/test \
    --model models/merged \
    --output tests/evalgen/dpo_results.json \
    2>&1 | tee logs/dpo_eval.log
echo "[ STEP 5 ] Eval complete → tests/evalgen/dpo_results.json"

echo ""
echo "============================================================"
echo " Pipeline complete! $(date)"
echo "============================================================"

# Print 3-way comparison: baseline / post-SFT / post-DPO
if [ -f tests/evalgen/baselines.json ]; then
    echo ""
    echo "Results comparison:"
    python3 -c "
import json, os

def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}

b = load('tests/evalgen/baselines.json')
s = load('tests/evalgen/sft_results.json')
d = load('tests/evalgen/dpo_results.json')

print(f'  Metric      | Baseline | Post-SFT | Post-DPO | Δ vs base')
print(f'  ------------|----------|----------|----------|----------')
for k in ['EX', 'RS(0)', 'RS(5)', 'RS(10)']:
    bv = b.get(k, 0); sv = s.get(k, 0); dv = d.get(k, 0)
    delta = dv - bv
    sign = '+' if delta >= 0 else ''
    sv_str = f'{sv:8.4f}' if sv else '    n/a '
    dv_str = f'{dv:8.4f}' if dv else '    n/a '
    print(f'  {k:11s} | {bv:8.4f} | {sv_str} | {dv_str} | {sign}{delta:.4f}')
"
fi
