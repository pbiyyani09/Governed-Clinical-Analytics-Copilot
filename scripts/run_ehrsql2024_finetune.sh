#!/bin/bash
# EHRSQL-2024 (MIMIC-IV) fine-tune + eval pipeline — the apples-to-apples benchmark
# (leaderboard RS(10)=0.8132). Mirrors the Colab notebook; resumable; detached-friendly.
#
#   prepare SFT (augmented) -> QLoRA SFT -> ORPO pairs -> Abstention-ORPO -> eval2024 (official RS)
#
# Env overrides: MODEL SFT_EPOCHS(1) ORPO_EPOCHS(2) MAX_ANSWERABLE(1500) BS GA
set -uo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-.venv/bin/python}"
MODEL="${MODEL:-unsloth/gemma-4-12b-it}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"          # 44k augmented examples → 1 epoch ≈ old 8 epochs of updates
ORPO_EPOCHS="${ORPO_EPOCHS:-2}"
MAX_ANSWERABLE="${MAX_ANSWERABLE:-1500}"
BS="${BS:-1}"; GA="${GA:-16}"

D=data/ehrsql2024/mimic_iv
SFT_DATA=data/ehrsql2024/sft_train_aug.jsonl
SFT_OUT=checkpoints/sft_g4_2024
ORPO_OUT=checkpoints/orpo_g4_2024
PAIRS=data/ehrsql2024/orpo_pairs.jsonl

export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(grep -E "^HF_TOKEN" .env 2>/dev/null | sed -E 's/.*=//' | tr -d '" ')
mkdir -p logs checkpoints tests/evalgen
banner(){ echo "============================================================"; echo " $*  ($(date))"; echo "============================================================"; }
die(){ echo "ABORT: $* ($(date))"; exit 1; }

[ -f "$D/mimic_iv.sqlite" ] || die "missing $D/mimic_iv.sqlite"

banner "[0/5] Prepare augmented SFT data"
if [ -s "$SFT_DATA" ]; then echo "SFT data exists — skip"; else
  [ -d "$D/train_aug" ] || $PYBIN -m ehrcopilot.finetune.augment_ehrsql2024 \
      --target-answerable 40000 --synthetic-unanswerable 3000
  $PYBIN -m ehrcopilot.finetune.prepare_sft --train "$D/train_aug" --valid "$D/valid" --output "$SFT_DATA" \
      || die "prepare_sft failed"
fi

banner "[1/5] QLoRA SFT ($MODEL, ${SFT_EPOCHS}ep)"
if [ -d "$SFT_OUT/adapter_final" ]; then echo "SFT adapter exists — skip"; else
  $PYBIN -m ehrcopilot.finetune.qlora_sft --base-model "$MODEL" --data "$SFT_DATA" \
      --output "$SFT_OUT" --epochs "$SFT_EPOCHS" --batch-size "$BS" --grad-accum "$GA" \
      2>&1 | tee logs/sft_2024.log
  [ -d "$SFT_OUT/adapter_final" ] || die "SFT produced no adapter"
fi

banner "[2/5] Build Abstention-ORPO pairs"
if [ -s "$PAIRS" ]; then echo "pairs exist — skip"; else
  $PYBIN -m ehrcopilot.finetune.build_pairs --train "$D/train_aug" --valid "$D/valid" \
      --adapter "$SFT_OUT/adapter_final" --output "$PAIRS" \
      --max-answerable "$MAX_ANSWERABLE" --verify-execution 2>&1 | tee logs/pairs_2024.log
  [ -s "$PAIRS" ] || die "no pairs"
fi

banner "[3/5] Abstention-ORPO (${ORPO_EPOCHS}ep)"
if [ -d "$ORPO_OUT/adapter_final" ]; then echo "ORPO adapter exists — skip"; else
  $PYBIN -m ehrcopilot.finetune.abstention_dpo --pairs "$PAIRS" --adapter "$SFT_OUT/adapter_final" \
      --output "$ORPO_OUT" --epochs "$ORPO_EPOCHS" --max-length 1536 2>&1 | tee logs/orpo_2024.log
  [ -d "$ORPO_OUT/adapter_final" ] || die "ORPO produced no adapter"
fi

banner "[4/5] Eval SFT-only (official RS)"
$PYBIN -m ehrcopilot.eval.eval2024 "$D/test" --model "$SFT_OUT/adapter_final" \
    --few-shot "$D/train" --retrieval-mode embed --repair \
    --output tests/evalgen/g4_2024_sft.json 2>&1 | tail -16

banner "[5/5] Eval SFT+ORPO (official RS)"
$PYBIN -m ehrcopilot.eval.eval2024 "$D/test" --model "$ORPO_OUT/adapter_final" \
    --few-shot "$D/train" --retrieval-mode embed --repair \
    --output tests/evalgen/g4_2024_orpo.json 2>&1 | tail -16

banner "EHRSQL-2024 pipeline COMPLETE"
$PYBIN - <<'PY'
import json, os
for n,f in [("SFT","g4_2024_sft.json"),("SFT+ORPO","g4_2024_orpo.json")]:
    p=f"tests/evalgen/{f}"
    if os.path.exists(p):
        d=json.load(open(p)); print(f"{n:10s} EX={d['EX']:.4f} RS(0)={d['RS(0)']:.2f} RS(10)={d['RS(10)']:.2f}  (target RS(10)=81.32)")
PY
