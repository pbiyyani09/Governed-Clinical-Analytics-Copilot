#!/bin/bash
# Gemma 3 12B fine-tuning pipeline (branch gemma_dev_rebased), justified by
# FINETUNING_JUSTIFICATION.md (inference-time techniques exhausted).
#
#   QLoRA SFT (1 epoch) -> eval -> ORPO pairs -> Abstention-ORPO -> eval
#
# Targets the two walls: SFT learns MIMIC-IV SQL patterns (EX), Abstention-ORPO
# calibrates [ABSTAIN] (RS). Adapters evaluated directly via the Unsloth path with
# the classifier-fusion few-shot retriever + repair. Resumable; detached-friendly.
#
# Env overrides: SFT_EPOCHS(1) ORPO_EPOCHS(2) MAX_ANSWERABLE(500)

set -uo pipefail
cd "$(dirname "$0")/.."

PYBIN=.venv/bin/python
SFT_EPOCHS="${SFT_EPOCHS:-1}"
ORPO_EPOCHS="${ORPO_EPOCHS:-2}"
MAX_ANSWERABLE="${MAX_ANSWERABLE:-500}"
BASE=unsloth/gemma-3-12b-it
TRAIN=data/ehrsql/ehrsql/mimic_iii/train.json
VALID=data/ehrsql/ehrsql/mimic_iii/valid.json
SUBSET=data/ehrsql/ehrsql/mimic_iii/test_cmp75.json
SFT_DATA=data/ehrsql/sft_train_v2.jsonl

export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(grep -E "^HF_TOKEN" .env 2>/dev/null | sed -E 's/.*=//' | tr -d '" ')
NVLIB=$($PYBIN -c "import os,glob;b=os.path.join(os.getcwd(),'.venv','lib','python3.13','site-packages','nvidia');print(':'.join(sorted(glob.glob(os.path.join(b,'*','lib')))))" 2>/dev/null || echo "")
[ -n "$NVLIB" ] && export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}"
mkdir -p logs checkpoints tests/evalgen
PLOG=logs/gemma_finetune.log
banner(){ echo "============================================================"; echo " $*"; echo " $(date)"; echo "============================================================"; }
die(){ echo "ABORT: $* ($(date))" | tee -a "$PLOG"; exit 1; }

banner "Gemma 3 12B fine-tune START (SFT_EPOCHS=$SFT_EPOCHS ORPO_EPOCHS=$ORPO_EPOCHS)" | tee "$PLOG"
[ -f "$SFT_DATA" ] || die "missing $SFT_DATA"
[ -f data/mimic_iv_demo.db ] || die "missing DB"

# Stage 1 — SFT
banner "[1/5] QLoRA SFT" | tee -a "$PLOG"
if [ -d checkpoints/sft_gemma/adapter_final ]; then
  echo "SFT adapter exists — skip" | tee -a "$PLOG"
else
  $PYBIN -m ehrcopilot.finetune.qlora_sft --base-model "$BASE" \
      --data "$SFT_DATA" --output checkpoints/sft_gemma --epochs "$SFT_EPOCHS" \
      2>&1 | tee logs/gemma_sft.log
  [ -d checkpoints/sft_gemma/adapter_final ] || die "SFT produced no adapter"
fi

# Stage 2 — eval SFT (classifier retriever + repair) on subset
banner "[2/5] Eval SFT (classifier + repair)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness "$SUBSET" --model checkpoints/sft_gemma/adapter_final \
    --few-shot "$TRAIN" --retrieval-mode classifier --repair \
    --output tests/evalgen/gemma_sft_ex.json 2>&1 | tail -4 | tee -a "$PLOG" || echo "WARN sft eval" | tee -a "$PLOG"

# Stage 3 — ORPO pairs
banner "[3/5] Build Abstention-ORPO pairs" | tee -a "$PLOG"
if [ -s data/ehrsql/gemma_orpo_pairs.jsonl ]; then
  echo "pairs exist — skip" | tee -a "$PLOG"
else
  $PYBIN -m ehrcopilot.finetune.build_pairs --train "$TRAIN" --valid "$VALID" \
      --adapter checkpoints/sft_gemma/adapter_final --output data/ehrsql/gemma_orpo_pairs.jsonl \
      --max-answerable "$MAX_ANSWERABLE" --verify-execution 2>&1 | tee logs/gemma_pairs.log
  [ -s data/ehrsql/gemma_orpo_pairs.jsonl ] || die "no pairs"
fi

# Stage 4 — Abstention-ORPO
banner "[4/5] Abstention-ORPO" | tee -a "$PLOG"
if [ -d checkpoints/orpo_gemma/adapter_final ]; then
  echo "ORPO adapter exists — skip" | tee -a "$PLOG"
else
  $PYBIN -m ehrcopilot.finetune.abstention_dpo --pairs data/ehrsql/gemma_orpo_pairs.jsonl \
      --adapter checkpoints/sft_gemma/adapter_final --output checkpoints/orpo_gemma \
      --epochs "$ORPO_EPOCHS" 2>&1 | tee logs/gemma_orpo.log
  [ -d checkpoints/orpo_gemma/adapter_final ] || die "ORPO produced no adapter"
fi

# Stage 5 — eval ORPO (subset + full)
banner "[5/5] Eval ORPO (classifier + repair)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness "$SUBSET" --model checkpoints/orpo_gemma/adapter_final \
    --few-shot "$TRAIN" --retrieval-mode classifier --repair \
    --output tests/evalgen/gemma_orpo_ex_subset.json 2>&1 | tail -4 | tee -a "$PLOG" || echo "WARN orpo subset eval" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness data/ehrsql/ehrsql/mimic_iii/test.json \
    --model checkpoints/orpo_gemma/adapter_final \
    --few-shot "$TRAIN" --retrieval-mode classifier --repair \
    --output tests/evalgen/gemma_orpo_ex_full.json 2>&1 | tail -4 | tee -a "$PLOG" || echo "WARN orpo full eval" | tee -a "$PLOG"

banner "Gemma fine-tune COMPLETE" | tee -a "$PLOG"
$PYBIN - <<'PY' 2>&1 | tee -a "$PLOG"
import json,os
for n,f in [("base (no FT, prompting-best)","prompting_experiments.json"),
            ("Gemma SFT","gemma_sft_ex.json"),
            ("Gemma SFT+ORPO subset","gemma_orpo_ex_subset.json"),
            ("Gemma SFT+ORPO FULL","gemma_orpo_ex_full.json")]:
    p=f"tests/evalgen/{f}"
    if os.path.exists(p):
        d=json.load(open(p))
        if "EX" in d: print(f"{n:30s} EX={d['EX']:.4f} RS(10)={d.get('RS(10)',0):+.4f}")
PY
echo "DONE $(date)"
