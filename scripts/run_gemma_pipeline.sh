#!/bin/bash
# Autonomous Gemma 3 12B finetuning pipeline (branch Gemma_dev).
#
#   SFT (QLoRA, 1 epoch) -> eval(subset) -> ORPO pairs -> Abstention-ORPO -> eval(subset) -> eval(FULL)
#
# Why 1 epoch + a 500-question subset for the intermediate evals:
#   On this RTX 3090, 4-bit Gemma-3-12B trains at ~43 s/step and generates at
#   ~5-8 s/question. A 3-epoch SFT + two full 1786-question evals would be ~24 h.
#   1 epoch (~7 h, loss already <0.55 by step 20) plus subset evals for the
#   SFT->ORPO delta, with ONE full-set eval on the final model for a
#   leaderboard-comparable RS(10), keeps the run to an overnight window.
#
# Adapters are evaluated directly via the Unsloth path (no fragile Gemma-3 merge).
# Eval stages are non-fatal; training stages abort on failure. Results are
# written incrementally so a mid-run check still shows progress.
#
# Env overrides: SFT_EPOCHS(1) ORPO_EPOCHS(2) MAX_ANSWERABLE(500) PYBIN(.venv/bin/python)

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

PYBIN="${PYBIN:-.venv/bin/python}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
ORPO_EPOCHS="${ORPO_EPOCHS:-2}"
MAX_ANSWERABLE="${MAX_ANSWERABLE:-500}"

TRAIN=data/ehrsql/ehrsql/mimic_iii/train.json
VALID=data/ehrsql/ehrsql/mimic_iii/valid.json
TEST=data/ehrsql/ehrsql/mimic_iii/test.json
SUBSET=data/ehrsql/ehrsql/mimic_iii/test_subset500.json
SFT_DATA=data/ehrsql/sft_train_v2.jsonl

export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Reduce CUDA fragmentation — ORPO OOM'd at step 2 from fragmentation otherwise.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NVLIB=$($PYBIN -c "import os,glob; base=os.path.join(os.getcwd(),'.venv','lib','python3.13','site-packages','nvidia'); print(':'.join(sorted(glob.glob(os.path.join(base,'*','lib')))))" 2>/dev/null || echo "")
[ -n "$NVLIB" ] && export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}"

mkdir -p logs checkpoints tests/evalgen
PLOG=logs/gemma_pipeline.log
banner() { echo "============================================================"; echo " $*"; echo " $(date)"; echo "============================================================"; }
die()    { echo "PIPELINE ABORTED: $*  ($(date))" | tee -a "$PLOG"; exit 1; }

banner "Gemma 3 12B pipeline START (SFT_EPOCHS=$SFT_EPOCHS ORPO_EPOCHS=$ORPO_EPOCHS)" | tee "$PLOG"
[ -f "$SFT_DATA" ] || die "missing SFT data $SFT_DATA"
[ -f data/mimic_iv_demo.db ] || die "missing DB data/mimic_iv_demo.db"

# ── Stage 1: QLoRA SFT ───────────────────────────────────────────────
banner "[1/6] QLoRA SFT ($SFT_EPOCHS epoch)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.finetune.qlora_sft \
    --data "$SFT_DATA" --output checkpoints/sft --epochs "$SFT_EPOCHS" \
    2>&1 | tee logs/gemma_sft.log
[ -d checkpoints/sft/adapter_final ] || die "SFT produced no adapter"

# ── Stage 2: eval SFT on subset ──────────────────────────────────────
banner "[2/6] Eval SFT on 500-subset (--repair --few-shot)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness "$SUBSET" \
    --model checkpoints/sft/adapter_final \
    --output tests/evalgen/gemma_sft_subset500.json \
    --repair --few-shot "$TRAIN" 2>&1 | tee logs/gemma_sft_eval.log || echo "WARN: SFT subset eval failed" | tee -a "$PLOG"

# ── Stage 3: build ORPO pairs ────────────────────────────────────────
banner "[3/6] Build Abstention-ORPO pairs (max_answerable=$MAX_ANSWERABLE)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.finetune.build_pairs \
    --train "$TRAIN" --valid "$VALID" \
    --adapter checkpoints/sft/adapter_final \
    --output data/ehrsql/dpo_pairs.jsonl \
    --max-answerable "$MAX_ANSWERABLE" --verify-execution \
    2>&1 | tee logs/gemma_pairs.log
[ -s data/ehrsql/dpo_pairs.jsonl ] || die "no ORPO pairs written"

# ── Stage 4: Abstention-ORPO ─────────────────────────────────────────
banner "[4/6] Abstention-ORPO ($ORPO_EPOCHS epochs)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.finetune.abstention_dpo \
    --pairs data/ehrsql/dpo_pairs.jsonl \
    --adapter checkpoints/sft/adapter_final \
    --output checkpoints/orpo --epochs "$ORPO_EPOCHS" --orpo-lambda 0.1 \
    2>&1 | tee logs/gemma_orpo.log
[ -d checkpoints/orpo/adapter_final ] || die "ORPO produced no adapter"

# ── Stage 5: eval ORPO on subset (delta vs SFT) ──────────────────────
banner "[5/6] Eval ORPO on 500-subset (--repair --few-shot)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness "$SUBSET" \
    --model checkpoints/orpo/adapter_final \
    --output tests/evalgen/gemma_orpo_subset500.json \
    --repair --few-shot "$TRAIN" 2>&1 | tee logs/gemma_orpo_eval.log || echo "WARN: ORPO subset eval failed" | tee -a "$PLOG"

# ── Stage 6: eval ORPO on FULL test set (headline RS) ────────────────
banner "[6/6] Eval ORPO on FULL 1786 test set (--repair --few-shot)" | tee -a "$PLOG"
$PYBIN -m ehrcopilot.eval.harness "$TEST" \
    --model checkpoints/orpo/adapter_final \
    --output tests/evalgen/gemma_orpo_repair_rag_results.json \
    --repair --few-shot "$TRAIN" 2>&1 | tee logs/gemma_orpo_full_eval.log || echo "WARN: ORPO full eval failed" | tee -a "$PLOG"

# ── Summary ──────────────────────────────────────────────────────────
banner "Gemma 3 12B pipeline COMPLETE" | tee -a "$PLOG"
$PYBIN - <<'PY' 2>&1 | tee -a "$PLOG"
import json, os
def load(p): return json.load(open(p)) if os.path.exists(p) else {}
rows = [
    ("Gemma SFT  (subset500, repair+RAG)",  load("tests/evalgen/gemma_sft_subset500.json")),
    ("Gemma ORPO (subset500, repair+RAG)",  load("tests/evalgen/gemma_orpo_subset500.json")),
    ("Gemma ORPO (FULL 1786, repair+RAG)",  load("tests/evalgen/gemma_orpo_repair_rag_results.json")),
    ("Qwen ORPO v3 (FULL, repair+RAG) prior", load("tests/evalgen/orpo_v3_repair_rag_results.json")),
]
print(f"{'stage':40s} | {'n':>4} | {'EX':>7} | {'RS(10)':>8} | {'corr_ans':>8} | {'wrong_unans':>11}")
print("-"*95)
for name, d in rows:
    print(f"{name:40s} | {d.get('total',0):4} | {d.get('EX',0):7.4f} | {d.get('RS(10)',0):8.4f} | "
          f"{d.get('correct_answers',0):8} | {d.get('wrong_answers_on_unanswerable',0):11}")
PY
echo "DONE $(date)"
