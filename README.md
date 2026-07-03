# Governed Clinical Analytics Copilot

**Decision support over de-identified EHR data. Not a medical device. Outputs require clinician review.**

A self-hostable, governance-first natural-language analytics agent over MIMIC-IV / EHRSQL data.
Translates clinical questions into SQL, enforces a 5-layer AST guardrail stack, knows when to
refuse, and caches frequent queries with access-role scoping.

> **Result:** RS(10) = **0.873** on EHRSQL 2024 MIMIC-IV (1,167 questions) — surpassing the
> leaderboard winner (0.813, LG AI/KAIST with GPT-3.5) using a fully open 7B model.

📄 [Project Report (PDF)](docs/project_report.pdf)

---

## Benchmark Results

| Model | EX | Hallucinations | Correct Abstentions | RS(10) |
|---|---|---|---|---|
| **OmniSQL-7B + SFT (ours)** | **92.4%** | **7 / 233** | **226 / 233** | **0.873** |
| EHRSQL 2024 Winner (LG AI/KAIST, GPT-3.5) | ~75% | — | — | 0.813 |
| EHRSQL 2024 3rd Place (ProbGate, GPT-3.5) | — | — | — | 0.742 |
| OmniSQL-7B zero-shot baseline | 61.2% | 133 / 233 | 100 / 233 | −0.564 |

**Metric:** RS(10) = (correct SQL + correct abstentions − 10 × hallucinations) / total questions.
Hallucinating on an unanswerable question is penalised 10× more than a missed abstention.

---

## What Makes This Different

| Capability | This project | Commercial tools | Other open repos |
|---|---|---|---|
| Open 7B model, no API dependency | ✅ | ❌ | partial |
| SFT-trained abstention (`[ABSTAIN]`) | ✅ | ❌ | ❌ |
| 5-layer AST guardrail stack | ✅ | ❌ | ❌ |
| Role-scoped semantic cache | ✅ | ❌ | ❌ |
| Self-hostable + published RS metrics | ✅ | ❌ | partial |
| k-anonymity suppression (k=11, NHS standard) | ✅ | ❌ | ❌ |

---

## Architecture

```
User question
     │
     ▼
[Layer 5: Prompt-injection detection]  ← NL guard before SQL generation
     │
     ▼
[Planner] → intent + answerability check
     │
     ▼
[Schema Linker] → BM25 retrieval → minimal relevant schema subset
     │
     ▼
[SQL Writer] → OmniSQL-7B (SFT + LoRA adapter)
     │
     ▼
[Layer 1: Read-only / single-statement]  ─┐
[Layer 2: Table + column allowlist]       ├── pre-execution AST checks
[Layer 3: PHI-column hard block]          ─┘
     │
     ▼
[Executor] → read-only SQLite (10s timeout, 1000-row cap)
     │
     ▼
[Layer 4: Small-cell suppression (k=11)]  ← post-execution
     │
     ▼
[Reliability Gate] → entropy threshold → ANSWER or [ABSTAIN]
     │
     ▼
[Clinical-Safe Summarizer] → plain-English answer + disclaimer
     │
     ▼
[Semantic Cache] → role-scoped ChromaDB → store for future hits
```

---

## Hardware

| GPU | Role |
|---|---|
| RTX 4080 Super (16 GB) | Local inference, evaluation, data prep |
| Google Colab RTX 6000 Pro (96 GB) | SFT training (2 epochs, ~5 hrs) |

---

## Fine-Tuning Pipeline

### Step 1 — Build SFT data

```bash
python scripts/build_omnisql_sft_data.py
# Output: data/sft/omnisql_sft_train.jsonl
# 53,006 examples: 40,961 answerable SQL + 12,045 [ABSTAIN]
# 3× unanswerable oversampling to match test-set distribution
```

### Step 2 — SFT training (Google Colab)

Upload `data/sft/omnisql_sft_train.jsonl` and `data/ehrsql2024/mimic_iv/mimic_iv.sqlite`
to Drive, then run `colab/OmniSQL_SFT_Train.ipynb`.

| Parameter | Value |
|---|---|
| Base model | `seeklhy/OmniSQL-7B` |
| LoRA rank | r=32, α=64 |
| Target modules | q, k, v, o, gate, up, down (7 modules) |
| Epochs | 2 |
| Effective batch | 16 (2 per device × 8 grad accum) |
| Learning rate | 1e-4, cosine schedule |
| Max seq length | 1,536 tokens |

### Step 3 — Build ORPO pairs (optional)

```bash
python scripts/build_omnisql_orpo_pairs.py
# Extracts 492 preference pairs from baseline predictions
```

### Step 4 — ORPO training (optional, Colab)

Run `colab/OmniSQL_ORPO_Train.ipynb` with the SFT adapter as base.
Note: SFT-only (RS=0.873) outperformed SFT+ORPO on this dataset due to
asymmetric gradient dynamics — ORPO is provided for experimentation.

---

## Evaluation

### Quick local eval (100 questions, ~6 min)

```bash
bash scripts/run_quick_eval.sh \
    --adapter path/to/adapter
# Stratified 80 answerable + 20 unanswerable (SEED=42)
# Outputs: EX, RS(10), hallucinations, correct abstentions
```

### Full test-set eval (1,167 questions)

```bash
bash scripts/run_sft_eval.sh \
    --adapter path/to/adapter \
    --output tests/evalgen/results.json \
    --save-predictions tests/evalgen/preds.jsonl
```

Optional flags:
- `--entropy-threshold FLOAT` — convert high-entropy predictions to [ABSTAIN]
- `--abstain-on-error` — treat SQL execution errors as abstentions
- `--repair` — enable execution-guided SQL repair loop (up to 3 retries)
- `--few-shot data/ehrsql2024/mimic_iv/train` — enable RAG few-shot retrieval

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/pbiyyani09/Governed-Clinical-Analytics-Copilot
cd Governed-Clinical-Analytics-Copilot
pip install -e ".[eval,agents,dev]"

# 2. Obtain MIMIC-IV Clinical Database Demo (free PhysioNet account required)
#    https://physionet.org/content/mimic-iv-demo/
#    Place under data/ehrsql2024/mimic_iv/

# 3. Run guardrail tests (no GPU or data required)
pytest tests/ -v

# 4. Run quick eval (requires GPU + downloaded adapter)
bash scripts/run_quick_eval.sh --adapter path/to/omnisql_sft_adapter
```

---

## Project Structure

```
src/ehrcopilot/
├── config.py                  # SYSTEM_PROMPT, schema allowlist, PHI columns, k threshold
├── db/
│   └── connection.py          # read-only SQLite conn, timeout, row cap
├── eval/
│   ├── harness.py             # EX + RS(0/5/10) harness, self-consistency, entropy filter
│   └── rag_eval.py            # BM25 + embedding hybrid retrieval, reranking
├── finetune/
│   ├── prepare_sft.py         # SFT data preparation utilities
│   ├── qlora_sft.py           # QLoRA SFT trainer
│   ├── build_pairs.py         # ORPO preference pair construction
│   └── abstention_dpo.py      # ORPO/DPO trainer with abstention support
└── finetune/grpo_train.py     # GRPO trainer (experimental)

scripts/
├── build_omnisql_sft_data.py  # Build 53K SFT examples with unanswerable oversampling
├── build_omnisql_orpo_pairs.py# Build ORPO preference pairs from predictions
├── quick_eval_adapter.py      # 100-question local eval (80 ans + 20 unans)
├── run_quick_eval.sh          # Wrapper with correct LD_LIBRARY_PATH for bitsandbytes
├── run_sft_eval.sh            # Full test-set eval driver
└── export_submission.py       # EHRSQL submission format exporter

colab/
├── OmniSQL_SFT_Train.ipynb    # Session 1 — SFT training on RTX 6000 Pro
├── OmniSQL_ORPO_Train.ipynb   # Session 2 — ORPO training + full eval
├── OmniSQL_EHRSQL_Eval.ipynb  # Standalone eval notebook (adapter-agnostic)
└── system_prompt.json         # Full 2,733-char schema-inclusive system prompt

docs/
└── project_report.pdf         # Full project report with results and lessons learned
```

---

## 5-Layer Guardrail Stack

| Layer | When | What |
|---|---|---|
| 1 | Pre-execution | Read-only + single-statement enforcement (AST) |
| 2 | Pre-execution | Table + column allowlist (AST) |
| 3 | Pre-execution | PHI-column hard block (AST) |
| 4 | Post-execution | Small-cell suppression k=11 (NHS standard) |
| 5 | Pre-generation | Prompt-injection + exfiltration detection (NL) |

---

## Data

This project uses the **MIMIC-IV Clinical Database Demo** — a freely accessible 94-patient subset
of MIMIC-IV. It requires only a standard (non-credentialed) PhysioNet account and agreement to the
data use terms, with no training course or IRB process.

Data files are **not included in this repository** per the PhysioNet DUA. Download the demo at
[physionet.org/content/mimic-iv-demo](https://physionet.org/content/mimic-iv-demo/) and place it
under `data/ehrsql2024/mimic_iv/`. The `.gitignore` permanently blocks all paths under
`data/ehrsql2024/` and `data/pairs/` from being committed.

---

## Disclaimer

This system is designed for decision support over de-identified data only. **NOT a medical device.**
Outputs must be reviewed by a qualified clinician before any clinical use.

---

## Licenses

- Source code: [MIT](LICENSES/MIT.txt)
- EHRSQL dataset: [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)
- MIMIC-IV Demo: [PhysioNet Data Use Agreement](https://physionet.org/content/mimic-iv-demo/) (standard account, no credentials required)
