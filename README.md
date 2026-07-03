# Governed Clinical-Analytics Copilot

**Decision support over de-identified EHR data. Not a medical device. Outputs require clinician review.**

A self-hostable, governance-first natural-language analytics agent over MIMIC-IV-Demo / EHRSQL data. Translates clinical questions into SQL, enforces a 5-layer AST guardrail stack, knows when to refuse, and caches frequent queries with access-role scoping.

---

## What makes this different

| Capability | This project | Commercial tools | Other open repos |
|-----------|-------------|-----------------|-----------------|
| 5-layer AST guardrail stack | ✅ | ❌ | ❌ |
| DPO-trained abstention (`[ABSTAIN]`) | ✅ | ❌ | ❌ |
| Role-scoped semantic cache | ✅ | ❌ | ❌ |
| Self-hostable + published RS metrics | ✅ | ❌ | partial |
| k-anonymity suppression (k=11, NHS standard) | ✅ | ❌ | ❌ |

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
[SQL Writer] → Qwen2.5-Coder-7B (QLoRA SFT + Abstention-DPO)
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
[Reliability Gate] → confidence score → ANSWER or ABSTAIN
     │
     ▼
[Clinical-Safe Summarizer] → plain-English answer + disclaimer
     │
     ▼
[Semantic Cache] → role-scoped ChromaDB → store for future hits
```

## Hardware

| GPU | Role |
|-----|------|
| RTX 4080 Super (16 GB) | Fine-tuning (QLoRA SFT + Abstention-DPO) |
| RTX 5080 (16 GB GDDR7) | Inference serving (vLLM AWQ-4bit) + interactive dev |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/pbiyyani09/Governed-Clinical-Analytics-Copilot
cd Governed-Clinical-Analytics-Copilot
pip install -e ".[eval,agents,dev]"

# 2. Download MIMIC-IV-Demo (free PhysioNet registration required)
#    https://physionet.org/content/mimic-iv-demo/2.2/
#    Unzip into data/mimic-iv-demo/

# 3. Build SQLite DB
bash src/ehrcopilot/db/build_sqlite.sh

# 4. Run guardrail tests (no GPU or data required)
pytest src/ehrcopilot/guardrails/tests/ -v

# 5. Run baseline evaluation (requires GPU + model download)
python -m ehrcopilot.eval.harness data/ehrsql/ehrsql/mimic_iii/test.json \
  --model Qwen/Qwen2.5-Coder-7B-Instruct

# 6. Start the API server (requires AWQ model artifact)
uvicorn ehrcopilot.serve.app:app --reload
```

## Eval Results (fill in after M7)

| Model | EX | RS(0) | RS(5) | RS(10) |
|-------|----|-------|-------|--------|
| Qwen2.5-Coder-7B base | TBD | TBD | TBD | TBD |
| + QLoRA SFT | TBD | TBD | TBD | TBD |
| + Abstention-DPO | TBD | TBD | TBD | TBD |
| EHRSQL 2024 winner (LG AI/KAIST) | — | — | — | 81.32 |

## 5-Layer Guardrail Stack

| Layer | When | What |
|-------|------|------|
| 1 | Pre-execution | Read-only + single-statement enforcement (AST) |
| 2 | Pre-execution | Table + column allowlist (AST) |
| 3 | Pre-execution | PHI-column hard block (AST) |
| 4 | Post-execution | Small-cell suppression k=11 (NHS standard) |
| 5 | Pre-generation | Prompt-injection + exfiltration detection (NL) |

See [docs/guardrail_metrics.md](docs/guardrail_metrics.md) for detection rates.

## Fine-Tuning

```bash
# Step 1: Prepare SFT data
python -m ehrcopilot.finetune.prepare_sft \
  --train data/ehrsql/ehrsql/mimic_iii/train.json \
  --output data/ehrsql/sft_train.jsonl

# Step 2: QLoRA SFT (RTX 4080 Super, ~11-13 GB VRAM)
python -m ehrcopilot.finetune.qlora_sft \
  --data data/ehrsql/sft_train.jsonl \
  --output checkpoints/sft

# Step 3: Build DPO preference pairs
python -m ehrcopilot.finetune.build_pairs \
  --train data/ehrsql/ehrsql/mimic_iii/train.json \
  --adapter checkpoints/sft/adapter_final \
  --output data/ehrsql/dpo_pairs.jsonl

# Step 4: Abstention-DPO
python -m ehrcopilot.finetune.abstention_dpo \
  --pairs data/ehrsql/dpo_pairs.jsonl \
  --adapter checkpoints/sft/adapter_final \
  --output checkpoints/dpo
```

## Docker

```bash
docker compose up
# API: http://localhost:8000
# Phoenix tracing: http://localhost:6006
```

## API

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many female patients were admitted in 2180?", "role": "clinician"}'

# Health
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

## Project Structure

```
src/ehrcopilot/
├── config.py                  # schema allowlist, PHI columns, k threshold, cache threshold
├── db/
│   ├── build_sqlite.sh        # MIMIC-IV-Demo CSV → SQLite
│   ├── build_sqlite.py
│   └── connection.py          # read-only conn, timeout, row cap
├── agents/
│   ├── state.py               # CopilotState TypedDict
│   ├── graph.py               # LangGraph wiring
│   └── nodes/
│       ├── planner.py
│       ├── schema_linker.py
│       ├── sql_writer.py
│       ├── guard_validator.py
│       ├── repair.py
│       ├── executor.py
│       ├── reliability_gate.py
│       └── clinical_safe_summarizer.py
├── guardrails/
│   ├── layers.py              # all 5 layers, GuardResult
│   └── tests/
│       └── adversarial_suite.py
├── cache/
│   └── semantic_cache.py      # role-scoped ChromaDB cache
├── evalgen/
│   └── generator.py           # auto test generator + buckets + metamorphic
├── eval/
│   └── harness.py             # EX + RS(0/5/10) harness
├── finetune/
│   ├── prepare_sft.py
│   ├── qlora_sft.py
│   ├── build_pairs.py         # preference pair construction
│   └── abstention_dpo.py
└── serve/
    └── app.py                 # FastAPI + vLLM
```

## Why agents here

The LangGraph graph makes sense because each node has distinct typed inputs/outputs with
conditional routing: the repair loop (guard failure → re-prompt → re-validate) maps naturally
to graph edges, the guardrail gate between generation and execution is non-negotiable, and
Phoenix tracing gives per-node observability impossible in a monolithic function.

## Disclaimer

This system is designed for decision support over de-identified demo data only.
NOT a medical device. Outputs must be reviewed by a qualified clinician before
clinical use. The MIMIC-IV-Demo dataset is a small, de-identified subset not suitable
for population-level inference.

## Licenses

- Source code: [MIT](LICENSES/MIT.txt)
- EHRSQL dataset: [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)
- MIMIC-IV-Demo: [ODbL v1.0](LICENSES/ODbL-1.0.txt)
