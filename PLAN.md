# Active Project Plan

**Goal:** RS(10) ≥ 0.813 (match EHRSQL 2024 leaderboard top score)  
**Current best:** RS(10) = 0.5879 (ORPO v3 + Repair + BM25 RAG)  
**Model:** Qwen2.5-Coder-7B-Instruct, QLoRA 4-bit NF4, RTX 5080 16 GB GDDR7

---

## What's Running Now

### ORPO v4 Training (active, ~21 hrs remaining from 2026-06-26 12:00 EDT)

- **Command:** `bash scripts/run_orpo_v4.sh`
- **Progress:** ~30/327 steps (9%)
- **ETA:** 2026-06-27 ~07:00 EDT
- **Pairs:** 5,218 total (4,856 answerable + 362 unanswerable)
- **Why:** ORPO v3 used only 503/9,318 training questions (5.4% coverage). ORPO v4 covers all questions where the model still fails.

**After training completes → eval:**
```bash
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v4/adapter_final \
  --output tests/evalgen/orpo_v4_results.json \
  --repair --few-shot
```

---

## Parallel Work (RAG Overhaul)

The BM25 retrieval was failing on the hard test cluster [800–1175] for two reasons:

1. **SQL truncation at 120 chars** — Fixed. Full SQL now included in retrieved examples.
2. **BM25 keyword mismatch** — Fixed. Hybrid BM25 + semantic embedding retrieval implemented.

### What Was Built

| File | What it does |
|---|---|
| `src/ehrcopilot/eval/rag_eval.py` | Standalone retrieval quality evaluator (RAGAS-style). Measures Context Recall@K, Precision@K, MRR using `tag` field as relevance signal. |
| `src/ehrcopilot/eval/harness.py` | Rewritten `build_few_shot_retriever()`: supports `--retrieval-mode bm25/embed/hybrid`. Uses RRF fusion + SQL skeleton indexing. No more 120-char SQL truncation. |
| `scripts/eval_retrieval.sh` | CLI wrapper for rag_eval.py. Run `bash scripts/eval_retrieval.sh all` to compare all three modes. |
| `scripts/run_sft_eval.sh` | Updated: now accepts `--retrieval-mode` flag (default: bm25). |

### BM25 Baseline (measured 2026-06-26)

| K | Recall@K | Precision@K |
|---|---|---|
| 1 | 0.2654 | 0.2654 |
| 2 | 0.3456 | 0.2250 |
| 5 | 0.4699 | 0.1731 |
| 10 | 0.5684 | 0.1379 |
| **MRR** | **0.3517** | |
| **Hard cluster Hit@2** | **0.237** | (vs overall 0.346) |

### Hybrid Retrieval Design

- **Embedding model:** BAAI/bge-large-en-v1.5 (335 MB, auto-detects GPU)
- **Index text:** `question + sql_skeleton(gold_sql)` — skeletonizing SQL clusters structural templates
- **Fusion:** Reciprocal Rank Fusion `1/(60+rank_bm25) + 1/(60+rank_embed)` (RRF, no normalization needed)
- **Cache:** Pre-computed embeddings saved to `data/ehrsql/train_embeddings_bge_large.npy`

---

## Next Steps (in order)

### Step 1 — Run Hybrid RAGAS Eval (after ORPO v4 frees GPU)

```bash
bash scripts/eval_retrieval.sh all    # compares BM25, embed, hybrid
```

Expected: Context Recall@2 → 55-65% (up from 34.6%). Hard cluster Hit@2 → 45-55%.

### Step 2 — Full Eval: ORPO v4 + Hybrid RAG

```bash
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v4/adapter_final \
  --output tests/evalgen/orpo_v4_hybrid_rag_results.json \
  --repair --few-shot --retrieval-mode hybrid
```

### Step 3 — Decision Point

| ORPO v4 + hybrid RAG result | Next step |
|---|---|
| RS(10) ≥ 0.68 | Continue with 7B model: more ORPO refinement, self-consistency voting |
| RS(10) < 0.65 | Upgrade to Qwen2.5-Coder-14B (fits in 16 GB at 4-bit, ~8.5 GB) |

### Step 4 — If Staying on 7B: Self-Consistency Voting

```bash
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v4/adapter_final \
  --repair --few-shot --retrieval-mode hybrid \
  --num-samples 5 \
  --output tests/evalgen/orpo_v4_hybrid_vote5_results.json
```

Voting rule: if ≥ ceil(5/2) rollouts abstain → [ABSTAIN], else take majority result set. Expected +3–5% EX.

### Step 5 — If Model Upgrade: Change config.py

```python
# src/ehrcopilot/config.py
INFERENCE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"  # was 7B
```

The rest of the pipeline (SFT, ORPO, eval) picks up the change automatically. Re-run the full pipeline: SFT → ORPO v1 → v2 → v3 for the 14B model.

---

## Scoring Reference

```
RS(10) = (correct_answers + correct_abstentions − 10 × wrong_on_unanswerable) / 1786
```

| To RS(10) | Need correct_answers ≈ | Need wrong_on_unans ≤ |
|---|---|---|
| 0.65 (next milestone) | 700 | 10 |
| 0.75 | 820 | 7 |
| 0.813 (target) | 905 | 5 |

Current: correct_answers = 605, wrong_on_unans = 13.

---

## Key Technical Notes

### Why Hard Cluster [800–1175] Is Hard

Survival/mortality calculation queries in this range require:
- 3-level nested subqueries
- `strftime()` date arithmetic for multi-year survival windows
- `HAVING min(charttime)` for first-occurrence cohort selection
- 3-way JOINs (diagnoses → admissions → patients)
- `CASE/WHEN` for alive-vs-dead rate calculation

Gold SQL for these is 600–800 chars. The old 120-char retrieval cap cut off the most important structural information.

### Why GRPO Doesn't Work Here (Both Attempts Failed)

SQL generation is bimodal: the model either knows the join pattern (all K rollouts correct → std=0) or doesn't (all K rollouts wrong → std=0). Zero within-group variance → zero advantage estimates → zero policy gradient. Not fixable with 3-tier rewards or temperature scaling. GRPO is the wrong tool for constrained-structure generation.

### Embedding GPU Usage

The embedding model (BAAI/bge-large-en-v1.5) uses GPU when available:
- Training embeddings: computed once, cached to disk (9318 examples, ~5 min on GPU, ~15 min on CPU)
- At eval time: 7B LLM uses GPU (5.6 GB at 4-bit), embedding model uses GPU for per-query encoding (~300 MB additional VRAM, well within 16 GB budget)
- DO NOT run hybrid eval while ORPO v4 is training (GPU already at capacity)

### Evaluation Protocol

Always evaluate with `--repair --few-shot` as the standard configuration. Compare against `tests/evalgen/baselines.json` baseline.

For RAGAS eval: use `bash scripts/eval_retrieval.sh [mode]`. Results written to `tests/evalgen/rag_eval_results.json`.
