# Training Progress & Experiment Log

**Goal:** Beat the EHRSQL 2024 leaderboard top score of RS(10) = 0.8132 (LG AI / KAIST)  
**Model:** Qwen2.5-Coder-7B-Instruct fine-tuned with QLoRA on MIMIC-IV-Demo / EHRSQL  
**Hardware:** RTX 4080 Super 16GB (training) · RTX 5080 16GB GDDR7 (inference)  
**Current best:** RS(10) = **0.5879** (ORPO v3 + Repair Loop + BM25 RAG)

---

## The Scoring Formula

```
RS(N) = (correct_answers + correct_abstentions − N × wrong_on_unanswerable) / total
```

The test set has **1,786 questions**: 1,198 answerable + 588 unanswerable.

| Model output | Points |
|---|---|
| Correct SQL on answerable | +1 |
| Correct [ABSTAIN] on unanswerable | +1 |
| Wrong SQL on answerable | 0 |
| [ABSTAIN] on answerable | 0 |
| **SQL on unanswerable** | **−10** (clinical safety penalty) |

At N=10, each hallucination on an unanswerable question costs **11 net RS points** (you lose +1 for correct abstention AND take −10). This penalty structure mirrors the real-world risk of a model confidently answering an impossible clinical question.

---

## Experiment History

### Baseline — Qwen2.5-Coder-7B-Instruct (no fine-tuning)

**RS(10) = −0.705**

The raw model had no concept of when to refuse. It answered every question with SQL, including ones with no valid answer. 176 hallucinations on unanswerable questions × −10 = −1,760 raw penalty points, overwhelming the 90 correct SQL answers.

```json
{
  "EX": 0.0751,
  "RS(10)": -0.705,
  "correct_answers": 90,
  "wrong_answers_on_unanswerable": 176,
  "correct_abstentions": 412
}
```

**Takeaway:** Without abstention training, RS(10) is negative. The penalty for hallucinating on unanswerable questions must be addressed.

---

### SFT v1 — QLoRA Supervised Fine-Tuning (first attempt)

**RS(10) = −0.262** (computed post-hoc with corrected formula)

Trained on 9,318 answerable + 362 unanswerable examples (format: question → SQL or [ABSTAIN]).  
3 epochs, final loss 0.06. Adapter at `checkpoints/sft/adapter_final`.

```json
{
  "EX": 0.1027,
  "RS(10)": -0.262,
  "correct_answers": 123,
  "wrong_answers_on_unanswerable": 24,
  "correct_abstentions": 564
}
```

**Problem discovered:** The DPO training that followed this SFT checkpoint used the base model as reference (instead of the SFT model), creating a large distributional gap → poorly conditioned DPO gradients → catastrophic over-abstention on answerable questions.

---

### T1 Retraining — Bug Fixes Applied Before Retraining

Before continuing, several bugs were identified and fixed:

| Bug | Fix |
|-----|-----|
| **T1-1: Gold SQL not canonicalized** | Eval harness now renames MIMIC-III column names (e.g. `icustay_id → stay_id`) to match MIMIC-IV-Demo schema before execution |
| **T1-2: Training data used incompatible tables** | `prepare_sft.py` now filters to MIMIC-IV-Demo tables, applies the same column renames, and oversamples unanswerable examples by 20% |
| **T1-4: DPO had reference model mismatch** | Switched to ORPO (no reference model needed — preferred/rejected ratio directly in the loss) |
| **T1-5: Incomplete schema in prompts** | Full schema passed at training and eval time |
| **RS formula bug** | `harness.py` was penalizing wrong abstentions at rate −N. Correct formula: wrong abstentions score 0, only wrong_on_unanswerable are penalized |

---

### SFT v2 — Retrained with T1 Fixes

**RS(10) = 0.467**

3 epochs, loss 0.022 (vs 0.06 before — much tighter fit after schema/data fixes).

```json
{
  "EX": 0.5083,
  "RS(10)": 0.467,
  "correct_answers": 609,
  "wrong_answers_on_unanswerable": 33,
  "correct_abstentions": 492
}
```

EX jumped from 10.3% → 50.8% — a 3.6× improvement purely from fixing the data/schema bugs. The model now rarely abstains on answerable questions (only 59 wrong abstentions vs 35 in DPO v1).

---

### ORPO v2 — Abstention Fine-Tuning

**RS(10) = 0.501** (best at this point)

ORPO (Odds Ratio Preference Optimization) doesn't need a reference model, avoiding the distributional gap that killed DPO v1.

Built 510 pairs: for each unanswerable question, chosen = [ABSTAIN], rejected = the SQL the SFT model would generate. Also added answerable counterbalance pairs.

1 epoch (epoch 2 OOM'd at step 33 on the RTX 4080 Super — used checkpoint-32 as final).

```json
{
  "EX": 0.5042,
  "RS(10)": 0.501,
  "correct_answers": 604,
  "wrong_answers_on_unanswerable": 27,
  "correct_abstentions": 561
}
```

ORPO reduced wrong_on_unans from 33 → 27 (−6). Each unit saved = +11 RS points = +66 net points. Small EX dip (50.8% → 50.4%) was worth it.

---

### GRPO v1 — Execution-Reward Training (failed)

GRPO (Group Relative Policy Optimization) generates K=4 SQL variants per question, executes them against the real database, and updates the model toward whichever variants were correct.

**Failed with `frac_reward_zero_std = 1.0`** — 100% of training batches had zero reward variance.

**Root cause:** The binary reward {0 = wrong, 1 = correct} combined with a high-confidence model means: for each question, all K=4 rollouts tended to either all succeed or all fail → std=0 within the group → advantage=0 → zero gradient. The model was too certain to explore.

Abandoned. Switched to ORPO v3 instead.

---

### ORPO v3 — Execution-Verified Pairs

**RS(10) = 0.514** (new best)

Key improvements over ORPO v2:
- 865 pairs (362 unanswerable + 503 answerable) vs 510 pairs
- `--verify-execution` flag: for answerable pairs, the model's rejected output is only used if it actually executes to a *wrong* result. If the model's output is correct (executes and matches gold), the pair is skipped — no noise in the training signal.
- Skipped 304 questions where ORPO v2 already answers correctly and 193 where it abstains.
- Starting from ORPO v2 adapter (refining instead of from scratch).

2 epochs, 110 gradient steps, lr=5e-5, orpo_lambda=0.1.

```json
{
  "EX": 0.4499,
  "RS(10)": 0.514,
  "correct_answers": 539,
  "wrong_answers_on_unanswerable": 19,
  "correct_abstentions": 569
}
```

EX dipped (50.4% → 45.0%) but wrong_on_unans fell from 27 → 19. Each of the 8 saved hallucinations = +11 RS points = +88 RS points gained. Net: +31 RS points → RS(10) 0.501 → 0.514.

**Hard cluster phenomenon discovered:** The EHRSQL test set has a dense cluster of multi-join / nested subquery questions from approximately examples [800–1175]. EX dropped from 55.9% to 45.8% through this segment, then flatlined. This is an inherent property of the test set ordering, not a model regression.

---

### P1 + P5 — Repair Loop + BM25 RAG (inference-only, no retraining)

**RS(10) = 0.5879** — largest single gain (+0.074)

Two inference-time techniques applied to the ORPO v3 adapter with no additional training:

**P1 — Execution-Guided Repair Loop** (`--repair` flag):  
When the model's SQL fails with a SQLite error, show the model the error message and ask it to fix the SQL. Up to 3 retry attempts per question. Implemented in `harness.py` via `_UnslothPredictor.repair()`.

**P5 — BM25 Few-Shot RAG** (`--few-shot` flag):  
Before generating SQL for a question, retrieve the 2 most similar question→SQL pairs from the training set using BM25 (keyword-based retrieval). Prepend these as examples in the system prompt.

```
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v3/adapter_final \
  --output tests/evalgen/orpo_v3_repair_rag_results.json \
  --repair --few-shot
```

```json
{
  "EX": 0.505,
  "RS(10)": 0.5879,
  "correct_answers": 605,
  "wrong_answers_on_unanswerable": 13,
  "correct_abstentions": 575,
  "repair_attempts": 1494,
  "repair_successes": 125
}
```

**What moved:**
- +66 correct answers (539 → 605): RAG examples helped on medium-difficulty questions; repair loop rescued 125 broken SQL queries
- −6 wrong_on_unans (19 → 13): few-shot examples improved calibration on unanswerable questions
- −18 wrong abstentions (79 → 61): model more confident on answerable questions it now recognizes from examples

**EX profile through the test set:**

| Range | EX | Notes |
|-------|----|-------|
| [0–300] | 32% → 58% | Early variance + steady climb |
| [300–775] | 58% → 66% | Best window, RAG helping most |
| [775–1175] | 66% → 51% | Hard cluster (multi-join / subquery) |
| [1175–1786] | 51% (flat) | Stabilized, matches ORPO v3 pattern |

RS(10) check: (605 + 575 − 130) / 1786 = **1050 / 1786 = 0.5879** ✓

---

## Current Leaderboard

| Model | EX | RS(0) | RS(5) | RS(10) | correct_ans | wrong_on_unans |
|-------|----|-------|-------|--------|-------------|----------------|
| Baseline (no FT) | 7.5% | 0.050 | −1.31 | −0.705 | 90 | 176 |
| SFT v2 | 50.8% | 0.530 | 0.498 | 0.467 | 609 | 33 |
| ORPO v2 | 50.4% | 0.525 | 0.513 | 0.501 | 604 | 27 |
| ORPO v3 | 45.0% | 0.620 | 0.567 | 0.514 | 539 | 19 |
| **ORPO v3 + Repair + RAG** | **50.5%** | **0.661** | **0.624** | **0.5879** | **605** | **13** |
| EHRSQL 2024 leader (target) | ~77% | — | — | **0.813** | ~922 | ~5 |

---

## Gap Analysis

```
Current RS(10) score : 1050 / 1786  (correct_ans=605, correct_abs=575, wrong_on_unans=13)
Target RS(10) score  : 1452 / 1786
Gap                  : 402 points
```

To close 402 points:
- **Reduce wrong_on_unans 13 → 3** → saves 10 × 11 = **+110 RS points**
- **Increase EX 50.5% → 77%** → adds ~317 correct answers = **+317 RS points**
- Margin above target: 25 points

---

---

### GRPO v2 — Execution-Reward Training (failed, same root cause as v1)

Both GRPO attempts failed with the same structural problem.

**Root cause — SQL generation is bimodal and constrained:**

For any given question, the model generates near-identical SQL across all K=4 rollouts. Temperature=1.5 doesn't help because the SQL structure is dictated by the schema and question — the model either knows the right join pattern or it doesn't. Within each K-group:
- If the model "knows" the answer: all 4 rollouts correct → all +1.0 → std=0
- If the model doesn't know: all 4 rollouts wrong → all -0.5 or all -0.2 → std=0

Step-10 metrics from the final smoke test confirmed total failure:
```
'frac_reward_zero_std': '0.9'   → 90% of groups still zero within-group variance
'grad_norm': '0'                 → no policy gradient at all
'loss': '0.0005834'             → only KL penalty term, zero RL signal
```

**GRPO is not the right tool for SQL generation.** It requires diversity within rollout groups; SQL doesn't provide it. Abandoned.

---

## Currently Running: ORPO v4

**Status (as of 2026-06-26):** Training at step ~30/327 (~9%). ETA: Saturday 2026-06-27 ~07:00 EDT.  
PID: 1257446 · Log: `logs/orpo_v4_*.log`

**Why ORPO v4:** ORPO v3 used only 503 answerable pairs out of 9,318 training questions (5.4% coverage). ORPO v4 builds execution-verified pairs for **all 9,318 training questions**, producing:

| Pair type | Count |
|---|---|
| Answerable (wrong SQL → correct SQL) | 4,856 |
| Unanswerable (SQL → [ABSTAIN]) | 362 |
| **Total** | **5,218** |

3,885 questions skipped — model already answers correctly. 577 skipped — model abstains on answerable (handled in a later pass if needed).

**After training completes, run:**

```bash
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v4/adapter_final \
  --output tests/evalgen/orpo_v4_results.json \
  --repair --few-shot
```

---

## RAG Overhaul (Parallel Work)

While ORPO v4 trains, the retrieval system has been rebuilt. The BM25-only retrieval had two structural failures:

1. **120-char SQL truncation** — survival/mortality SQL (600–800 chars) was cut off mid-expression, making retrieved examples misleading rather than helpful. **Fixed: full SQL now included.**

2. **BM25 keyword failure on hard cluster** — survival queries ("mortality percentage", "survival rate") share keywords with simple lookup queries, so BM25 retrieves irrelevant examples. **Fixed: semantic embedding retrieval added.**

### RAGAS Baseline Measurement (BM25)

Measured using `tag` field as relevance signal (abstract template class, 919 unique values):

| K | Context Recall@K | Context Precision@K |
|---|---|---|
| 1 | 0.2654 | 0.2654 |
| 2 | 0.3456 | 0.2250 |
| 3 | 0.3923 | 0.1992 |
| 5 | 0.4699 | 0.1731 |
| 10 | 0.5684 | 0.1379 |

**MRR: 0.3517**

Per-cluster breakdown:
- Overall: Hit@2 = 34.6%
- Hard cluster [800–1175]: Hit@2 = **23.7%** — 31% below average

### Hybrid Retrieval (BM25 + BAAI/bge-large-en-v1.5 + RRF)

New retrieval architecture in `src/ehrcopilot/eval/harness.py`:

- Embedding model: **BAAI/bge-large-en-v1.5** (335 MB, GPU if available, CPU fallback)
- Index text: `question + sql_skeleton(gold_sql)` — SQL skeleton exposes structural template
- Fusion: **Reciprocal Rank Fusion** `1/(60+rank_bm25) + 1/(60+rank_embed)`
- Embeddings pre-computed once and cached to `data/ehrsql/train_embeddings_bge_large.npy`

**Run hybrid RAGAS measurement (after ORPO v4 frees the GPU):**

```bash
bash scripts/eval_retrieval.sh hybrid      # or: all (compares BM25 vs embed vs hybrid)
```

**Expected improvement:**
- Hard cluster Hit@2: 23.7% → 45-55%
- Overall MRR: 0.35 → 0.50+

### Full Eval with Hybrid RAG

```bash
bash scripts/run_sft_eval.sh \
  --adapter checkpoints/orpo_v4/adapter_final \
  --output tests/evalgen/orpo_v4_hybrid_rag_results.json \
  --repair --few-shot --retrieval-mode hybrid
```

---

## Projected Path to Target

| Step | EX | RS(10) |
|------|-----|--------|
| ORPO v3 + Repair + BM25 RAG (current best) | 50.5% | 0.5879 |
| ORPO v4 + Repair + BM25 RAG | ~58–65% | ~0.63–0.70 |
| ORPO v4 + Repair + **Hybrid RAG** | ~62–70% | ~0.67–0.77 |
| Model upgrade to Qwen2.5-Coder-14B (if <0.65) | ~70–75% | ~0.78–0.85 |
| **EHRSQL 2024 target** | **~77%** | **0.813** |

---

## Model Upgrade Decision Threshold

If ORPO v4 + hybrid RAG eval shows RS(10) < 0.65, upgrade to **Qwen2.5-Coder-14B**:
- 14B at 4-bit NF4 fits in ~8.5 GB VRAM (hardware has 16 GB)
- Single `config.py` change propagates through entire pipeline
- Projected RS(10): 0.80–0.87 with same fine-tuning approach

---

## Key Technical Files

| File | Purpose |
|------|---------|
| `src/ehrcopilot/eval/harness.py` | Eval loop: EX + RS metrics, repair loop, BM25 RAG, progress logging |
| `src/ehrcopilot/finetune/prepare_sft.py` | Build SFT training data with MIMIC-IV column renames and unanswerable oversample |
| `src/ehrcopilot/finetune/qlora_sft.py` | QLoRA SFT training (Unsloth + TRL SFTTrainer) |
| `src/ehrcopilot/finetune/build_pairs.py` | Build ORPO/DPO preference pairs with `--verify-execution` |
| `src/ehrcopilot/finetune/abstention_dpo.py` | ORPO training (Unsloth + TRL ORPOTrainer) |
| `src/ehrcopilot/finetune/grpo_train.py` | GRPO v2 training with 3-tier execution reward |
| `scripts/run_sft_eval.sh` | Run eval with optional `--repair` and `--few-shot` flags |
| `scripts/run_grpo.sh` | Launch GRPO v2 training (defaults: ORPO v3 adapter, temp=1.2, full dataset) |
| `scripts/train_pipeline.sh` | End-to-end pipeline: SFT → ORPO → merge → eval |

## Result Files

| File | Description |
|------|-------------|
| `tests/evalgen/baseline_results.json` | Qwen2.5-Coder-7B-Instruct, no fine-tuning |
| `tests/evalgen/sft_results.json` | SFT v2 results |
| `tests/evalgen/dpo_results.json` | DPO v1 results (poor — reference model mismatch) |
| `tests/evalgen/orpo_v3_results.json` | ORPO v3 results |
| `tests/evalgen/orpo_v3_repair_rag_results.json` | ORPO v3 + repair loop + BM25 RAG (current best) |

---

## Lessons Learned

**1. The RS formula penalty structure dominates early.**  
A model that confidently answers unanswerable questions will have negative RS(10) no matter how good its SQL is. The first priority must be calibrated abstention, not raw EX.

**2. Schema alignment is everything.**  
The EHRSQL dataset uses MIMIC-III column names (e.g. `icustay_id`) but MIMIC-IV-Demo uses different names (e.g. `stay_id`). Without canonicalization in both training data and the eval harness, EX was artificially suppressed — the model was penalized for wrong answers when the gold SQL itself couldn't execute against the target database.

**3. Binary GRPO rewards fail on high-confidence models.**  
With K=4 rollouts and a model at 50% EX, you'd naively expect mixed rewards within each group. In practice, the model is strongly bimodal: it either knows the answer (all 4 correct) or doesn't (all 4 wrong). Binary {0,1} rewards → std=0 → zero gradient. The fix: 3-tier rewards that create variance even when all rollouts fail.

**4. Inference-time techniques compound well with fine-tuning.**  
The repair loop + BM25 RAG gave +0.074 RS(10) on top of ORPO v3 with zero GPU training time. These are now the standard eval configuration — every adapter is evaluated with `--repair --few-shot`.

**5. ORPO > DPO for abstention without a reference model.**  
DPO requires a reference model (frozen copy of the trained model). If the reference model doesn't match the current adapter's training distribution, gradients are poorly conditioned. ORPO avoids this entirely by computing the preference ratio directly, making it robust to the SFT → ORPO transition.
