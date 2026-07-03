# EHR Copilot — Full Diagnostic Report

**Date:** 2026-06-28  
**Model evaluated:** ORPO v4 Colab (full run, 327/327 steps, Qwen2.5-Coder-7B-Instruct + QLoRA 4-bit)  
**Benchmark:** EHRSQL MIMIC-III, 1786 test questions (1198 answerable, 588 unanswerable)  
**Database used:** MIMIC-IV-Demo v2.2 (local)

---

## Reported Scores (Last Full Eval)

| Metric | Value |
|---|---|
| EX (execution accuracy) | 0.4775 |
| RS(0) | 0.6338 |
| RS(5) | 0.5554 |
| RS(10) | **0.4770** |
| correct_answers | 572 / 1198 |
| wrong_abstentions | 10 / 1198 |
| wrong_answers_on_unanswerable | 28 / 588 |
| correct_abstentions | 560 / 588 |
| repair_attempts / successes | 1867 / 87 |
| p50 latency | 1746 ms |

**Target:** RS(10) ≥ 0.813 (EHRSQL 2024 leaderboard, LG AI / KAIST)  
**Gap:** −0.336

---

## Problem 1 — Wrong Database (Root Cause of Everything)

**Severity: CRITICAL**

The EHRSQL 2024 competition runs on **MIMIC-III** (full database, ~46,000 patients, ~58,000 admissions). Our local environment uses **MIMIC-IV-Demo v2.2** — a demo subset with a different schema and a fraction of the data.

| Property | MIMIC-III (competition) | MIMIC-IV-Demo (our DB) |
|---|---|---|
| Patients | ~46,000 | **100** (0.2%) |
| Admissions | ~58,000 | **275** (0.5%) |
| Schema | MIMIC-III | MIMIC-IV (restructured) |
| `cost` table | ✓ present | ✗ missing |
| `inputevents_cv` table | ✓ present | ✗ missing |
| `outputevents` table | ✓ present | ✗ missing |
| `diagnoses_icd.charttime` | ✓ present | ✗ missing |

**Consequence:** Every RS(10) score measured locally is against the wrong database. The 0.813 leaderboard target was measured on MIMIC-III. Our local scores are not comparable to competition scores.

---

## Problem 2 — The EX Metric Is Broken (Massive False Positives)

**Severity: CRITICAL**

The scoring function compares execution results as `frozenset()`. When a query errors or returns empty, `_normalize_result` returns `frozenset()` for both cases. This means **two different SQL queries that both fail count as matching**.

```
gold SQL has error   → _exec_safe returns (None, err)  → frozenset()
model SQL has error  → _exec_safe returns (None, err)  → frozenset()
frozenset() == frozenset()  →  True  →  counted CORRECT
```

**Audit of all 1198 answerable test questions against our DB:**

| Gold SQL outcome | Count | % |
|---|---|---|
| Returns actual results | **256** | **21.4%** |
| Returns empty (patient not in demo) | 361 | 30.1% |
| SQL execution error | 581 | 48.5% |

**The 572 "correct" answers are unreliable.** Since only 256 gold queries return real data, the maximum possible true correct answers is 256. The other **316+ are false positives** — cases where both gold and model return empty or error, producing a `frozenset() == frozenset()` match regardless of what the model generates.

**A model that outputs random gibberish would still score "correct" on 48.5% of questions** (those where gold SQL errors — any model output that also errors produces a match).

---

## Problem 3 — Three Entire Tables Missing from Our DB

**Severity: HIGH**

| Missing table | Queries affected | Description |
|---|---|---|
| `inputevents_cv` | 97 | MIMIC-III ICU fluid inputs (merged into `inputevents` in MIMIC-IV) |
| `outputevents` | 65 | MIMIC-III ICU fluid outputs (not in MIMIC-IV-Demo) |
| `cost` | 52 | MIMIC-III procedure/drug cost table (removed in MIMIC-IV) |

**214 test questions cannot execute on our DB under any circumstances.** No amount of fine-tuning fixes this — the data simply does not exist in MIMIC-IV-Demo.

---

## Problem 4 — Five Missing Columns Not Handled by Canonicalization

**Severity: HIGH**

The codebase has a `_canonicalize_gold_sql()` function that remaps 4 MIMIC-III column names to MIMIC-IV equivalents. It misses the following:

| Missing column | Queries affected | Issue |
|---|---|---|
| `diagnoses_icd.charttime` | 162 | Does not exist in MIMIC-IV (date is on `admissions`) |
| `procedures_icd.charttime` | 125 | DB has `chartdate`, not `charttime` |
| `admissions.age` | 55 | MIMIC-III column; MIMIC-IV uses `patients.anchor_age` |
| `transfers.wardid` | 10 | Replaced by `transfers.careunit` in MIMIC-IV |
| `patients.dob` | 6 | Removed (PHI); MIMIC-IV uses `anchor_age` + `anchor_year` |

**Total: 358 additional column errors** that fail silently and contribute to false positive scoring.

---

## Problem 5 — ORPO Training Pairs Have Backwards Signal

**Severity: HIGH**

ORPO v4 used **string-diff matching** (not execution verification) to identify "wrong" model outputs. This causes a systematic training error:

**Example of backwards pair:**
- Gold SQL (MIMIC-III): `SELECT ... FROM inputevents_cv WHERE ...`
- Model output (MIMIC-IV): `SELECT ... FROM inputevents WHERE ...`
- String-diff: different → **added as training pair with gold as "chosen", model as "rejected"**
- Model learns: **prefer `inputevents_cv` (broken on our DB) over `inputevents` (works)**

Since 48.8% of training gold SQL has errors on our DB, roughly **half of the 4,856 answerable ORPO v4 training pairs** trained the model to prefer MIMIC-III-specific SQL that cannot execute in our evaluation environment. The model was actively trained in the wrong direction on these questions.

**Training data status (9,318 answerable train questions):**

| Gold SQL outcome | Count | % |
|---|---|---|
| Returns actual results | 1,914 | 20.5% |
| Returns empty | 2,860 | 30.7% |
| SQL execution error | 4,544 | 48.8% |

Only 1,914 training questions (20.5%) provide clean, verifiable preference signal.

---

## Problem 6 — Prompt Truncation on the Hardest Questions

**Severity: MEDIUM**

Full prompt length analysis across all 1198 answerable test questions with K=2 template retrieval:

| Percentile | Estimated tokens |
|---|---|
| p25 | 1,032 |
| p50 (median) | 1,090 |
| p75 | 1,167 |
| p90 | 1,426 |
| p95 | 1,474 |
| max | 1,958 |

**42 prompts (3.5%) exceed the 1,536 token limit — ALL in the hardest complexity tier (tier 8+).** These are the survival/vital-change templates with SQL averaging 1,133 characters. 25.3% of tier 8 examples get truncated, cutting off the actual test question mid-sentence. The model generates for a different question than intended.

---

## Problem 7 — Unanswerable Detection Weakness

**Severity: MEDIUM**

28 wrong-on-unanswerable × 10 RS(10) penalty = **280 lost RS(10) points**.

The unanswerable questions have the same syntactic structure as answerable ones (avg length 65 chars vs 90 for answerable — similar). The model cannot distinguish them from surface text alone. 

Sample questions the model incorrectly generates SQL for:
- "Which doctor did patient 85895 see during their last visit to the hematology department?" — no provider table in DB
- "What effect does soma have?" — drug pharmacology not tracked
- "Does patient 19175 have a family history of colostomy status?" — family history not in DB
- "What is the phone number for patient 52342's companion?" — contact info not in DB

The current system prompt names these categories explicitly, but the model still generates SQL for ~4.8% of unanswerable questions.

---

## Summary of All Issues

| # | Problem | Queries affected | Fixable without MIMIC-III? |
|---|---|---|---|
| 1 | Wrong database (MIMIC-IV-Demo vs MIMIC-III) | All 1786 | No — need MIMIC-III |
| 2 | False positive scoring (empty↔empty match) | ≥ 316 false positives | Partial (filter to valid-gold-only) |
| 3 | Three missing tables (`cost`, `inputevents_cv`, `outputevents`) | 214 | No |
| 4 | Five missing columns not canonicalized | 358 | Partial (code fix) |
| 5 | Backwards ORPO training pairs (string-diff vs broken gold) | ~2,300 pairs | Yes (use verify-execution) |
| 6 | Prompt truncation on tier 8 (25% of hardest questions) | 42 | Yes (max_length=2048 on A100) |
| 7 | Unanswerable detection (28 wrong → 280 RS pts lost) | 28 | Partial (ORPO v5) |

---

## Fix Priority

### Priority 1 — Get MIMIC-III Database (unlocks everything)

Request the MIMIC-III SQLite database from PhysioNet (same credentialing as MIMIC-IV). The EHRSQL 2022 paper provides the exact DB used. With it: all gold SQL executes correctly, false positives vanish, EX becomes meaningful, and training pairs are verifiable.

Without MIMIC-III, all local measurements are estimates with an unknown bias. The 0.813 leaderboard target is unreachable to verify locally.

### Priority 2 — Fix the Training Signal Now (no DB needed)

Switch all pair generation to `--verify-execution`. Only generate pairs for the 1,914 train questions where gold SQL returns real data. This is already the plan for ORPO v5 — do not reintroduce string-diff pairs.

### Priority 3 — Fix Local Evaluation (no DB needed)

Add a `--valid-gold-only` mode to `harness.py` that skips questions where gold SQL errors or returns empty. This gives a clean local signal: model accuracy on the 256 questions where execution actually means something. Current "EX=0.4775" would become a meaningful number (probably lower, since many of the 572 "correct" are false positives).

### Priority 4 — Expand Canonicalization (code fix, 1 hour)

Add the 5 missing column remaps to `_canonicalize_gold_sql`. This would recover ~358 broken gold SQL queries and reduce false positives. Won't fix the 3 missing tables (214 queries) but improves the signal.

### Priority 5 — Prompt Length (Colab A100)

Set `max_length=2048` on A100 GPU (it has 40GB VRAM vs 16GB RTX 4080 Super locally). Eliminates the tier 8 truncation issue entirely. Already in the Colab batch-size autodetect logic — just needs the max_length to be bumped from 1536 to 2048 in the A100 branch.

---

## What the Real Performance Likely Is

| Metric | Reported | Estimated Real |
|---|---|---|
| correct_answers | 572 | ≤ 256 (bounded by valid gold SQL count) |
| False positives | 0 reported | ≥ 316 |
| EX on valid-gold-only questions | 0.4775 | Unknown (requires filtering) |
| RS(10) on correct DB | 0.477 | Unknown (requires MIMIC-III) |

The model has genuinely learned SQL generation and abstention — the RS trajectory (baseline 0.476 → ORPO v3 0.588 → ORPO v4 0.477 locally) does show real improvement in abstention quality (wrong_on_unanswerable dropped from 176 baseline to 28). However, the absolute numbers cannot be trusted until the DB mismatch is resolved.

---

*Generated by diagnostic run: `scripts/diagnostic.py`, 2026-06-28*  
*Pair generation (ORPO v5) running in background: `logs/build_pairs_v5_full.log`*
