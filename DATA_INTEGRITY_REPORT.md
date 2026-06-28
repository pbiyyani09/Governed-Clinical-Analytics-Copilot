# EHRSQL Data-Integrity Report — wrong database build + wrong question set

**Date:** 2026-06-28 · Branch: `gemma_dev_rebased`
**Status:** EDA final; literature **source-verified** by a `research-nlp` investigation (claim
ledger + URLs in §7). Net finding revised after research: **no credentialed data access is needed.**

---

## TL;DR (read this)

Two independent mistakes are stacked on top of each other:

1. **Wrong question set.** The repo ships the **EHRSQL-2022 MIMIC-III** question/SQL files
   (`data/ehrsql/ehrsql/mimic_iii/`, plus an eICU split). The **leaderboard you're targeting —
   RS(10)=0.8132 — is the EHRSQL-2024 shared task**, which is a *different, smaller* dataset.
2. **Wrong database build.** `data/mimic_iv_demo.db` is the **raw** MIMIC-IV-Demo schema
   (`anchor_age`, `race`, `stay_id`, `pharmacy`, `edstays`, `icd_code+icd_version`, no `cost`).
   Neither benchmark uses that schema. EHRSQL uses its own **preprocessed, flattened** schema
   (`dob`, `cost`, `row_id`, `icd9_code`/`short_title`, `icustay_id`).

Executing the (MIMIC-III) gold SQL against the (raw MIMIC-IV-Demo) DB: **71.0% raise SQL errors,
28.7% return empty, 0.3% return a real row.** So **EX has been un-measurable** — the ~0.40 plateau
across prompting *and* fine-tuning is a broken-benchmark artifact, not a model ceiling.

**The good news — your MIMIC-III plan isn't needed.** EHRSQL-2024 is built on the
**open-access MIMIC-IV Demo v2.2** (ODC-ODbL, *no PhysioNet credentialing*), and the organizers ship
a ready-to-run dataset + scorer. The fix is to **adopt the EHRSQL-2024 repo wholesale** (its
preprocessed SQLite + its question files + its official scorer) — not to acquire MIMIC-III, and not
even to get full MIMIC-IV. See §6.

---

## 1. What is actually in the repo (EDA)

`data/ehrsql/ehrsql/` is the **original EHRSQL (NeurIPS 2022)** release — both databases:

| split | mimic_iii | eicu |
|---|---|---|
| train | 9,318 | 9,270 |
| valid | 1,122 | 1,117 |
| test  | 1,786 | 1,792 |

Every record has `db_id:"mimic_iii"` and the EHRSQL-2022 field set (`template, q_tag, t_tag, o_tag,
department, importance, para_type, is_impossible, …`). Unanswerable (`is_impossible`): train 0,
valid 362 (32.3%), test 588 (32.9%).

The DB we execute against, `data/mimic_iv_demo.db`, is a **raw MIMIC-IV-Demo** build (17 tables incl.
`pharmacy`, `edstays`; `patients.anchor_age/anchor_year`, `admissions.race`, `*.icd_code+icd_version`,
`*.stay_id`; **no `cost`/`inputevents`/`outputevents`**).

---

## 2. Schema mismatch (MIMIC-III gold SQL vs raw MIMIC-IV-Demo DB)

**Tables in the gold schema but absent from the DB:** `cost`, `inputevents_cv`, `outputevents`.
**Tables only in the DB:** `edstays`, `pharmacy`.

**Column breaks driving the failures** (gold uses → DB reality):

| MIMIC-III gold | raw MIMIC-IV-Demo DB | gold uses (count, all splits) |
|---|---|---|
| `patients.dob` | `anchor_age`,`anchor_year` | 50 |
| `admissions.age`,`.ethnicity` | `age` gone; `ethnicity`→`race` | 605 (`age`) |
| `*.row_id` | dropped | many |
| `diagnoses_icd.icd9_code`, `d_icd_diagnoses.short_title` | `icd_code`+`icd_version`; no `short_title` | 2,014 (`short_title`) |
| `icustays.icustay_id`, `chartevents.icustay_id` | `stay_id` | 2,494 + 1,171 |
| `prescriptions.startdate/enddate` | `starttime`/`stoptime` | 2,088 |
| `diagnoses_icd.charttime`, `procedures_icd.charttime` | absent / `chartdate` | 1,796 + 1,473 |

These are the spine of the query set, not edge cases.

---

## 3. Execution test: EX is un-measurable on this pairing

All **1,198 answerable test** gold queries run against `mimic_iv_demo.db`:

| outcome | count | % |
|---|---|---|
| non-empty result (only these can score a real EX) | **3** | **0.3%** |
| empty / zero | 344 | 28.7% |
| SQL error (`no such column` 637, `no such table` 214) | 851 | 71.0% |

219 answerable test queries reference a MIMIC-III-only table (`cost` 52, `inputevents_cv` 97,
`outputevents` 70). With 99.7% of gold executions degenerate, the EX oracle is broken: the only way
to "match" gold is to also return empty/abstain. **Every EX/RS number produced on this pairing is
untrustworthy** — base prompting sweep and the Gemma-3/Gemma-4 fine-tunes alike.

---

## 4. EHRSQL-2022 vs EHRSQL-2024 (source-verified)

**EHRSQL (NeurIPS 2022 D&B)** — Lee et al., arXiv:2301.07695, `glee4810/EHRSQL`. Databases:
**MIMIC-III v1.4 + eICU v2.0** (credentialed; value-shuffled SQLite mirrors are on Google Drive).
Questions from a 222-staff poll at a hospital; 230 templates (174 answerable / 56 unanswerable);
per-DB splits 9.3K/1.1K/1.8K. **← exactly what's in this repo.** (2022 used F1_exe, not RS.)

**EHRSQL-2024 shared task** — Lee et al., ACL Anthology `2024.clinicalnlp-1.62`, arXiv:2405.06673,
`glee4810/ehrsql-2024`, Codabench comp. 1889. **Database: MIMIC-IV *Demo* v2.2** — open-access,
ODC-ODbL, **no credentialing**; the organizers' `preprocess.sh` turns the 100-patient demo into a
bespoke flattened SQLite and they also ship a pre-built `data/mimic_iv/mimic_iv.sqlite`. This is the
**official evaluation DB** — empty result sets are *valid* answers when the gold also returns empty,
so 100 patients is by design, not a limitation.

The 2024 schema is deliberately **EHRSQL-flavored** (17 tables: …, `cost`, `inputevents`,
`outputevents`; `patients.dob` synthesized as `anchor_year − anchor_age`; **no `race`/`ethnicity`**;
`row_id` retained). So it *looks* MIMIC-III-ish but is sourced from MIMIC-IV demo patients,
time-shifted to year 2100, with values shuffled for de-id.

**2024 split sizes / abstention balance** (counted from the repo's `label.json`):

| split | total | answerable | unanswerable | unans% |
|---|---|---|---|---|
| train | 5,124 | 4,674 | 450 | ~8.8% |
| valid | 1,163 | 931 | 232 | ~20% |
| test  | 1,167 | 934 | 233 | ~20% |

Data format: `data.json` (`{id, question}`) + separate `label.json` (`{id → sql | "null"}`) — leaner
than the 2022 format we have.

**Likely origin of the bug:** the repo was aimed at EHRSQL-2024 (hence a MIMIC-IV-Demo DB and a
MIMIC-IV-style prompt schema) but loaded the EHRSQL-2022 MIMIC-III question files **and** built the
DB from the *raw* demo CSVs instead of the EHRSQL preprocessing pipeline — so the DB matches neither
benchmark.

---

## 5. The Reliability Score, and what 0.8132 is

Per-sample `RS(c)`: **+1** correct answered (answerable, non-abstain, exec-correct); **0** abstain on
answerable; **−c** wrong SQL on answerable; **−c** answered an unanswerable; **+1** correctly
abstained on unanswerable. Overall RS(c) = mean × 100. Penalties used: c ∈ {0, 5, 10, N}; **RS(10)
is the official metric** ("ten correct ≈ one wrong"). `RS(N)` (c = test size) is the brutal variant.

**Leaderboard (test set):**

| rank | team | RS(0) | **RS(10)** | RS(N) |
|---|---|---|---|---|
| 1 | LG AI Research & KAIST | 88.17 | **81.32** | −711.83 |
| 2 | PromptMind | 82.60 | 74.89 | −817.40 |
| 3 | ProbGate | 81.92 | 74.21 | −818.08 |
| — | **ABSTAIN-ALL baseline** | **20.00** | **20.00** | 20.00 |

So **RS(10)=0.8132 is the #1 system on the 2024 test set**; the abstain-everything floor is 20.0
(= the unanswerable fraction). The small RS(0)→RS(10) drop for the top teams (88.17→81.32) is the
signature of *good abstention* — exactly the calibrated-abstention thesis of this project.

---

## 6. What to actually do (recommendation)

**Adopt EHRSQL-2024 directly. No MIMIC-III. No credentialed MIMIC-IV.**

Concretely:
1. **Get the open demo + official assets** (all no-credential):
   - `git clone https://github.com/glee4810/ehrsql-2024`
   - MIMIC-IV Demo v2.2: `https://physionet.org/content/mimic-iv-demo/2.2/` (open). Use the repo's
     pre-built `data/mimic_iv/mimic_iv.sqlite` if present; otherwise run `preprocess/preprocess.sh`
     (`--num_patient 100 --timeshift --current_time "2100-12-31 23:59:00" …`).
2. **Swap the data foundation** to `data/mimic_iv/{train,valid,test}/{data.json,label.json}` and point
   the DB to the EHRSQL-2024 `mimic_iv.sqlite`.
3. **Re-target our code** (small, mechanical):
   - eval harness: replace the hard-coded MIMIC-IV-raw schema string with the EHRSQL-2024 17-table
     schema; load `data.json`/`label.json`; score with the **organizers' scorer** so RS is
     apples-to-apples with 81.32.
   - retrieval / few-shot: rebuild the index from the 2024 `train` (our retrieval method is
     DB-independent and carries over).
   - fine-tune: regenerate SFT JSONL + ORPO pairs from the 2024 train (note: 5,124 ex, ~8.8% unans
     in train vs our current set) and re-run SFT→ORPO→eval.
4. **Sanity gate before any training:** re-run the §3 execution test against the new SQLite — gold
   answerable SQL should now execute cleanly (errors ≈ 0). Only then are EX/RS meaningful.

**What carries over unchanged:** the retrieval findings (recall/precision/nDCG, logreg q_tag) and the
whole fine-tuning machinery (SFT loss 0.066, ORPO abstention ~0.99). We're re-pointing them at a
valid benchmark, not rebuilding them.

**If instead you want the EHRSQL-2022 MIMIC-III benchmark** (the data we currently hold): that needs
an EHRSQL-2022-style MIMIC-III SQLite (the value-shuffled mirror is on the authors' Google Drive
*without* credentialing; full MIMIC-III would need PhysioNet). But this is **not** the RS(10)=0.8132
leaderboard — it's a different, older benchmark. Given the stated goal, Path A above is the match.

---

## 7. Claim ledger (verified) + sources

- EHRSQL-2024 uses **MIMIC-IV Demo v2.2** (open, no credentialing); `db_id:"mimic_iv"`; prebuilt
  `mimic_iv.sqlite` shipped — README + paper Table 1 + preprocess scripts. **VERIFIED**
- EHRSQL-2024 splits 5,124 / 1,163 / 1,167; unans 450 / 232 / 233 — counted from `label.json`.
  **VERIFIED**
- RS(10)=81.32 = LG AI Research & KAIST, **test** set; ABSTAIN-ALL=20.0; RS(10) is the official
  metric — overview paper Table 3 / §4.1. **VERIFIED**
- 2024 schema synthesizes `dob = anchor_year − anchor_age`, omits `race`/`ethnicity`, keeps `cost`
  and `row_id` — `preprocess_db_mimic_iv.py` + `tables.json`. **VERIFIED**
- EHRSQL-2022 = MIMIC-III v1.4 + eICU v2.0 (credentialed; shuffled SQLite on Drive) — arXiv:2301.07695
  + repo layout. **VERIFIED**

Sources: ACL Anthology [2024.clinicalnlp-1.62](https://aclanthology.org/2024.clinicalnlp-1.62) ·
arXiv [2405.06673](https://arxiv.org/abs/2405.06673) ·
[glee4810/ehrsql-2024](https://github.com/glee4810/ehrsql-2024) ·
arXiv [2301.07695](https://arxiv.org/abs/2301.07695) ·
[glee4810/EHRSQL](https://github.com/glee4810/EHRSQL) ·
MIMIC-IV Demo [physionet.org/content/mimic-iv-demo/2.2](https://physionet.org/content/mimic-iv-demo/2.2/) ·
Codabench [comp/1889](https://www.codabench.org/competitions/1889) ·
task site [sites.google.com/view/ehrsql-2024](https://sites.google.com/view/ehrsql-2024).

---

## Appendix — reproduce the EDA
```bash
python scratchpad/eda_recon.py        # data tree, record schema, counts
python scratchpad/eda_schema_diff.py  # tables.json(mimic_iii) vs mimic_iv_demo.db + exec sample
python scratchpad/eda_test_focus.py   # full answerable-test execution + MIMIC-III-only table refs
```
