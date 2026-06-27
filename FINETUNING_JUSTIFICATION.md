# Why fine-tuning is the last resort — and now the only lever left

**Branch:** `gemma_dev_rebased` · **Generator:** gemma-3-12b-it (base, no task FT) ·
**Date:** 2026-06-27

This report documents the systematic elimination of every inference-time
alternative to fine-tuning for EHRSQL (clinical NL→SQL over MIMIC-IV-Demo). Each
lever was implemented and measured on the **same 75-question stratified subset**
(50 answerable + 25 unanswerable) with the same gemma-3-12b-it generator, varying
only the technique. The conclusion is that **fine-tuning is the only remaining
mechanism** to (1) raise execution accuracy past a hard ~0.42 ceiling and (2)
achieve *calibrated abstention* (without which RS(10) stays negative).

---

## The metric and the target

```
EX     = fraction of ANSWERABLE questions whose SQL executes to the gold result
RS(10) = (correct_SQL + correct_abstentions − 10·SQL_on_unanswerable) / total
```
RS(10) penalizes a hallucinated SQL on an unanswerable question by −10. **One
hallucination wipes out ten correct answers.** EHRSQL-2024 leaderboard top =
RS(10) 0.81; competitive ≥ 0.60. Our target is positive, ideally ≥ 0.60.

Base gemma-3-12b-it: **EX ≈ 0.42, RS(10) ≈ −0.7.** It answers ~42% of answerable
questions but hallucinates SQL on ~36% of unanswerable ones (no abstention).

---

## Levers tried and eliminated

### Lever 1 — Retrieval (few-shot example selection)
A full retrieval study (see `RETRIEVAL.md`) improved retrieval *metrics* hugely
(MQS + bge + the q_tag classifier/fusion: hit@2 0.48→0.87, P@5 0.85). But the
end-to-end 3-way comparison (same 75q) showed **no EX movement**:

| retriever | EX | RS(10) | wrong-on-unans |
|---|---|---|---|
| zero-shot (no examples) | 0.42 | −0.71 | 9/25 |
| bi-encoder hybrid | 0.42 | −0.85 | 10/25 |
| classifier fusion top-5 | 0.42 | −0.72 | 9/25 |

Better examples are retrieved and used (over-abstention falls), but the base model
cannot convert them into correct SQL on hard questions. **Retrieval alone: no EX gain.**

### Levers 2–5 — Context engineering, CoT, agentic loops, abstention prompting
Implemented in `src/ehrcopilot/eval/prompting_experiments.py` (results:
`tests/evalgen/prompting_experiments.json`), cumulative, same 75q:

| config | EX | RS(0) | RS(10) | correct/50 | wrong-on-unans/25 | abstained-on-answerable |
|---|---|---|---|---|---|---|
| base (fusion-5, direct) | 0.40 | 0.48 | −0.72 | 20 | 9 | 5 |
| + context engineering¹ | **0.34** | 0.29 | −2.37 | 17 | **20** | 1 |
| + chain-of-thought² | 0.40 | 0.31 | −2.63 | 20 | **22** | 0 |
| + agentic loop³ | 0.42 | 0.36 | −2.17 | 21 | 19 | 25 |
| + abstention verify⁴ | 0.42 | 0.44 | −1.29 | 21 | 13 | 27 |

¹ Enriched schema: SQLite dialect rules, MIMIC join/value hints (itemid→d_items,
icd_code→d_icd_*), explicit schema-use + abstain rules.
² "reason step-by-step → final ```sql```" decomposition.
³ Execution-guided self-correction: feed back SQL errors AND empty-results, up to
3 rounds, with reasoning.
⁴ Post-hoc LLM verification pass ("is this answerable & schema-grounded?") → [ABSTAIN].

**What the data shows**

1. **EX is pinned at 0.40–0.42.** No technique broke past it. The best (agentic,
   abstain) added exactly **+1 correct answer** (20→21) at ~4× the latency.
   Context engineering *lowered* EX to 0.34.
2. **The "smarter" prompts destroy abstention.** Context-engineering and CoT push
   the model to reason its way into producing SQL for nearly every question —
   wrong-on-unanswerable jumps 9 → 20 → 22 of 25. RS(10) collapses to −2.4 / −2.6.
3. **Prompted abstention is fundamentally uncalibrated.** The verification pass
   was the targeted fix; it over-abstains on **27/50 answerable** questions (throwing
   away correct answers) while *still* letting **13/25 unanswerable** through. It
   cannot tell answerable from unanswerable — it does both errors at once. Best
   prompted RS(10) = −1.29, still deeply negative and *worse* than the plain base
   prompt's −0.72.
4. **Every RS(10) is negative.** Across 3 retrieval variants and 5 prompting
   variants — 8 distinct inference-time configurations — not one produced a
   positive RS(10).

---

## The two walls

**Wall A — EX ceiling.** A base 12B model on an opaque clinical schema (itemid,
hadm_id, ICD/LOINC/DRG codes, MIMIC temporal logic) tops out at ~0.42 EX here.
Retrieval, decomposition, and self-correction each add ≤ +1 answer. Closing the
gap to leaderboard EX (~0.77) requires the model to *learn* MIMIC-IV SQL patterns
— i.e., training.

**Wall B — abstention is uncalibrated and unfixable by prompting.** RS(10) is
dominated by hallucinations on unanswerable questions. Reliable abstention needs a
calibrated confidence signal the base model does not have; prompting either
under-abstains (base/ctx/cot) or over-abstains-and-still-leaks (verify).

---

## Literature corroboration (full survey by the research agent)

- **Every EHRSQL-2024 team with RS(10) > 40 used fine-tuning.** The one
  prompting-only team (Project PRIMUS) scored **RS(10) = −713**.
- **ProbGate** showed mechanistically that probability-threshold abstention only
  works *after* fine-tuning (base log-probs are not calibrated): RS −191 → +74.
- Prompting frameworks (DIN-SQL, MAC-SQL, CHASE-SQL) lift EX ~+7–13 pp on Spider/
  BIRD with GPT-4, but their realistic ceiling for a 12B base model on a
  domain-specific schema is ~0.55–0.65 EX — still short of the target, and they do
  not solve calibrated abstention.
- Qwen2.5-Coder-7B fine-tuned on Spider+BIRD scores only ~31% EX on EHRSQL
  out-of-domain → clinical schemas require *domain* fine-tuning, not just a strong
  SQL model.

Our empirical results are *consistent with the low end* of these predictions and,
on abstention, match the literature exactly: prompting cannot do it.

---

## Verdict

Retrieval, context engineering, chain-of-thought, agentic self-correction, and
prompted abstention have all been implemented and measured. **None raises EX above
0.42 and none yields a positive RS(10).** Both failure modes — SQL correctness on
hard clinical queries and calibrated abstention — are exactly what supervised
fine-tuning addresses, and what every competitive EHRSQL system relied on.

**Fine-tuning is therefore the only remaining lever, and is now justified as a last
resort.**

### Plan
1. **QLoRA SFT** of gemma-3-12b-it on the EHRSQL training set (answerable→SQL,
   unanswerable→[ABSTAIN]) — targets Wall A (learn MIMIC-IV SQL patterns) and seeds
   abstention.
2. **Abstention-ORPO** on preference pairs (chosen=[ABSTAIN] / gold SQL,
   rejected=hallucination) — targets Wall B (calibrate abstention), the move that
   took EHRSQL teams from negative to RS > 60.
3. **Re-run the retrieval + inference-time comparison on the fine-tuned model** —
   retrieval is expected to finally pay off post-FT (as RAG did for the tuned Qwen
   campaign: +66 correct answers).

### Caveats
- Subset n = 75 (50 answerable); 95% CI on EX ≈ ±0.14. The *direction* (flat EX,
  negative RS across all 8 configs) is unambiguous; exact values are indicative.
- Code: `prompting_experiments.py` (sweep), `template_retriever.py` (retrieval),
  `harness.py` (eval). Raw results in `tests/evalgen/`.
