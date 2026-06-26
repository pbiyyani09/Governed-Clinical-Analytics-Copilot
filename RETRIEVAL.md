> **Major update (supervised template classification).** The single biggest lever
> is not the embedding model — it is recognizing that EHRSQL's `q_tag` field is a
> **question-template label**, which makes example selection a *supervised
> classification* problem, not unsupervised retrieval. See
> **"Supervised template classification (the q_tag insight)"** at the bottom — a
> logreg+bi-encoder gate hits **hit@2 0.866 / P@2 0.811** on the q_tag oracle,
> beating both pure logreg (0.837, flat recall) and the bi-encoder (0.823). The
> earlier sections used the stricter `tag` oracle (3,282 classes); read them with
> that denominator in mind.

# Retrieval-Mechanism Study — EHRSQL NL→SQL few-shot example selection

**Branch:** `gemma_dev_rebased`
**Question:** what is the *right* retrieval mechanism for selecting few-shot
(question, SQL) exemplars, and how high can we push retrieval quality before
spending any GPU on training?

Everything here is **retrieval-only** — no model training was run.

---

## Setup

- **Corpus:** 9,318 answerable train (question, SQL) pairs.
- **Queries:** 1,198 answerable test questions carrying an EHRSQL `tag` template.
- **Relevance oracle:** a corpus item is *relevant* to a query iff they share the
  same `tag` (the masked question template). This is the signal the prior
  `rag_eval.py` used; here it is graded into full recall, precision, MRR, nDCG.
- **Ceiling:** only **86.5%** of test queries have ≥1 same-template train example
  (162 queries are unhittable), so the maximum achievable hit-rate is **0.865**,
  not 1.0. Median same-template corpus count per covered query = 3 (mean 7.2).
- **Harness:** `src/ehrcopilot/eval/retrieval_bench.py` (self-contained, no
  generation model). Validated: it reproduces the prior BM25 baseline exactly
  (MRR 0.3518, P@2 0.2250, hit@2 0.3456).

## Dimensions ablated

| Dimension | Variants |
|---|---|
| **Method** | BM25 (lexical) · dense (SLM embedding) · hybrid (BM25+dense, RRF k=60) |
| **Embedding model** | bge-large-en-v1.5 · mxbai-embed-large · arctic-embed-l-v2.0 · e5-large-v2 · nomic-embed-v1.5 · Qwen3-Embedding-0.6B/4B |
| **Representation** | `q` (question) · `q_sql` (question + SQL-skeleton, the prior design) · `mqs` (masked question — DAIL-SQL) |
| **Index** | flat exact (faiss `IndexFlatIP`) · HNSW (`IndexHNSWFlat`) |
| **Metrics** | hit_rate@K, graded recall@K, precision@K, MRR, nDCG@K for K∈{1,2,3,5,10,20,50,100,500,1000} |

Models embedded in **isolated subprocesses** (one CUDA context each) so a single
model's device-side assert can't poison the run.

---

## Headline results (full 1,198 queries, flat index)

Ranked by MRR. (See `tests/evalgen/retrieval_bench_full.json` and `retrieval_bench_mqs.json`.)

| config | hit@2 | hit@10 | P@2 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **dense · bge-large · mqs** | **0.4775** | 0.6753 | **0.3485** | **0.4846** | 0.4126 |
| dense · arctic-l · mqs | 0.4633 | 0.6561 | 0.3285 | 0.4640 | 0.3877 |
| hybrid · bge-large · mqs | 0.4508 | 0.6669 | 0.3159 | 0.4624 | 0.3628 |
| hybrid · e5-large · q | 0.4182 | 0.6578 | 0.2867 | 0.4303 | 0.3383 |
| hybrid · bge-large · q | 0.4057 | 0.6344 | 0.2817 | 0.4244 | 0.3283 |
| hybrid · qwen3-4b · q | 0.3840 | 0.6077 | 0.2638 | 0.3986 | 0.3068 |
| hybrid · bge-large · **q_sql** (prior design) | 0.3907 | 0.6277 | 0.2667 | 0.4079 | 0.3179 |
| **bm25 · q_sql** (prior default) | 0.3072 | 0.5250 | 0.2020 | 0.3321 | 0.2336 |
| bm25 · q (original baseline) | 0.3456 | 0.5693 | 0.2250 | 0.3614 | 0.2583 |

**Best vs original baseline: MRR 0.4846 vs 0.3614 (+34%), hit@2 0.4775 vs 0.3456 (+38%).**

---

## Findings

1. **Representation is the biggest lever: MQS > q > q_sql.** Masking surface
   literals in the question (DAIL-SQL "Masked-Question-Similarity") beat the plain
   question for *every* model. The prior design — indexing `question + SQL-skeleton`
   — was consistently the **worst** representation: appending the SQL skeleton
   *dilutes* the question signal that the template oracle rewards. This alone
   explains much of the prior retriever's weakness.

2. **Embedding prefixes matter.** bge/e5/arctic want an instruction prefix on the
   query side; the prior harness applied none. Adding them measurably improved
   recall (the numbers above all use the correct prefixes).

3. **Classic BERT-family retrievers beat the LLM-based embedders here.**
   bge-large (335M), arctic-l, and e5-large lead. Qwen3-Embedding-4B (4B) and
   0.6B did **not** win on this clinical-question-template task despite their
   MTEB standing — bigger ≠ better for short template matching.

4. **Method depends on representation.** For the noisy plain-question key, **hybrid
   (BM25+dense) beats dense-alone** (BM25 adds complementary lexical signal). But
   once the key is the clean **masked question, dense-alone is best** — BM25 fusion
   slightly dilutes an already template-aligned signal.

5. **HNSW ≈ flat exact.** `IndexHNSWFlat` matched exact search within noise
   (MRR 0.4306 vs 0.4317) on the 9.3k corpus — so HNSW is safe when scale demands
   it; flat is exact and fast enough here.

6. **Deep recall** (graded recall@K): the top dense/hybrid configs reach ~0.74 at
   K=100 and ~0.85 at K=1000 (vs BM25 0.63 / 0.82) — i.e. ~85% of all
   same-template examples are within the top-1000.

---

## Recommendation (wired into `harness.py`)

Default few-shot retriever (`build_few_shot_retriever`):
**hybrid (BM25 + bge-large-en-v1.5, RRF) over a masked-question index, with prefixes.**

- This is one row below the absolute oracle winner (`dense · bge-large · mqs`),
  chosen deliberately: BM25 contributes exact-match signal on **medical entities**
  (drug names, ICD codes, lab labels) that the template oracle does not reward but
  end-to-end SQL generation does benefit from (clinical-RAG literature). It keeps
  the +MQS / +prefix / drop-SQL-skeleton wins (the bulk of the gain) while staying
  robust. `--retrieval-mode embed` selects the oracle-topping dense-only variant.

Validated end-to-end: the wired retriever scores hit@2 = 0.57 on the first 400
test queries (vs the easier-cluster nature of that slice; full-set ≈ 0.45).

**Caveat / next step:** the template oracle is a proxy. The definitive test is
end-to-end execution accuracy (EX) with a generation pass — to be run once the
generation model is settled. RAGAS Faithfulness / Answer-Relevance
(`src/ehrcopilot/eval/rag_ragas.py`) likewise need a generation pass; Context
Precision / Context Recall are already covered by the precision/recall above.

## Models that could not be evaluated

| Model | Reason |
|---|---|
| gte-large-en-v1.5, gte-Qwen2-1.5B | custom RoPE / `rope_theta` incompatible with transformers 5.5 (device-side assert) |
| SFR-Embedding-Code-2B (code-specialized) | `HybridCache` import removed in transformers 5.5 |
| embeddinggemma-300m | evaluated (license accepted): q_tag P@2 0.710 / hit@2 0.812 — competitive but slightly below bge-large (0.712 / 0.823); does not change the recommendation |
| Qwen3-Embedding-4B `q_sql` | OOM at batch 64 (got `q`; lower than bge anyway) |

To include the code-specialized / gte models, run them in a side venv pinned to
an older `transformers`; Qwen3 already covers the code-aware-embedding angle and
did not win, so this is low priority.

## How to reproduce

```bash
bash scripts/run_retrieval_ablation.sh q q_sql      # isolated model ablation
bash scripts/run_retrieval_ablation.sh mqs          # masked-question pass
# scoring reuses cached embeddings; HNSW: add --index hnsw to retrieval_bench
```

---

## Supervised template classification (the q_tag insight)

A colleague reported a **logistic-regression** retriever reaching precision@2 /
recall@2 ≈ 0.92 — far above the bi-encoder numbers above. Investigating this
exposed a framing error in the first study and a better method.

### Two things were wrong in the first pass

1. **Wrong oracle.** The first study scored against the `tag` field — which has
   **3,282** unique values in train (it concatenates question + time + operation
   sub-templates). The colleague scored against **`q_tag`** — the *question*
   template alone: **167** unique values, **100% test coverage**. The same
   bi-encoder scores hit@2 **0.823** on `q_tag` vs **0.477** on `tag`. Most of the
   "0.92 vs 0.48" gap was the denominator, not the method.
2. **Wrong problem class.** `q_tag` is a **label that exists in the training data**.
   That makes example selection a *supervised classification* problem (predict the
   template, return its examples), not unsupervised cosine similarity. This is the
   EHRSQL-2024 / KU-DMIS "question templatization" idea and the Meta-Sel
   (arXiv:2602.12123) TF-IDF-logreg ICL-selection result. The first round of
   research surveyed unsupervised retrievers and missed it.

### Head-to-head on the q_tag oracle (1,198 queries)

| method | P@2 | hit@2 | hit@5 | hit@10 | hit@20 |
|---|---|---|---|---|---|
| TF-IDF + LogReg (q_tag) | 0.837 | 0.837 | 0.837 | 0.837 | 0.837 |
| bi-encoder (bge / MQS) | 0.712 | 0.823 | 0.917 | 0.967 | 0.986 |
| fusion (z-norm logreg + cosine) | **0.851** | 0.856 | 0.860 | 0.862 | 0.872 |
| **GATE (logreg-if-confident → bi)** | 0.811 | **0.866** | 0.913 | 0.949 | 0.967 |

- **LogReg** has high precision but **flat recall** — it commits to one predicted
  template and cannot recover from a wrong top-1 (hit@10 == hit@2 == 0.837). A
  quick TF-IDF logreg lands at 0.837; tuned features (char n-grams, entity masking,
  embedding inputs) reach the reported ~0.92.
- **bi-encoder** has lower precision but **climbs to hit@10 0.967** — it hedges
  across templates.
- **GATE** = best of both: if the classifier is confident (max prob > θ) return the
  predicted template's examples ranked by bi-encoder similarity, else fall back to
  pure bi-encoder. Beats both on hit@2 (0.866) while keeping recall (hit@10 0.949).
  It also lifts the finer `tag` oracle (hit@2 0.477 → 0.504).

Implemented as `src/ehrcopilot/eval/template_retriever.py`; selectable via
`harness --retrieval-mode classifier`. Validated end-to-end (hit@2 0.895 on the
first 400 q_tag queries).

### Honest caveats

- **`q_tag` is a coarser relevance target than `tag`.** Same `q_tag` ⇒ same question
  type ⇒ usually (not always) the same SQL skeleton; the finer `tag` also pins the
  time/operation variation. Which target best predicts end-to-end EX is unknown
  until a generation pass is run. Report q_tag and tag numbers side by side; do not
  quote the q_tag number alone as "retrieval precision".
- **Closed-set dependence.** `q_tag` has 100% coverage on this mimic_iii test split,
  so abstention is rarely triggered. The EHRSQL-2024 (mimic_iv) split deliberately
  holds out ~25% unseen templates; there the classifier would mis-route confidently
  and the bi-encoder fallback (the gate's θ) is what prevents catastrophic failure.
- The classifier predicts q_tag **from the question text** (it does not read the
  gold q_tag at test time), so this is a legitimate "given a question, find
  same-template examples" result — but it is measuring template-classification
  accuracy on a closed set, which is an easier task than open retrieval.

**Recommended default:** the GATE classifier hybrid (`--retrieval-mode classifier`).
It dominates pure logreg (recall) and pure bi-encoder (precision) and degrades
gracefully via the confidence gate. Confirm with an end-to-end EX pass before
finalizing.

---

## Reranking & the retrieve→top-5 pipeline (for a Gemma generator)

Goal: feed a Gemma SQL generator the best 5 exemplars (Gemma 3's 1024-token
sliding-window attention favors few, tightly-relevant chunks). So **P@5** — how
many of the 5 share the query's template — is the metric that matters.

| top-5 method | q_tag P@5 | hit@2 | hit@5 | tag P@5 |
|---|---|---|---|---|
| bi-encoder (bge / MQS) | 0.672 | 0.823 | 0.917 | 0.269 |
| cross-encoder rerank (bge-reranker-v2-m3, Q-Q, N=10) | 0.669 | 0.815 | 0.922 | 0.266 |
| **classifier FUSION** (z(cosine)+z(logreg-q_tag)) | **0.850** | 0.856 | 0.860 | 0.295 |
| classifier GATE (logreg-if-confident → bi) | 0.791 | 0.866 | 0.913 | 0.294 |

**A generic cross-encoder reranker does NOT help here** — bge-reranker-v2-m3 (the
strongest QA-pair cross-encoder) reranking the top-10/20 by question-question
relevance left hit@5 flat and slightly *hurt* hit@2. It was trained on passage
relevance, not EHRSQL template identity, so it reorders by surface similarity that
is *less* template-aligned than the MQS embedding already is.

**The effective "reranker" for this task is the supervised template classifier.**
The FUSION reranker (rank every candidate by z-normalised bi-encoder cosine +
z-normalised logreg probability of its q_tag) puts **~85% same-template exemplars
in the top-5** (P@5 0.850) vs the bi-encoder's 0.672 — a much cleaner few-shot
prompt for the generator. It is the default of `build_classifier_retriever`
(`top_k=5, method="fusion"`); `method="gate"` trades a little P@5 for the best
hit@2 (0.866).

### Recommended generation pipeline
1. **Retrieve** top-N (bi-encoder, bge-large + MQS; recall@10 on q_tag = 0.97).
2. **Rerank/select** with the FUSION classifier → top-5 (P@5 ≈ 0.85).
3. **Generate** SQL with a **Gemma** model. All gated Gemma generators are
   accessible with the configured HF token: `gemma-3-4b/12b/27b-it`,
   `codegemma-7b-it`. Recommended Qwen2.5-Coder-7B analog: **`gemma-3-12b-it`**
   (128K context, ~8GB 4-bit). codegemma-7b is code-tuned but only 8K context.

The decisive number — end-to-end execution accuracy (EX) — still needs a Gemma
generation pass over the test set (next step).

---

## End-to-end EX with a Gemma generator (the honest result)

Wired the full pipeline — retrieve → fusion-top-5 → **gemma-3-12b-it** (base, no
fine-tuning) generates SQL → execute → EX/RS. (Required fixes: a SQL-execution
watchdog timeout, Unsloth `FastModel` loading for Gemma 3, and **stripping
markdown code fences** from generated SQL — without the last one EX was a silent
0 because Gemma wraps correct SQL in ```sqlite fences.)

3-way comparison, gemma-3-12b-it, 75-question stratified subset (50 answerable +
25 unanswerable), varying ONLY the retriever:

| config | EX | RS(10) | correct | wrong-abstentions | wrong_on_unans | p50 |
|---|---|---|---|---|---|---|
| zero-shot | 0.42 | −0.707 | 21/50 | 17 | 9 | 6.1s |
| bi-encoder hybrid (top-2) | 0.42 | −0.853 | 21/50 | 10 | 10 | 9.5s |
| classifier fusion (top-5) | 0.42 | −0.707 | 21/50 | 5 | 9 | 13.6s |

**Retrieval improves the retrieval metrics but NOT base-model EX.** The few-shot
examples are clearly used — they cut over-abstention from 17→5 (the model attempts
many more answerable questions with examples in context) — but those extra attempts
produce *wrong* SQL, so EX is flat at 0.42. The end-to-end bottleneck is the base
generator's SQL correctness on hard (multi-join / nested) questions, not the
quality of the retrieved examples.

**Implications**
- The retrieval improvements (MQS, q_tag classifier/fusion, P@5 0.85) are real and
  correct, but on their own they do not move EX with an *untuned* generator.
- The lever for EX is **fine-tuning the Gemma generator** (SFT + abstention ORPO),
  exactly the original pipeline. RS(10) is dominated by un-calibrated abstention
  (−0.7) — a training problem, not a retrieval one.
- Retrieval is expected to pay off *after* fine-tuning (as it did for the Qwen
  campaign: ORPO v3 + repair + RAG added +66 correct answers over no-RAG). It
  should be re-evaluated on the fine-tuned Gemma model.
- Caveat: 50 answerable questions is a small sample (95% CI ≈ ±0.14); the exact tie
  at 21 is partly coincidence, but the abstention/latency deltas confirm the
  retrieval is applied and EX is genuinely flat at this scale.

Per-config results: `tests/evalgen/gemma_ex_{zeroshot,hybrid,classifier}.json`.
