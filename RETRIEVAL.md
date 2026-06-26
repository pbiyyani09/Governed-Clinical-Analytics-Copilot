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
| embeddinggemma-300m | HF repo gated — license not accepted on the configured token |
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
