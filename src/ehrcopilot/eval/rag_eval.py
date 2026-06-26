"""RAG retrieval quality evaluation (RAGAS-style).

Measures Context Recall@K and Context Precision@K using the EHRSQL `tag` field
as the ground-truth relevance signal. The `tag` field contains the abstract
question template (e.g. "what is the intake method of {drug_name}?"), shared
across all paraphrases of the same query type — a clean, free relevance label
that requires no LLM-as-judge.

Metrics computed:
  - Hit Rate@K  (= Context Recall@K): any relevant result in top-K
  - Precision@K: fraction of top-K results that are relevant
  - MRR:        mean reciprocal rank of the first relevant result

Run (BM25 baseline):
    python -m ehrcopilot.eval.rag_eval --mode bm25

Run (hybrid comparison):
    python -m ehrcopilot.eval.rag_eval --mode hybrid

Run both and compare:
    python -m ehrcopilot.eval.rag_eval --mode all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# SQL skeleton helper (Query Capsule approach)
# ---------------------------------------------------------------------------

def sql_skeleton(sql: str) -> str:
    """Strip concrete values from SQL to expose the structural template.

    Replaces string literals with '?' and integers with ?, leaving SQL keywords,
    table/column names, and structural operators intact. This lets the embedding
    model cluster survival-rate queries together regardless of disease name.
    """
    s = sql.lower()
    s = re.sub(r"'[^']*'", "'?'", s)   # string literals → '?'
    s = re.sub(r'\b\d+\b', '?', s)      # integers → ?
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------------------------------------------------------------------------
# Retriever factories — return (question) -> list[int] (top-K train indices)
# ---------------------------------------------------------------------------

def _build_bm25_retriever(train_examples: list[dict], max_k: int = 10) -> Callable[[str], list[int]]:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    corpus = [ex["question"].lower().split() for ex in train_examples]
    bm25 = BM25Okapi(corpus)

    def retrieve(question: str) -> list[int]:
        scores = bm25.get_scores(question.lower().split())
        import numpy as np
        return list(map(int, (-scores).argsort()[:max_k]))

    return retrieve


def _build_embed_retriever(
    train_examples: list[dict],
    model_name: str,
    embed_cache: Path | None,
    max_k: int = 10,
    mode: str = "embed",  # "embed" or "hybrid"
) -> Callable[[str], list[int]]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")

    if embed_cache is None:
        safe_name = model_name.replace("/", "_").replace("-", "_").lower()
        embed_cache = Path("data/ehrsql") / f"train_embeddings_{safe_name}.npy"

    if embed_cache.exists():
        print(f"Loading cached embeddings from {embed_cache}")
        train_embeds = np.load(str(embed_cache))
    else:
        print(f"Computing embeddings with {model_name} on {device} ...")
        embed_model_tmp = SentenceTransformer(model_name, device=device)
        index_texts = [
            ex["question"] + " " + sql_skeleton(ex.get("query") or "")
            for ex in train_examples
        ]
        train_embeds = embed_model_tmp.encode(
            index_texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        embed_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(embed_cache), train_embeds)
        print(f"Saved embeddings to {embed_cache}")

    embed_model = SentenceTransformer(model_name, device=device)
    n = len(train_examples)

    if mode == "embed":
        def retrieve(question: str) -> list[int]:
            q_vec = embed_model.encode([question], normalize_embeddings=True)
            cosine = (train_embeds @ q_vec.T).squeeze()
            return list(map(int, (-cosine).argsort()[:max_k]))
        return retrieve

    # hybrid RRF
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    corpus = [ex["question"].lower().split() for ex in train_examples]
    bm25 = BM25Okapi(corpus)

    def retrieve_hybrid(question: str) -> list[int]:
        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_order = (-bm25_scores).argsort()
        bm25_ranks = np.empty(n, dtype=np.float32)
        bm25_ranks[bm25_order] = np.arange(1, n + 1, dtype=np.float32)

        q_vec = embed_model.encode([question], normalize_embeddings=True)
        cosine = (train_embeds @ q_vec.T).squeeze()
        embed_order = (-cosine).argsort()
        embed_ranks = np.empty(n, dtype=np.float32)
        embed_ranks[embed_order] = np.arange(1, n + 1, dtype=np.float32)

        rrf = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
        return list(map(int, (-rrf).argsort()[:max_k]))

    return retrieve_hybrid


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    train_examples: list[dict],
    test_examples: list[dict],
    retriever_fn: Callable[[str], list[int]],
    k_values: list[int] = [1, 2, 3, 5, 10],
) -> dict:
    """Compute Context Recall@K, Precision@K, Hit Rate@K, and MRR.

    Uses the `tag` field as the relevance signal: a retrieved training example
    is relevant if its `tag` matches the test question's `tag`.
    """
    import numpy as np

    tags_train = [ex.get("tag", "") for ex in train_examples]

    # Only evaluate answerable test questions that have a tag
    test_eval = [
        ex for ex in test_examples
        if not ex.get("is_impossible") and ex.get("tag")
    ]

    max_k = max(k_values)
    results = {k: {"precision": 0.0, "recall": 0.0} for k in k_values}
    mrr_total = 0.0
    n = 0

    for ex in test_eval:
        test_tag = ex["tag"]
        top_idx = retriever_fn(ex["question"])          # list[int], len=max_k
        retrieved_tags = [tags_train[i] for i in top_idx[:max_k]]

        # MRR
        for rank, t in enumerate(retrieved_tags, 1):
            if t == test_tag:
                mrr_total += 1.0 / rank
                break

        for k in k_values:
            top_k = retrieved_tags[:k]
            n_relevant = sum(1 for t in top_k if t == test_tag)
            results[k]["precision"] += n_relevant / k
            results[k]["recall"] += float(n_relevant > 0)

        n += 1

    if n == 0:
        return {"error": "no evaluable test examples", "n_evaluated": 0}

    for k in k_values:
        results[k]["precision"] /= n
        results[k]["recall"] /= n
        # recall == hit_rate here (binary: did we hit at all?)
        results[k]["hit_rate"] = results[k]["recall"]

    return {
        "per_k": results,
        "mrr": mrr_total / n,
        "n_evaluated": n,
    }


def print_results(label: str, metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return
    print(f"  N evaluated: {metrics['n_evaluated']}")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print()
    print(f"  {'K':>4}  {'Recall@K':>10}  {'Precision@K':>12}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*12}")
    for k, v in sorted(metrics["per_k"].items()):
        print(f"  {k:>4}  {v['recall']:>10.4f}  {v['precision']:>12.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG retrieval quality evaluation (RAGAS-style)")
    parser.add_argument(
        "--train", default="data/ehrsql/ehrsql/mimic_iii/train.json",
        help="Path to EHRSQL train split",
    )
    parser.add_argument(
        "--test", default="data/ehrsql/ehrsql/mimic_iii/test.json",
        help="Path to EHRSQL test split",
    )
    parser.add_argument(
        "--mode", default="all", choices=["bm25", "embed", "hybrid", "all"],
        help="Retrieval mode(s) to evaluate",
    )
    parser.add_argument(
        "--embed-model", default="BAAI/bge-large-en-v1.5",
        help="HuggingFace embedding model name (default: BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--embed-cache", default=None,
        help="Path to cached .npy embeddings (auto-derived if not set)",
    )
    parser.add_argument(
        "--k", default="1,2,3,5,10",
        help="Comma-separated K values to evaluate (default: 1,2,3,5,10)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write results JSON to this file",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    embed_cache = Path(args.embed_cache) if args.embed_cache else None

    print(f"Loading train split: {args.train}")
    with open(args.train) as f:
        train_examples = json.load(f)
    # Only use answerable training examples with SQL
    train_answerable = [
        ex for ex in train_examples
        if not ex.get("is_impossible")
        and ex.get("query", "").strip().lower() not in ("", "null", "none", "n/a")
    ]
    print(f"  {len(train_answerable)} answerable training examples with SQL")

    print(f"Loading test split: {args.test}")
    with open(args.test) as f:
        test_examples = json.load(f)
    print(f"  {len(test_examples)} test examples loaded")

    all_results: dict[str, dict] = {}
    modes = ["bm25", "embed", "hybrid"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n--- Building {mode.upper()} retriever ---")
        if mode == "bm25":
            retriever = _build_bm25_retriever(train_answerable, max_k=max(k_values))
        else:
            retriever = _build_embed_retriever(
                train_answerable,
                model_name=args.embed_model,
                embed_cache=embed_cache,
                max_k=max(k_values),
                mode=mode,
            )

        print(f"Evaluating {mode.upper()} retrieval ...")
        metrics = evaluate_retrieval(train_answerable, test_examples, retriever, k_values)
        print_results(f"{mode.upper()} Retrieval", metrics)
        all_results[mode] = metrics

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {out}")

    # Print comparison table if multiple modes
    if len(modes) > 1 and not any("error" in v for v in all_results.values()):
        print(f"\n{'='*60}")
        print("  COMPARISON TABLE")
        print(f"{'='*60}")
        header = f"  {'Mode':>8}"
        for k in sorted(k_values):
            header += f"  {'Recall@'+str(k):>12}  {'Prec@'+str(k):>10}"
        header += f"  {'MRR':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for mode in modes:
            r = all_results[mode]
            row = f"  {mode:>8}"
            for k in sorted(k_values):
                row += f"  {r['per_k'][k]['recall']:>12.4f}  {r['per_k'][k]['precision']:>10.4f}"
            row += f"  {r['mrr']:>8.4f}"
            print(row)


if __name__ == "__main__":
    main()
