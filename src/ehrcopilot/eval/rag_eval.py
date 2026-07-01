"""RAG retrieval quality evaluation — EHRSQL 2024 MIMIC-IV.

Metrics:
  Hit@K       -- fraction of queries with ≥1 relevant doc in top-K (binary recall)
  Recall@K    -- fraction of ALL relevant docs retrieved in top-K (strict recall)
  Precision@K -- fraction of top-K docs that are relevant
  nDCG@K      -- normalised discounted cumulative gain at K
  MRR         -- mean reciprocal rank of first relevant result

Two relevance modes:
  skeleton  -- exact SQL skeleton match (ceil 71.1% on MIMIC-IV test)
  medium    -- same table-set + same SQL operation type (ceil 89.8%) [default]

Four retrieval modes:
  bm25     -- BM25Okapi on question tokens
  embed    -- BAAI/bge-large-en-v1.5, indexed on question + sql_skeleton
  hybrid   -- BM25 + embed via Reciprocal Rank Fusion (k=60)
  template -- LogReg on medium labels + cosine within predicted class

Two-stage reranking:
  --rerank-model MODEL  -- cross-encoder model name (sentence-transformers CrossEncoder)
  --retrieve-k K        -- number of first-stage candidates before reranking [default: 100]

Run:
    python -m ehrcopilot.eval.rag_eval --mode hybrid --train-aug data/ehrsql2024/mimic_iv/train_aug
    python -m ehrcopilot.eval.rag_eval --mode hybrid --train-aug ... --rerank-model cross-encoder/ms-marco-MiniLM-L-12-v2
    python -m ehrcopilot.eval.rag_eval --mode all --relevance skeleton   # exact-skeleton baseline
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Data loading — EHRSQL 2024 directory format (data.json + label.json)
# ---------------------------------------------------------------------------

def load_split(split_dir: str | Path) -> list[dict]:
    """Load EHRSQL 2024 split. Returns dicts with:
    {id, question, query, is_impossible, skeleton, medium}
    """
    split_dir = Path(split_dir)
    data = json.load(open(split_dir / "data.json"))["data"]
    labels: dict[str, str] = json.load(open(split_dir / "label.json"))

    examples = []
    for ex in data:
        sql = labels.get(ex["id"], "null")
        is_impossible = sql.strip().lower() in ("null", "none", "n/a", "")
        clean_sql = "" if is_impossible else sql.strip()
        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "query": clean_sql,
            "is_impossible": is_impossible,
            "skeleton": sql_skeleton(clean_sql) if clean_sql else "",
            "medium": medium_label(clean_sql) if clean_sql else "",
        })
    return examples


# ---------------------------------------------------------------------------
# Label functions
# ---------------------------------------------------------------------------

def sql_skeleton(sql: str) -> str:
    """Fine-grained label: strip concrete values, keep full structure.
    Ceiling on MIMIC-IV test: 71.1%."""
    s = sql.lower().strip()
    s = re.sub(r"'[^']*'", "'X'", s)
    s = re.sub(r'\b\d+(?:\.\d+)?\b', 'N', s)
    return re.sub(r'\s+', ' ', s).strip()


def medium_label(sql: str) -> str:
    """Coarse label: sorted table-set + SQL op-type flags.
    Ceiling on MIMIC-IV test: 89.8%.
    Format: 'table1,table2,...::AGG_GROUP_JOIN'
    """
    s = sql.lower()
    tables = sorted(set(re.findall(r'\b(?:from|join)\s+(\w+)', s)))
    is_agg = int(bool(re.search(r'\b(count|avg|sum|max|min)\b', s)))
    has_group = int('group by' in s)
    has_join = int('join' in s)
    return f"{','.join(tables)}::{is_agg}{has_group}{has_join}"


def _label_fn(relevance: str) -> Callable[[dict], str]:
    if relevance == "skeleton":
        return lambda ex: ex.get("skeleton", "")
    return lambda ex: ex.get("medium", "")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(
    retrieved_labels: list[str],
    true_label: str,
    n_relevant_in_corpus: int,
    k: int,
) -> float:
    """Binary nDCG@K.

    DCG@K: sum of 1/log2(rank+1) for each relevant doc in top-K.
    IDCG@K: best possible DCG — all top-K slots filled with relevant docs,
             capped at the total number of relevant docs in the full corpus.
    """
    gains = [1.0 if l == true_label else 0.0 for l in retrieved_labels[:k]]
    dcg = _dcg(gains)
    n_ideal = min(n_relevant_in_corpus, k)
    idcg = _dcg([1.0] * n_ideal)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_or_build_embeddings(
    train_examples: list[dict],
    model_name: str,
    embed_cache: Path,
) -> "np.ndarray":
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    if embed_cache.exists():
        arr = np.load(str(embed_cache))
        if arr.shape[0] == len(train_examples):
            print(f"  Loaded cached embeddings from {embed_cache} ({arr.shape})")
            return arr
        print(f"  Cache size mismatch ({arr.shape[0]} vs {len(train_examples)}) — rebuilding")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Computing {len(train_examples)} embeddings with {model_name} on {device} ...")
    model = SentenceTransformer(model_name, device=device)
    index_texts = [ex["question"] + " " + ex.get("skeleton", "") for ex in train_examples]
    embeds = model.encode(
        index_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )
    embed_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(embed_cache), embeds)
    print(f"  Saved embeddings to {embed_cache}")
    return embeds


# ---------------------------------------------------------------------------
# Retriever factories (return callable: question -> list[int], top-K indices)
# ---------------------------------------------------------------------------

def _build_bm25_retriever(
    train_examples: list[dict], max_k: int = 10
) -> Callable[[str], list[int]]:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    import numpy as np

    corpus = [ex["question"].lower().split() for ex in train_examples]
    bm25 = BM25Okapi(corpus)

    def retrieve(question: str) -> list[int]:
        scores = bm25.get_scores(question.lower().split())
        return list(map(int, (-scores).argsort()[:max_k]))

    return retrieve


def _build_embed_retriever(
    train_examples: list[dict],
    model_name: str,
    embed_cache: Path,
    max_k: int,
    mode: str = "embed",
) -> Callable[[str], list[int]]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    train_embeds = _load_or_build_embeddings(train_examples, model_name, embed_cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(model_name, device=device)
    n = len(train_examples)

    if mode == "embed":
        def retrieve_embed(question: str) -> list[int]:
            q = embed_model.encode([question], normalize_embeddings=True)
            cos = (train_embeds @ q.T).squeeze()
            return list(map(int, (-cos).argsort()[:max_k]))
        return retrieve_embed

    # hybrid RRF
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    bm25 = BM25Okapi([ex["question"].lower().split() for ex in train_examples])

    def retrieve_hybrid(question: str) -> list[int]:
        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_order = (-bm25_scores).argsort()
        bm25_ranks = np.empty(n, dtype=np.float32)
        bm25_ranks[bm25_order] = np.arange(1, n + 1, dtype=np.float32)

        q = embed_model.encode([question], normalize_embeddings=True)
        cos = (train_embeds @ q.T).squeeze()
        embed_order = (-cos).argsort()
        embed_ranks = np.empty(n, dtype=np.float32)
        embed_ranks[embed_order] = np.arange(1, n + 1, dtype=np.float32)

        rrf = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
        return list(map(int, (-rrf).argsort()[:max_k]))

    return retrieve_hybrid


def _build_template_retriever(
    train_examples: list[dict],
    embed_model_name: str,
    embed_cache: Path,
    classifier_cache: Path,
    relevance: str = "medium",
    max_k: int = 10,
) -> Callable[[str], list[int]]:
    import numpy as np
    import torch
    import joblib  # type: ignore[import]
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_embeds = _load_or_build_embeddings(train_examples, embed_model_name, embed_cache)

    if not classifier_cache.exists():
        print(f"  Classifier not found — training now ...")
        _train_template_classifier(train_examples, train_embeds, classifier_cache, relevance)

    clf_data = joblib.load(str(classifier_cache))
    clf = clf_data["clf"]
    label_list = clf_data["skeleton_list"]
    print(f"  Loaded classifier ({len(label_list)} classes, relevance={relevance})")

    get_label = _label_fn(relevance)
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(train_examples):
        lbl = get_label(ex)
        if lbl:
            label_to_indices[lbl].append(i)

    embed_model = SentenceTransformer(embed_model_name, device=device)

    def retrieve_template(question: str) -> list[int]:
        q_vec = embed_model.encode([question], normalize_embeddings=True)
        pred_idx = int(clf.predict(q_vec)[0])
        predicted_label = label_list[pred_idx]
        candidates = label_to_indices.get(predicted_label, [])
        if candidates:
            cand_embeds = train_embeds[candidates]
            cos = (cand_embeds @ q_vec.T).squeeze()
            if cos.ndim == 0:
                cos = cos.reshape(1)
            top = (-cos).argsort()[:max_k]
            return [candidates[int(i)] for i in top]
        cos = (train_embeds @ q_vec.T).squeeze()
        return list(map(int, (-cos).argsort()[:max_k]))

    return retrieve_template


def _build_smart_retriever(
    train_examples: list[dict],
    embed_model_name: str,
    embed_cache: Path,
    classifier_cache: Path,
    relevance: str = "medium",
    max_k: int = 10,
) -> "Callable[[str], list[int]]":
    """Probabilistic label-weighted semantic retrieval.

    Combines bge-large cosine with LogReg medium-label probabilities:
        score_i = P(label_i | q) × (cosine(q, x_i) + 1) / 2

    When classifier is confident (high P for one class) → behaves like template mode.
    When uncertain → degrades gracefully to cosine (embed-like).
    Avoids the hard classification error that limits pure template mode.
    """
    import numpy as np
    import torch
    import joblib  # type: ignore[import]
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_embeds = _load_or_build_embeddings(train_examples, embed_model_name, embed_cache)

    if not classifier_cache.exists():
        print(f"  Classifier not found — training now ...")
        _train_template_classifier(train_examples, train_embeds, classifier_cache, relevance)

    clf_data = joblib.load(str(classifier_cache))
    clf = clf_data["clf"]
    label_list = clf_data["skeleton_list"]
    label2idx = {l: i for i, l in enumerate(label_list)}

    get_label = _label_fn(relevance)
    # Pre-compute label index for every train example (-1 if label not in classifier)
    train_label_idxs = np.array(
        [label2idx.get(get_label(ex), -1) for ex in train_examples], dtype=np.int32
    )

    embed_model = SentenceTransformer(embed_model_name, device=device)

    def retrieve_smart(question: str) -> list[int]:
        q_vec = embed_model.encode([question], normalize_embeddings=True)
        cosine = (train_embeds @ q_vec.T).squeeze()
        # Normalise cosine to [0, 1] so it's on same scale as probability
        cos_norm = (cosine.astype(np.float32) + 1.0) / 2.0

        proba = clf.predict_proba(q_vec)[0]  # shape (n_classes,)
        # Look up each train example's class probability
        label_probs = np.where(
            train_label_idxs >= 0,
            proba[np.clip(train_label_idxs, 0, len(proba) - 1)],
            0.0,
        ).astype(np.float32)

        # Combined score: probability × normalised_cosine
        score = label_probs * cos_norm
        return list(map(int, (-score).argsort()[:max_k]))

    return retrieve_smart


def _train_template_classifier(
    train_examples: list[dict],
    train_embeds: "np.ndarray",
    output_path: Path,
    relevance: str = "medium",
) -> None:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import joblib

    get_label = _label_fn(relevance)
    labeled_idx = [i for i, ex in enumerate(train_examples) if get_label(ex)]
    labels = [get_label(train_examples[i]) for i in labeled_idx]
    unique_labels = sorted(set(labels))
    label2idx = {l: i for i, l in enumerate(unique_labels)}

    X = train_embeds[labeled_idx]
    y = [label2idx[l] for l in labels]

    n_classes = len(unique_labels)
    print(f"  {n_classes} unique label classes, {len(labeled_idx)} examples (relevance={relevance})")
    solver = "saga" if n_classes > 200 else "lbfgs"
    clf = LogisticRegression(C=10, max_iter=2000, solver=solver, n_jobs=-1)
    clf.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "skeleton_list": unique_labels}, str(output_path))
    print(f"  Classifier saved → {output_path}")


# ---------------------------------------------------------------------------
# Cross-encoder reranker factory
# ---------------------------------------------------------------------------

def _build_reranker(
    train_examples: list[dict],
    model_name: str,
) -> "Callable[[str, list[int]], list[int]]":
    """Return a reranker: (question, candidate_indices) -> reranked_indices.

    Special token 'prf' enables pseudo-relevance-feedback (PRF) reranking,
    which doesn't use a cross-encoder but instead uses the top-1 candidate's
    medium label to filter and reorder the remaining candidates.
    Designed for SQL structural similarity where cross-encoders trained on
    web data degrade performance.
    """
    if model_name in ("prf", "prf3"):
        # Pseudo-Relevance Feedback: majority-vote over top-N results to determine query label
        # prf: top-1 pivot, prf3: top-3 majority vote (more robust)
        from collections import Counter as _Counter
        import numpy as np

        vote_k = 1 if model_name == "prf" else 3
        get_train_label = _label_fn("medium")
        train_labels = [get_train_label(ex) for ex in train_examples]

        def rerank_prf(question: str, candidate_indices: list[int]) -> list[int]:
            if len(candidate_indices) <= 1:
                return candidate_indices
            top_labels = [train_labels[i] for i in candidate_indices[:vote_k]]
            pivot_label = _Counter(top_labels).most_common(1)[0][0]
            in_class = [i for i in candidate_indices if train_labels[i] == pivot_label]
            out_of_class = [i for i in candidate_indices if train_labels[i] != pivot_label]
            return in_class + out_of_class

        return rerank_prf

    import torch
    from sentence_transformers import CrossEncoder  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading cross-encoder {model_name} on {device} ...")
    reranker = CrossEncoder(model_name, device=device)

    def rerank(question: str, candidate_indices: list[int]) -> list[int]:
        if not candidate_indices:
            return candidate_indices
        pairs = [
            (question,
             train_examples[i]["question"] + " [SQL] " + train_examples[i].get("skeleton", ""))
            for i in candidate_indices
        ]
        scores = reranker.predict(pairs)
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        return [candidate_indices[j] for j in order]

    return rerank


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    train_examples: list[dict],
    test_examples: list[dict],
    retriever_fn: Callable[[str], list[int]],
    k_values: list[int],
    relevance: str = "medium",
    reranker_fn: "Callable | None" = None,
    final_k: int = 2,
) -> dict:
    """Compute Hit@K, Recall@K (strict), Precision@K, nDCG@K, MRR.

    If reranker_fn is provided:
      - retriever_fn returns top retrieve_k candidates (large K for high recall)
      - reranker_fn reorders those candidates
      - metrics are computed on the reranked top-final_k results
      - k_values in this case should be [final_k] (e.g. [2])

    Relevance: medium label (default) or skeleton.
    Only evaluates answerable test examples with a non-empty label.
    """
    get_label = _label_fn(relevance)
    labels_train = [get_label(ex) for ex in train_examples]

    # Build label → count mapping for strict Recall@K and nDCG IDCG
    label_count_train: dict[str, int] = defaultdict(int)
    for lbl in labels_train:
        if lbl:
            label_count_train[lbl] += 1

    test_eval = [ex for ex in test_examples if not ex.get("is_impossible") and get_label(ex)]
    if not test_eval:
        return {"error": f"no evaluable test examples for relevance={relevance}", "n_evaluated": 0}

    max_k = max(k_values)
    results: dict[int, dict] = {k: {"hit": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0}
                                 for k in k_values}
    mrr_total = 0.0
    n = 0

    for ex in test_eval:
        test_lbl = get_label(ex)
        top_idx = retriever_fn(ex["question"])

        if reranker_fn is not None:
            top_idx = reranker_fn(ex["question"], top_idx)

        retrieved_lbls = [labels_train[i] for i in top_idx[:max_k]]
        n_relevant_corpus = label_count_train.get(test_lbl, 0)

        for rank, lbl in enumerate(retrieved_lbls, 1):
            if lbl == test_lbl:
                mrr_total += 1.0 / rank
                break

        for k in k_values:
            top_k_lbls = retrieved_lbls[:k]
            n_relevant_retrieved = sum(1 for l in top_k_lbls if l == test_lbl)
            results[k]["hit"] += float(n_relevant_retrieved > 0)
            results[k]["recall"] += (n_relevant_retrieved / n_relevant_corpus
                                     if n_relevant_corpus > 0 else 0.0)
            results[k]["precision"] += n_relevant_retrieved / k
            results[k]["ndcg"] += _ndcg_at_k(top_k_lbls, test_lbl, n_relevant_corpus, k)

        n += 1

    for k in k_values:
        results[k]["hit"] /= n
        results[k]["recall"] /= n
        results[k]["precision"] /= n
        results[k]["ndcg"] /= n

    return {"per_k": results, "mrr": mrr_total / n, "n_evaluated": n}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(label: str, metrics: dict, relevance: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {label}  [relevance={relevance}]")
    print(f"{'='*68}")
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return
    print(f"  N evaluated : {metrics['n_evaluated']}")
    print(f"  MRR         : {metrics['mrr']:.4f}")
    print()
    print(f"  {'K':>5}  {'Hit@K':>8}  {'Recall@K':>10}  {'Prec@K':>8}  {'nDCG@K':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for k, v in sorted(metrics["per_k"].items()):
        print(f"  {k:>5}  {v['hit']:>8.4f}  {v['recall']:>10.4f}  "
              f"{v['precision']:>8.4f}  {v['ndcg']:>8.4f}")


def print_comparison(
    modes: list[str], all_results: dict[str, dict], relevance: str, k_show: list[int]
) -> None:
    print(f"\n{'='*82}")
    print(f"  COMPARISON — EHRSQL 2024 MIMIC-IV  [relevance={relevance}]")
    print(f"{'='*82}")
    header = f"  {'Mode':>12}  {'MRR':>6}"
    for k in k_show:
        header += f"  {'Hit@'+str(k):>7}  {'nDCG@'+str(k):>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode in modes:
        r = all_results.get(mode, {})
        if "error" in r:
            print(f"  {mode:>12}  ERROR")
            continue
        row = f"  {mode:>12}  {r['mrr']:>6.4f}"
        for k in k_show:
            if k in r["per_k"]:
                row += f"  {r['per_k'][k]['hit']:>7.4f}  {r['per_k'][k]['ndcg']:>8.4f}"
            else:
                row += f"  {'N/A':>7}  {'N/A':>8}"
        print(row)
    print()
    if relevance == "medium":
        print("  Hit@K: ≥1 relevant example in top-K | Ceiling: 89.8% (medium label)")
    else:
        print("  Hit@K: ≥1 relevant example in top-K | Ceiling: 71.1% (skeleton)")
    print("  nDCG@K: normalised DCG with binary relevance")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG retrieval quality — EHRSQL 2024 MIMIC-IV")
    parser.add_argument("--train", default="data/ehrsql2024/mimic_iv/train")
    parser.add_argument("--train-aug", default=None)
    parser.add_argument("--test", default="data/ehrsql2024/mimic_iv/test")
    parser.add_argument("--mode", default="all",
                        choices=["bm25", "embed", "hybrid", "template", "smart", "all"])
    parser.add_argument("--relevance", default="medium",
                        choices=["skeleton", "medium"])
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-cache", default=None)
    parser.add_argument("--classifier-cache", default=None)
    parser.add_argument("--k", default="1,2,5,10",
                        help="Comma-separated K values for evaluation (e.g. '2,10,100,500,1000')")
    # Two-stage reranking
    parser.add_argument("--rerank-model", default=None,
                        help="Cross-encoder model for reranking (e.g. cross-encoder/ms-marco-MiniLM-L-12-v2)")
    parser.add_argument("--retrieve-k", type=int, default=100,
                        help="First-stage retrieval size before reranking [default: 100]")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    base_dir = Path("data/ehrsql2024/mimic_iv")

    aug_suffix = "_combined" if args.train_aug else ""
    rel_suffix = f"_{args.relevance}"
    embed_cache = Path(args.embed_cache) if args.embed_cache else \
        base_dir / f"train{aug_suffix}_embeddings_bge_large.npy"
    clf_cache = Path(args.classifier_cache) if args.classifier_cache else \
        base_dir / f"template_classifier{rel_suffix}{aug_suffix}.pkl"

    print(f"Loading train: {args.train}")
    train_examples = [e for e in load_split(args.train) if not e["is_impossible"]]
    print(f"  {len(train_examples)} answerable train examples")

    if args.train_aug:
        aug_examples = [e for e in load_split(args.train_aug) if not e["is_impossible"]]
        print(f"  + {len(aug_examples)} train_aug examples")
        train_examples = train_examples + aug_examples
        print(f"  Combined corpus: {len(train_examples)} examples")

    get_label = _label_fn(args.relevance)
    unique_labels = len(set(get_label(e) for e in train_examples if get_label(e)))
    print(f"  {unique_labels} unique {args.relevance} labels in corpus")

    print(f"\nLoading test: {args.test}")
    test_examples = load_split(args.test)
    test_ans = [e for e in test_examples if not e["is_impossible"] and get_label(e)]
    print(f"  {len(test_examples)} total, {len(test_ans)} answerable with label")

    # Reranker setup
    reranker_fn = None
    first_stage_k = max(k_values)
    if args.rerank_model:
        reranker_fn = _build_reranker(train_examples, args.rerank_model)
        first_stage_k = args.retrieve_k
        print(f"\nReranking: retrieve top-{first_stage_k} → rerank with {args.rerank_model}")
        print(f"  Evaluating top-{max(k_values)} of reranked results")

    print(f"\nEmbed cache : {embed_cache}")
    print(f"Clf cache   : {clf_cache}")

    all_results: dict[str, dict] = {}
    modes = ["bm25", "embed", "hybrid", "template", "smart"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n--- Building {mode.upper()} retriever (first-stage K={first_stage_k}) ---")
        if mode == "bm25":
            retriever = _build_bm25_retriever(train_examples, max_k=first_stage_k)
        elif mode == "template":
            retriever = _build_template_retriever(
                train_examples, args.embed_model, embed_cache,
                clf_cache, args.relevance, max_k=first_stage_k,
            )
        elif mode == "smart":
            retriever = _build_smart_retriever(
                train_examples, args.embed_model, embed_cache,
                clf_cache, args.relevance, max_k=first_stage_k,
            )
        else:
            retriever = _build_embed_retriever(
                train_examples, args.embed_model, embed_cache,
                max_k=first_stage_k, mode=mode,
            )

        print(f"Evaluating {mode.upper()} on {len(test_ans)} test questions ...")
        metrics = evaluate_retrieval(
            train_examples, test_examples, retriever, k_values,
            relevance=args.relevance,
            reranker_fn=reranker_fn,
            final_k=max(k_values),
        )
        tag = f"{mode.upper()}" + (f" → rerank@{max(k_values)}" if reranker_fn else "")
        print_results(f"{tag} — EHRSQL 2024 MIMIC-IV", metrics, args.relevance)
        all_results[mode] = metrics

    k_show = [k for k in k_values if k <= 10]
    if not k_show:
        k_show = k_values[:3]
    if len(modes) > 1:
        print_comparison(modes, all_results, args.relevance, k_show)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "relevance": args.relevance,
                "corpus": "train+aug" if args.train_aug else "train",
                "rerank_model": args.rerank_model,
                "retrieve_k": first_stage_k,
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
