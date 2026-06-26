"""Comprehensive retrieval-mechanism benchmark for NL->SQL few-shot example selection.

Compares retrieval METHODS x EMBEDDING MODELS x REPRESENTATIONS x INDEX backends on
the EHRSQL corpus and reports rich IR metrics, so we can pick the right retriever
*before* spending GPU on training.

Task setup
----------
- Corpus  = train answerable (question, SQL) pairs  (~9.3k items).
- Queries = test answerable questions that carry a `tag` template label (~1.2k).
- Relevance oracle = EHRSQL `tag` field (the masked question template). A corpus
  item is RELEVANT to a query iff they share the same `tag`. This is the same
  signal rag_eval.py uses; here it is graded into full recall (|rel∩topK|/|rel|),
  not just hit-rate.

Methods
-------
- bm25    : lexical BM25Okapi over the corpus text.
- dense   : a sentence-embedding model + exact (faiss FlatIP) or HNSW search.
- hybrid  : Reciprocal Rank Fusion of bm25 + dense ranks.

Representations (what text is indexed / queried) — chosen to respect the
inference-time asymmetry (at test time only the NL question is known):
- q       : question only (both sides).
- q_sql   : corpus = question + SQL-skeleton(gold SQL); query = question.
- mqs     : masked question (literals/numbers masked) both sides (DAIL-SQL MQS).

Metrics (per K in --k-values, up to 1000)
- hit_rate@K  : >=1 relevant in top-K (the legacy "recall" in rag_eval.py).
- recall@K    : |relevant ∩ topK| / |relevant total|   (graded recall).
- precision@K : |relevant ∩ topK| / K.
- mrr         : mean reciprocal rank of the first relevant.
- ndcg@K      : binary-relevance nDCG.

Index backends
- flat : faiss IndexFlatIP (exact; ground truth for recall).
- hnsw : faiss IndexHNSWFlat (approximate; reports recall-vs-flat parity + latency).

Usage
-----
    python -m ehrcopilot.eval.retrieval_bench \
        --train data/ehrsql/ehrsql/mimic_iii/train.json \
        --test  data/ehrsql/ehrsql/mimic_iii/test.json \
        --models bm25 bge-large mxbai arctic-l gte-large \
        --reprs q q_sql \
        --index flat \
        --k-values 1 2 3 5 10 20 50 100 500 1000 \
        --output tests/evalgen/retrieval_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Embedding-model registry. Each entry: hf id + the recommended query/document
# instruction prefixes (empty when the model wants none) + load flags.
# Sources: model cards / MTEB (see research notes).
# ---------------------------------------------------------------------------

EMBED_MODELS: dict[str, dict] = {
    "bge-large": {
        "hf": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "trust_remote_code": False,
    },
    "mxbai": {
        "hf": "mixedbread-ai/mxbai-embed-large-v1",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "trust_remote_code": False,
    },
    "arctic-l": {
        "hf": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "",
        "doc_prefix": "",
        "trust_remote_code": True,
    },
    "gte-large": {
        "hf": "Alibaba-NLP/gte-large-en-v1.5",
        "query_prefix": "",
        "doc_prefix": "",
        "trust_remote_code": True,
    },
    "gte-qwen2-1.5b": {
        "hf": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "query_prefix": "Instruct: Given a clinical question, retrieve question-SQL pairs that answer structurally similar questions.\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
    },
    "qwen3-0.6b": {
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "Instruct: Given a clinical question, retrieve question-SQL pairs that answer structurally similar questions.\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": False,
    },
    "embeddinggemma": {
        "hf": "google/embeddinggemma-300m",
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
        "trust_remote_code": False,
        "dtype": "float32",  # embeddinggemma does not support fp16
    },
    "nomic": {
        "hf": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "trust_remote_code": True,
    },
    "e5-large": {
        "hf": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "trust_remote_code": False,
    },
    "sfr-code-2b": {
        "hf": "Salesforce/SFR-Embedding-Code-2B_R",
        "query_prefix": "Instruct: Given Code or Text, retrieve relevant content\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
    },
    "qwen3-4b": {
        "hf": "Qwen/Qwen3-Embedding-4B",
        "query_prefix": "Instruct: Given a clinical question, retrieve question-SQL pairs that answer structurally similar questions.\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": False,
    },
}

ABSTAIN = "[ABSTAIN]"


# ---------------------------------------------------------------------------
# Data + representation
# ---------------------------------------------------------------------------

def _load_split(path: Path) -> list[dict]:
    raw = json.load(open(path))
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, list):
        return raw
    return [{**v, "id": k} for k, v in raw.items()]


def _is_answerable(ex: dict) -> bool:
    if str(ex.get("is_impossible", False)).lower() in ("true", "1"):
        return False
    sql = (ex.get("query") or ex.get("sql") or "").strip().lower()
    return sql not in ("", "null", "none", "n/a")


_SKEL_STR = re.compile(r"'[^']*'")
_SKEL_NUM = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")
_MASK_NUM = re.compile(r"\b\d+(\.\d+)?\b")
_MASK_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def sql_skeleton(sql: str) -> str:
    """Mask literals, keep SQL structure (keywords/tables/columns)."""
    s = _SKEL_STR.sub("'?'", sql)
    s = _SKEL_NUM.sub("?", s)
    return _WS.sub(" ", s).strip()


def mask_question(q: str) -> str:
    """Approximate DAIL-SQL Masked-Question-Similarity: mask numbers/quoted values.

    Does NOT use the gold `tag`/`template` (that is the relevance oracle), so this
    stays leakage-free — it only masks surface literals present in the raw question.
    """
    s = _MASK_QUOTED.sub("<v>", q)
    s = _MASK_NUM.sub("<n>", s)
    return _WS.sub(" ", s).strip().lower()


def make_repr(ex: dict, mode: str, side: str) -> str:
    """Build the index/query text for a corpus or query item.

    side: 'doc' (corpus) or 'query'. Respects the asymmetry — the query side
    never sees gold SQL (unknown at inference).
    """
    q = ex["question"]
    sql = ex.get("query") or ex.get("sql") or ""
    if mode == "q":
        return q
    if mode == "q_sql":
        # corpus carries the SQL skeleton; query is question-only
        return f"{q} {sql_skeleton(sql)}" if side == "doc" else q
    if mode == "mqs":
        return mask_question(q)
    raise ValueError(f"unknown repr mode: {mode}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(model_key: str, texts: list[str], prefix: str, cache: Path | None) -> np.ndarray:
    if cache is not None and cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    import torch

    cfg = EMBED_MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_kwargs = {"device": device, "trust_remote_code": cfg.get("trust_remote_code", False)}
    if cfg.get("dtype") == "float32":
        st_kwargs["model_kwargs"] = {"torch_dtype": torch.float32}
    else:
        st_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(cfg["hf"], **st_kwargs)
    full = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(
        full, batch_size=64, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb


def embed_one(model_key: str, train_path: Path, test_path: Path, reprs: list[str],
              cache_dir: Path, max_queries: int | None) -> None:
    """Embed corpus+queries for ONE model (fresh process) and cache to .npy.

    Isolating each model in its own process means a model whose custom CUDA
    kernel throws a device-side assert cannot corrupt the CUDA context for the
    rest of the ablation.
    """
    cfg = EMBED_MODELS[model_key]
    safe = cfg["hf"].replace("/", "_")
    train = [e for e in _load_split(train_path) if _is_answerable(e)]
    test = [e for e in _load_split(test_path) if _is_answerable(e) and e.get("tag")]
    if max_queries:
        test = test[:max_queries]
    for mode in reprs:
        doc_texts = [make_repr(e, mode, "doc") for e in train]
        q_texts = [make_repr(e, mode, "query") for e in test]
        embed_texts(model_key, doc_texts, cfg["doc_prefix"], cache_dir / f"doc_{safe}_{mode}.npy")
        embed_texts(model_key, q_texts, cfg["query_prefix"], cache_dir / f"qry_{safe}_{mode}.npy")
        print(f"[embed-only {model_key}/{mode}] cached", flush=True)


# ---------------------------------------------------------------------------
# Index backends
# ---------------------------------------------------------------------------

def search_flat(doc_emb: np.ndarray, q_emb: np.ndarray, k: int) -> np.ndarray:
    import faiss
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    _, idx = index.search(q_emb, k)
    return idx


def search_hnsw(doc_emb: np.ndarray, q_emb: np.ndarray, k: int,
                M: int = 32, ef_construction: int = 200, ef_search: int = 256) -> tuple[np.ndarray, float]:
    import faiss
    index = faiss.IndexHNSWFlat(doc_emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(doc_emb)
    index.hnsw.efSearch = ef_search
    t0 = time.time()
    _, idx = index.search(q_emb, k)
    latency_ms = 1000.0 * (time.time() - t0) / len(q_emb)
    return idx, latency_ms


# ---------------------------------------------------------------------------
# Retrievers -> ranked corpus-index lists (one per query, length max_k)
# ---------------------------------------------------------------------------

def bm25_ranking(corpus_texts: list[str], query_texts: list[str], max_k: int) -> list[list[int]]:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in corpus_texts])
    out = []
    for q in query_texts:
        scores = bm25.get_scores(q.lower().split())
        out.append(list(map(int, (-scores).argsort()[:max_k])))
    return out


def dense_ranking(doc_emb: np.ndarray, q_emb: np.ndarray, max_k: int, index: str) -> tuple[list[list[int]], float]:
    if index == "hnsw":
        idx, lat = search_hnsw(doc_emb, q_emb, max_k)
        return [list(map(int, row)) for row in idx], lat
    t0 = time.time()
    idx = search_flat(doc_emb, q_emb, max_k)
    lat = 1000.0 * (time.time() - t0) / len(q_emb)
    return [list(map(int, row)) for row in idx], lat


def rrf_fusion(rank_lists: list[list[list[int]]], max_k: int, k_const: int = 60) -> list[list[int]]:
    """Reciprocal Rank Fusion across multiple per-query ranked lists."""
    fused = []
    n_queries = len(rank_lists[0])
    for qi in range(n_queries):
        scores: dict[int, float] = {}
        for rl in rank_lists:
            for rank, doc in enumerate(rl[qi]):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k_const + rank)
        ranked = sorted(scores, key=lambda d: -scores[d])[:max_k]
        fused.append(ranked)
    return fused


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(rankings: list[list[int]], rel_sets: list[set[int]],
                    rel_counts: list[int], k_values: list[int]) -> dict:
    per_k = {k: {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0} for k in k_values}
    mrr_sum = 0.0
    n = len(rankings)
    for ranking, rel, total_rel in zip(rankings, rel_sets, rel_counts):
        # MRR (first relevant)
        for rank, doc in enumerate(ranking):
            if doc in rel:
                mrr_sum += 1.0 / (rank + 1)
                break
        for k in k_values:
            topk = ranking[:k]
            n_rel = sum(1 for d in topk if d in rel)
            per_k[k]["hit_rate"] += 1.0 if n_rel > 0 else 0.0
            per_k[k]["recall"] += (n_rel / total_rel) if total_rel else 0.0
            per_k[k]["precision"] += n_rel / k
            # binary nDCG@k
            dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(topk) if d in rel)
            ideal_hits = min(total_rel, k)
            idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
            per_k[k]["ndcg"] += (dcg / idcg) if idcg else 0.0
    for k in k_values:
        for m in per_k[k]:
            per_k[k][m] /= n
    return {"per_k": {str(k): per_k[k] for k in k_values}, "mrr": mrr_sum / n, "n_evaluated": n}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(train_path: Path, test_path: Path, model_keys: list[str], reprs: list[str],
        index: str, k_values: list[int], cache_dir: Path, max_queries: int | None) -> dict:
    train = [e for e in _load_split(train_path) if _is_answerable(e)]
    test = [e for e in _load_split(test_path) if _is_answerable(e) and e.get("tag")]
    if max_queries:
        test = test[:max_queries]
    print(f"corpus={len(train)}  queries={len(test)}")

    # relevance oracle: same `tag`
    corpus_tags = [e.get("tag") for e in train]
    tag_to_docs: dict[str, set[int]] = {}
    for i, t in enumerate(corpus_tags):
        tag_to_docs.setdefault(t, set()).add(i)
    rel_sets = [tag_to_docs.get(e["tag"], set()) for e in test]
    rel_counts = [len(s) for s in rel_sets]
    max_k = max(k_values)

    results: dict = {}

    for mode in reprs:
        doc_texts = [make_repr(e, mode, "doc") for e in train]
        q_texts = [make_repr(e, mode, "query") for e in test]

        # ---- BM25 (lexical) ----
        bm25_key = f"bm25/{mode}"
        print(f"[{bm25_key}] ...")
        bm25_ranks = bm25_ranking(doc_texts, q_texts, max_k)
        results[bm25_key] = compute_metrics(bm25_ranks, rel_sets, rel_counts, k_values)

        # ---- Dense per model + Hybrid. Embeddings are produced by isolated
        # `--embed-only` subprocesses (one CUDA context per model, so a single
        # model's device-side assert can't poison the others); scoring only
        # READS the cached .npy here. Missing cache => model skipped. ----
        for mk in model_keys:
            if mk == "bm25":
                continue
            cfg = EMBED_MODELS[mk]
            safe = cfg["hf"].replace("/", "_")
            dcache = cache_dir / f"doc_{safe}_{mode}.npy"
            qcache = cache_dir / f"qry_{safe}_{mode}.npy"
            if not (dcache.exists() and qcache.exists()):
                print(f"[dense/{mk}/{mode}] no cache — skipped (run --embed-only {mk})", flush=True)
                results[f"dense:{mk}/{mode}/{index}"] = {"error": "no embedding cache"}
                continue
            try:
                doc_emb = np.load(dcache)
                q_emb = np.load(qcache)
                dranks, lat = dense_ranking(doc_emb, q_emb, max_k, index)
                dkey = f"dense:{mk}/{mode}/{index}"
                m = compute_metrics(dranks, rel_sets, rel_counts, k_values)
                m["query_latency_ms"] = lat
                results[dkey] = m
                # hybrid = RRF(bm25, dense)
                hranks = rrf_fusion([bm25_ranks, dranks], max_k)
                results[f"hybrid:{mk}/{mode}/{index}"] = compute_metrics(hranks, rel_sets, rel_counts, k_values)
                print(f"[dense/{mk}/{mode}] R@10(graded)={m['per_k'][str(min(10,max_k))]['recall']:.4f} "
                      f"hit@10={m['per_k'][str(min(10,max_k))]['hit_rate']:.4f} MRR={m['mrr']:.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"[dense/{mk}/{mode}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                results[f"dense:{mk}/{mode}/{index}"] = {"error": f"{type(exc).__name__}: {exc}"}

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Retrieval-mechanism benchmark for EHRSQL few-shot")
    p.add_argument("--train", default="data/ehrsql/ehrsql/mimic_iii/train.json")
    p.add_argument("--test", default="data/ehrsql/ehrsql/mimic_iii/test.json")
    p.add_argument("--models", nargs="+", default=["bm25", "bge-large"],
                   help=f"bm25 + any of: {', '.join(EMBED_MODELS)}")
    p.add_argument("--reprs", nargs="+", default=["q", "q_sql"], choices=["q", "q_sql", "mqs"])
    p.add_argument("--index", default="flat", choices=["flat", "hnsw"])
    p.add_argument("--k-values", nargs="+", type=int,
                   default=[1, 2, 3, 5, 10, 20, 50, 100, 500, 1000])
    p.add_argument("--cache-dir", default="data/ehrsql/embed_cache")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--embed-only", default=None,
                   help="Embed just this ONE model (fresh process) and exit — for isolated ablation.")
    p.add_argument("--output", default="tests/evalgen/retrieval_bench.json")
    args = p.parse_args()

    if args.embed_only:
        embed_one(args.embed_only, Path(args.train), Path(args.test), args.reprs,
                  Path(args.cache_dir), args.max_queries)
        return

    res = run(Path(args.train), Path(args.test), args.models, args.reprs,
              args.index, args.k_values, Path(args.cache_dir), args.max_queries)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")

    # compact comparison table (sorted by recall@10 then mrr)
    kref = "10" if "10" in next(iter(res.values()))["per_k"] else next(iter(next(iter(res.values()))["per_k"]))
    rows = sorted(res.items(), key=lambda kv: (-kv[1]["per_k"][kref]["recall"], -kv[1]["mrr"]))
    print(f"\n{'method':38s} | {'R@2':>6} | {'R@10':>6} | {'R@100':>6} | {'P@2':>6} | {'MRR':>6} | {'nDCG@10':>7}")
    print("-" * 96)
    for name, m in rows:
        pk = m["per_k"]
        def g(k, f):
            return pk.get(str(k), {}).get(f, 0.0)
        print(f"{name:38s} | {g(2,'recall'):.4f} | {g(10,'recall'):.4f} | {g(100,'recall'):.4f} | "
              f"{g(2,'precision'):.4f} | {m['mrr']:.4f} | {g(10,'ndcg'):.4f}")


if __name__ == "__main__":
    main()
