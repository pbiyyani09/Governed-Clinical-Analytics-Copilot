"""Classifier-guided few-shot retriever for EHRSQL (the `q_tag` insight).

EHRSQL questions are templated: the `q_tag` field is the masked QUESTION template
(167 of them in mimic_iii train, 100% test coverage). That turns example selection
into a SUPERVISED CLASSIFICATION problem — predict the question template, return its
examples — which a logistic regression solves far better than unsupervised cosine.

Measured on the q_tag oracle (1,198 test queries):

    method            P@2     hit@2   hit@10
    logreg-only       0.837   0.837   0.837  (flat — commits to one template)
    bi-encoder(MQS)   0.712   0.823   0.967  (hedges — recovers at depth)
    GATE hybrid       0.811   0.866   0.949  (best of both)  <-- this module
    fusion            0.851   0.856   0.862

The GATE strategy: if the logreg is confident (max class prob > theta), return the
predicted template's examples ranked by bi-encoder similarity; otherwise fall back
to pure bi-encoder retrieval. This keeps the classifier's precision on seen, regular
templates AND the bi-encoder's recall when the classifier is unsure (or the template
is rare / unseen) — fixing logreg's catastrophic flat-recall failure mode.

Used via `harness` --retrieval-mode classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ehrcopilot.eval.harness import (
    _canonicalize_gold_sql,
    _mask_question,
    _EMBED_MODEL_NAME,
    _EMBED_QUERY_PREFIX,
    _EMBED_DOC_PREFIX,
)


def _load_raw(train_path: Path) -> list[dict]:
    raw = json.load(open(train_path))
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, list):
        return raw
    return [{**v, "id": k} for k, v in raw.items()]


def _answerable(e: dict) -> bool:
    if str(e.get("is_impossible", False)).lower() in ("true", "1"):
        return False
    sql = (e.get("query") or e.get("sql") or "").strip().lower()
    return sql not in ("", "null", "none", "n/a")


def build_classifier_retriever(
    train_path: Path,
    top_k: int = 5,
    method: str = "fusion",
    theta: float = 0.5,
    label_field: str = "q_tag",
    embed_model_name: str = _EMBED_MODEL_NAME,
    query_prefix: str = _EMBED_QUERY_PREFIX,
    doc_prefix: str = _EMBED_DOC_PREFIX,
    embed_cache: "Path | None" = None,
) -> "Callable[[str], str]":
    """Gate hybrid: TF-IDF logreg over `label_field` templates + bge/MQS bi-encoder.

    Falls back to a pure bi-encoder retriever if scikit-learn is unavailable or the
    training data lacks `label_field`.
    """
    import numpy as np

    rows = [e for e in _load_raw(train_path) if _answerable(e) and e.get(label_field)]
    questions = [e["question"] for e in rows]
    gold_sqls = [_canonicalize_gold_sql(e.get("query") or e.get("sql") or "") for e in rows]
    labels = [e[label_field] for e in rows]

    # --- bi-encoder (masked-question index), same config as the hybrid retriever ---
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    masked = [_mask_question(q) for q in questions]
    if embed_cache is None:
        safe = embed_model_name.replace("/", "_")
        embed_cache = train_path.parent / f"train_embeddings_{safe}_mqs.npy"
    embed_model = SentenceTransformer(embed_model_name, device=device)
    if embed_cache.exists():
        train_embeds = np.load(str(embed_cache))
        # cache may have been built over a different row filter; rebuild if size differs
        if train_embeds.shape[0] != len(rows):
            train_embeds = embed_model.encode([doc_prefix + m for m in masked],
                                              batch_size=64, normalize_embeddings=True,
                                              convert_to_numpy=True)
    else:
        train_embeds = embed_model.encode([doc_prefix + m for m in masked],
                                          batch_size=64, normalize_embeddings=True,
                                          show_progress_bar=True, convert_to_numpy=True)
        np.save(str(embed_cache), train_embeds)

    # --- TF-IDF + logistic-regression template classifier ---
    clf = vec = None
    ex_class_idx = None  # per-train-example index into clf.classes_
    class_to_examples: dict[str, list[int]] = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(questions)
        clf = LogisticRegression(max_iter=300, C=10.0).fit(Xtr, labels)
        cls_index = {c: i for i, c in enumerate(clf.classes_)}
        ex_class_idx = np.array([cls_index.get(l, -1) for l in labels])
        for i, lab in enumerate(labels):
            class_to_examples.setdefault(lab, []).append(i)
        print(f"Classifier retriever: method={method}, {len(set(labels))} "
              f"{label_field} templates, theta={theta}")
    except Exception as exc:  # noqa: BLE001
        print(f"[classifier retriever] sklearn unavailable ({exc}); pure bi-encoder fallback")

    def _format(idxs: list[int]) -> str:
        lines = ["Similar examples:"]
        for i in idxs[:top_k]:
            lines.append(f"Q: {questions[i]}")
            lines.append(f"SQL: {gold_sqls[i]}")
        return "\n".join(lines)

    def _sims(question: str) -> "np.ndarray":
        qv = embed_model.encode([query_prefix + _mask_question(question)],
                                normalize_embeddings=True, convert_to_numpy=True)
        return (train_embeds @ qv.T).squeeze()

    def _zn(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    def _retrieve(question: str) -> str:
        sims = _sims(question)
        order = np.argsort(-sims)
        if clf is None:
            return _format(list(map(int, order[:top_k])))
        proba = clf.predict_proba(vec.transform([question]))[0]

        if method == "fusion":
            # rank every candidate by z(cosine) + z(logreg prob of its template).
            # Best P@K — ~85% of the top-5 share the query's q_tag template.
            logreg_ex = np.where(ex_class_idx >= 0, proba[np.clip(ex_class_idx, 0, len(proba) - 1)], 0.0)
            fused = _zn(sims) + _zn(logreg_ex)
            return _format(list(map(int, np.argsort(-fused)[:top_k])))

        # method == "gate": commit to the predicted template when confident,
        # else fall back to pure bi-encoder (best hit@2, graceful on low conf).
        if float(proba.max()) > theta:
            pred = clf.classes_[int(proba.argmax())]
            cand = class_to_examples.get(pred, [])
            rank = {int(j): r for r, j in enumerate(order)}
            cand = sorted(cand, key=lambda j: rank.get(j, 1 << 30))
            if len(cand) < top_k:
                seen = set(cand)
                cand += [int(j) for j in order if int(j) not in seen]
            return _format(cand)
        return _format(list(map(int, order[:top_k])))

    return _retrieve
