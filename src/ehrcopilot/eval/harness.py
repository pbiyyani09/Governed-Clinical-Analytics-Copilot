"""Evaluation harness: Execution Accuracy (EX) and Reliability Score RS(N).

Metrics mirror the EHRSQL 2024 shared task official scoring program
(github.com/glee4810/ehrsql-2024/blob/master/scoring_program/scoring.py).

EX  — fraction of answerable questions where the predicted SQL produces the same
      execution result as the gold SQL.

RS(N) — official EHRSQL 2024 reliability score (scoring_utils.py::penalize):
         Each question gets a base score:
           +1  answerable  + correct SQL result
            0  answerable  + abstained  (wrong abstention)
           -1  answerable  + wrong SQL  (execution error OR wrong result)
           -1  unanswerable + any SQL   (hallucinated)
           +1  unanswerable + abstained (correct abstention)
         RS(N) = mean(score × N  if score == −1  else score)
               = (#correct_answers + #correct_abstentions
                  - N × (#wrong_SQL_answerable + #wrong_on_unanswerable)) / total
         Wrong SQL on ANSWERABLE questions is also penalised at rate N.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ehrcopilot import config
from ehrcopilot.db.connection import execute_query

ABSTAIN_TOKEN = "[ABSTAIN]"


_CURRENT_TIME = "2100-12-31 23:59:00"  # EHRSQL 2024 synthetic "now" (all MIMIC dates are year 2100)
_CURRENT_DATE = "2100-12-31"
_CURRENT_TIME_ONLY = "23:59:00"
_TIME_PATTERN = re.compile(
    r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+INTERVAL (\d+) (MONTH|YEAR|DAY)\)",
    re.IGNORECASE,
)
# Official EHRSQL 2024 vital-sign normal ranges (from scoring_program/postprocessing.py).
# Gold SQL references abstract column names like `temperature_lower` / `temperature_upper`
# that don't exist in the SQLite DB — the scorer substitutes these before execution.
_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (35.5, 38.1),
    "sao2": (95.0, 100.0),
    "heart rate": (60.0, 100.0),
    "respiration": (12.0, 18.0),
    "systolic bp": (90.0, 120.0),
    "diastolic bp": (60.0, 90.0),
    "mean bp": (60.0, 110.0),
}
_VITAL_LOWER_RE = re.compile(r"[ \n]+([a-zA-Z0-9_]+_lower)")
_VITAL_UPPER_RE = re.compile(r"[ \n]+([a-zA-Z0-9_]+_upper)")


def _date_fn_to_sqlite(m: re.Match) -> str:
    """Convert MySQL DATE_SUB/DATE_ADD to SQLite datetime() modifier."""
    fn, date, n, unit = m.group(1).upper(), m.group(2), m.group(3), m.group(4).lower()
    unit = unit.rstrip("s") if n == "1" else (unit if unit.endswith("s") else unit + "s")
    sign = "-" if fn == "DATE_SUB" else "+"
    return f"datetime({date}, '{sign}{n} {unit}')"


def post_process_sql(sql: str) -> str:
    """Mirror the official EHRSQL 2024 scoring_program/postprocessing.py exactly.

    Applied to BOTH gold and predicted SQL before execution so that queries using
    the 'current_time' placeholder (286/934 test queries) and strftime '%y'/'%j'
    patterns (363/934) produce the same results as the official scorer.

    Without this, our harness executes with the actual system date (2026) against
    synthetic MIMIC data whose dates are all in year 2100, causing 177/934 gold
    queries to return wrong reference results (82 empty instead of data).
    """
    sql = re.sub(r"[ ]+", " ", sql.replace("\n", " ")).strip()
    sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    sql = _TIME_PATTERN.sub(_date_fn_to_sqlite, sql)
    if "current_time" in sql:
        sql = sql.replace("current_time", f"'{_CURRENT_TIME}'")
    if "current_date" in sql:
        sql = sql.replace("current_date", f"'{_CURRENT_DATE}'")
    if "'now'" in sql:
        sql = sql.replace("'now'", f"'{_CURRENT_TIME}'")
    if "NOW()" in sql:
        sql = sql.replace("NOW()", f"'{_CURRENT_TIME}'")
    if "CURDATE()" in sql:
        sql = sql.replace("CURDATE()", f"'{_CURRENT_DATE}'")
    if "CURTIME()" in sql:
        sql = sql.replace("CURTIME()", f"'{_CURRENT_TIME_ONLY}'")
    # Vital sign range substitution: `temperature_lower` → 35.5, etc.
    lower_m = _VITAL_LOWER_RE.search(sql)
    upper_m = _VITAL_UPPER_RE.search(sql)
    if lower_m and upper_m:
        lower_expr = lower_m.group(1)
        upper_expr = upper_m.group(1)
        vital_names = list(set(
            re.findall(r"([a-zA-Z0-9_]+)_lower", lower_expr) +
            re.findall(r"([a-zA-Z0-9_]+)_upper", upper_expr)
        ))
        if len(vital_names) == 1:
            vital_key = vital_names[0].replace("_", " ")
            if vital_key in _VITAL_RANGES:
                lo, hi = _VITAL_RANGES[vital_key]
                sql = sql.replace(lower_expr, str(lo)).replace(upper_expr, str(hi))
    sql = sql.replace("%y", "%Y").replace("%j", "%J")
    return sql


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EHRSQLExample:
    id: str
    question: str
    gold_sql: str
    is_answerable: bool  # False for unanswerable questions
    tag: str = ""        # abstract template (EHRSQL 2022 format only)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EHRSQLExample":
        sql = d.get("sql") or d.get("query") or ""
        # EHRSQL marks unanswerable with query/sql="null" or is_impossible=True
        is_impossible = d.get("is_impossible", False)
        is_answerable = (
            not is_impossible
            and sql.strip().lower() not in ("", "null", "none", "n/a")
        )
        return cls(
            id=str(d.get("id", d.get("uid", ""))),
            question=d["question"],
            gold_sql=sql if is_answerable else "",
            is_answerable=is_answerable,
            tag=d.get("tag", ""),
        )


@dataclass
class PredictionResult:
    example_id: str
    predicted_sql: str | None        # None means model emitted [ABSTAIN]
    abstained: bool
    exec_result: list[dict[str, Any]] | None = None
    exec_error: str | None = None
    latency_ms: float = 0.0


@dataclass
class EvalMetrics:
    total: int = 0
    answerable: int = 0
    unanswerable: int = 0

    # EX components (answerable questions only)
    correct_answers: int = 0          # answerable + correct SQL result (gold exec OK)
    wrong_abstentions: int = 0        # answerable + model abstained
    wrong_answers: int = 0            # unanswerable + model gave SQL (any)
    correct_abstentions: int = 0      # unanswerable + model abstained

    # Gold SQL validity tracking (answerable questions only)
    # These reveal how many of "correct_answers" were real vs. false positives.
    gold_exec_valid: int = 0          # gold executed with actual data rows
    gold_exec_empty: int = 0          # gold executed but returned empty (patient absent from demo)
    gold_exec_error: int = 0          # gold failed — missing table/column in MIMIC-IV-Demo

    # Repair loop stats (non-zero only when repair is enabled)
    repair_attempts: int = 0
    repair_successes: int = 0

    # Entropy filtering stats (non-empty only when --entropy-threshold is set)
    entropy_samples: list[float] = field(default_factory=list)
    entropy_abstentions: int = 0   # predictions converted to [ABSTAIN] by entropy gate
    empty_abstentions: int = 0     # predictions converted to [ABSTAIN] by empty-result gate

    # Latency
    latency_samples: list[float] = field(default_factory=list)

    @property
    def ex(self) -> float:
        """Competition EX: correct_answers / all_answerable."""
        if self.answerable == 0:
            return 0.0
        return self.correct_answers / self.answerable

    @property
    def ex_valid(self) -> float:
        """Local signal: correct_answers / questions where gold SQL produced real data.

        Excludes questions where gold SQL errored (false-positive territory on MIMIC-IV-Demo)
        and questions where gold returned empty (unverifiable without the full MIMIC-III DB).
        This is the only trustworthy local accuracy signal until we have MIMIC-III.
        """
        if self.gold_exec_valid == 0:
            return 0.0
        # correct_answers includes matches on empty gold — bound it to valid gold count
        return min(self.correct_answers, self.gold_exec_valid) / self.gold_exec_valid

    @property
    def wrong_sql_answerable(self) -> int:
        """Answerable questions where model gave SQL but result was wrong or errored."""
        return self.answerable - self.correct_answers - self.wrong_abstentions

    def rs(self, n: int) -> float:
        """Official EHRSQL 2024 RS(N).

        Score per question:
          +1  correct answer (SQL result matches gold) or correct abstention
           0  wrong abstention ([ABSTAIN] on answerable) OR wrong SQL on answerable
          -1  hallucination on unanswerable (any SQL when gold='null') → ×N

        Only hallucinations on unanswerable questions are penalised at rate N.
        Wrong SQL on answerable questions is score 0, NOT −1.
        """
        if self.total == 0:
            return 0.0
        return (
            self.correct_answers + self.correct_abstentions
            - n * self.wrong_answers
        ) / self.total

    @property
    def p50_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        return sorted_samples[len(sorted_samples) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        return sorted_samples[int(len(sorted_samples) * 0.95)]

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "total": self.total,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "EX": round(self.ex, 4),
            "EX_valid_gold": round(self.ex_valid, 4),
            "RS(0)": round(self.rs(0), 4),
            "RS(5)": round(self.rs(5), 4),
            "RS(10)": round(self.rs(10), 4),
            "correct_answers": self.correct_answers,
            "wrong_abstentions": self.wrong_abstentions,
            "wrong_sql_answerable": self.wrong_sql_answerable,
            "wrong_answers_on_unanswerable": self.wrong_answers,
            "correct_abstentions": self.correct_abstentions,
            "gold_exec_valid": self.gold_exec_valid,
            "gold_exec_empty": self.gold_exec_empty,
            "gold_exec_error": self.gold_exec_error,
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
        }
        if self.repair_attempts > 0:
            d["repair_attempts"] = self.repair_attempts
            d["repair_successes"] = self.repair_successes
        if self.entropy_samples:
            sorted_ent = sorted(self.entropy_samples)
            d["entropy_p50"] = round(sorted_ent[len(sorted_ent) // 2], 4)
            d["entropy_p95"] = round(sorted_ent[int(len(sorted_ent) * 0.95)], 4)
            d["entropy_abstentions"] = self.entropy_abstentions
        if self.empty_abstentions:
            d["empty_abstentions"] = self.empty_abstentions
        return d


# ---------------------------------------------------------------------------
# Execution result comparison
# ---------------------------------------------------------------------------


def _normalize_item(v: Any) -> str:
    """Match official process_item: round floats to 3dp, then stringify."""
    try:
        return str(round(float(v), 3))
    except (TypeError, ValueError):
        return str(v)


def _normalize_result(rows: list[dict[str, Any]] | None) -> str:
    """Order-independent result-set comparison matching the official EHRSQL 2024 scorer.

    Mirrors scoring_utils.py::process_answer:
    - Values only (column names ignored — avoids alias mismatches like COUNT(*) vs count(*))
    - Floats rounded to 3 decimal places (avoids floating-point equality failures)
    - Sorted rows (order-independent)
    - First 100 rows only (matches official 100-row cap)
    """
    if not rows:
        return "[]"
    return str(sorted([[_normalize_item(v) for v in row.values()] for row in rows])[:100])


def _exec_safe(sql: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Execute SQL, returning (rows, error). Never raises."""
    try:
        rows = execute_query(sql)
        return rows, None
    except Exception as exc:
        return None, str(exc)


def results_match(
    pred_rows: list[dict[str, Any]] | None,
    gold_rows: list[dict[str, Any]] | None,
    *,
    gold_err: str | None = None,
) -> bool:
    """Return True only when gold executed successfully AND result sets are identical.

    Gold error guard: without it, any two SQL errors produce "[]" == "[]" and count
    as "correct" — this guard prevents false positives on schema-error questions.
    """
    if gold_err is not None:
        return False
    return _normalize_result(pred_rows) == _normalize_result(gold_rows)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_ehrsql_split(split_path: Path) -> list[EHRSQLExample]:
    """Load an EHRSQL split.

    Handles two formats:
      - Directory (EHRSQL 2024): contains data.json (questions) + label.json (id→sql)
      - Single JSON file (EHRSQL 2022): list of dicts or {"data": [...]}
    """
    split_path = Path(split_path)

    # EHRSQL 2024 directory format: data.json + label.json
    if split_path.is_dir():
        with open(split_path / "data.json") as f:
            data_raw = json.load(f)
        with open(split_path / "label.json") as f:
            labels: dict[str, str] = json.load(f)

        examples = data_raw["data"] if "data" in data_raw else data_raw
        merged = [{"id": ex["id"], "question": ex["question"], "sql": labels.get(ex["id"], "null")}
                  for ex in examples]
        return [EHRSQLExample.from_dict(d) for d in merged]

    # Single JSON file (EHRSQL 2022 / legacy format)
    with open(split_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [EHRSQLExample.from_dict(d) for d in raw]
    if "data" in raw:
        return [EHRSQLExample.from_dict(d) for d in raw["data"]]
    return [EHRSQLExample.from_dict({**v, "id": k}) for k, v in raw.items()]


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


def _base_tag(tag: str) -> str:
    """Strip time-filter annotations from an EHRSQL tag.

    E.g.: "count patients [time_filter_global1:abs-year-in]."
       → "count patients."

    The 9318 training questions map to 165 unique base templates (vs 3282 full tags).
    Used as the relevance signal in template-aware retrieval.
    """
    s = re.sub(r'\s*\[[^\]]*\]', '', tag).strip()
    return re.sub(r'\s+', ' ', s).strip()


def _sql_skeleton(sql: str) -> str:
    """Strip concrete values from SQL to expose the structural template.

    Replaces string literals → '?' and integers → ?, leaving SQL keywords,
    table/column names, and operators intact. This lets the embedding model
    cluster survival-rate queries together regardless of disease name.
    """
    s = sql.lower()
    s = re.sub(r"'[^']*'", "'?'", s)
    s = re.sub(r'\b\d+\b', '?', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


_EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def _medium_label(sql: str) -> str:
    """Coarse structural label for MIMIC-IV template retrieval.

    Groups SQL queries by the tables they touch + operation type flags.
    Matches the medium_label() function in rag_eval.py.
    Format: "table1,table2,...::AGG_GROUP_JOIN"
    """
    s = sql.lower()
    tables = sorted(set(re.findall(r'\b(?:from|join)\s+(\w+)', s)))
    is_agg = int(bool(re.search(r'\b(count|avg|sum|max|min)\b', s)))
    has_group = int('group by' in s)
    has_join = int('join' in s)
    return f"{','.join(tables)}::{is_agg}{has_group}{has_join}"


# ---------------------------------------------------------------------------
# Hybrid few-shot retriever (BM25 + semantic embeddings via RRF)
# ---------------------------------------------------------------------------


def build_few_shot_retriever(
    train_path: Path,
    top_k: int = 2,
    mode: str = "hybrid",
    embed_cache: "Path | None" = None,
    classifier_cache: "Path | None" = None,
    aug_path: "Path | None" = None,
) -> "Callable[[str], str]":
    """Build a few-shot retriever over train (+ optional train_aug) examples.

    mode:
      "bm25"     — keyword BM25 only
      "embed"    — semantic embedding only (BAAI/bge-large-en-v1.5, GPU if available)
      "hybrid"   — Reciprocal Rank Fusion of BM25 + embedding [default]
      "template" — LogReg classifier (medium label = table-set + op-type) predicts
                   structural class → top-K cosine within class. Falls back to hybrid
                   when predicted class has no candidates.

    aug_path: optional train_aug directory. When provided, aug examples are added to the
      retrieval corpus. The combined index (~40K) dramatically improves template mode
      (median class size 5 → 27) and gives hybrid more diverse candidates.

    embed_cache: path to save/load precomputed .npy embeddings (auto-derived if None).
    classifier_cache: path to the medium-label LogReg classifier (.pkl).
    """
    import numpy as np
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    from collections import defaultdict

    base_dir = Path("data/ehrsql2024/mimic_iv")

    examples = [e for e in load_ehrsql_split(train_path) if e.is_answerable and e.gold_sql]

    if aug_path is not None:
        aug_examples = [e for e in load_ehrsql_split(aug_path) if e.is_answerable and e.gold_sql]
        examples = examples + aug_examples
        print(f"Retrieval corpus: {len(examples)} examples (train + aug)")
    else:
        print(f"Retrieval corpus: {len(examples)} train examples")

    gold_sqls = [e.gold_sql for e in examples]
    questions = [e.question for e in examples]
    n = len(examples)

    # BM25 index (bm25, hybrid, and template-fallback modes)
    tokenized = [q.lower().split() for q in questions]
    bm25 = BM25Okapi(tokenized)

    embed_model = None
    train_embeds = None
    clf = None
    medium_to_indices: "dict | None" = None

    if mode in ("embed", "hybrid", "template"):
        import torch
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if embed_cache is None:
            suffix = "_combined" if aug_path else ""
            embed_cache = base_dir / f"train{suffix}_embeddings_bge_large.npy"

        cached = np.load(str(embed_cache)) if embed_cache.exists() else None
        if cached is not None and cached.shape[0] == n:
            print(f"Loading cached embeddings from {embed_cache}")
            train_embeds = cached
        else:
            if cached is not None:
                print(f"Cache size mismatch ({cached.shape[0]} vs {n}) — rebuilding")
            print(f"Computing {n} embeddings with {_EMBED_MODEL_NAME} on {device} ...")
            _tmp = SentenceTransformer(_EMBED_MODEL_NAME, device=device)
            index_texts = [q + " " + _sql_skeleton(sql) for q, sql in zip(questions, gold_sqls)]
            train_embeds = _tmp.encode(
                index_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True,
            )
            embed_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(embed_cache), train_embeds)
            print(f"Embeddings saved to {embed_cache}")

        embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device=device)
        print(f"Embedding model ready ({device})")

    if mode == "template":
        import joblib  # type: ignore[import]

        if classifier_cache is None:
            suffix = "_combined" if aug_path else ""
            classifier_cache = base_dir / f"template_classifier_medium{suffix}.pkl"

        if not classifier_cache.exists():
            raise FileNotFoundError(
                f"Template classifier not found at {classifier_cache}. "
                f"Run: python -m ehrcopilot.eval.rag_eval --mode template --relevance medium first."
            )

        resolved_clf = classifier_cache.resolve()
        project_data = (config.DATA_DIR).resolve()
        if not str(resolved_clf).startswith(str(project_data)):
            raise ValueError(
                f"--classifier-cache path {resolved_clf} is outside the project data "
                f"directory ({project_data}). Refusing to load."
            )

        clf_data = joblib.load(str(resolved_clf))
        clf = clf_data["clf"]
        clf_labels = clf_data["skeleton_list"]
        print(f"Template classifier loaded ({len(clf_labels)} medium-label classes)")

        medium_to_indices = defaultdict(list)
        for i, sql in enumerate(gold_sqls):
            lbl = _medium_label(sql)
            if lbl:
                medium_to_indices[lbl].append(i)

    def _hybrid_top_k(q_vec: "np.ndarray", bm25_scores: "np.ndarray") -> "list[int]":
        bm25_order = (-bm25_scores).argsort()
        bm25_ranks = np.empty(n, dtype=np.float32)
        bm25_ranks[bm25_order] = np.arange(1, n + 1, dtype=np.float32)
        cosine = (train_embeds @ q_vec.T).squeeze()
        embed_order = (-cosine).argsort()
        embed_ranks = np.empty(n, dtype=np.float32)
        embed_ranks[embed_order] = np.arange(1, n + 1, dtype=np.float32)
        rrf = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
        return list(map(int, (-rrf).argsort()[:top_k]))

    def _retrieve(question: str) -> str:
        if mode == "template":
            assert embed_model is not None and clf is not None
            assert train_embeds is not None and medium_to_indices is not None
            q_vec = embed_model.encode([question], normalize_embeddings=True)
            predicted_label = clf_labels[clf.predict(q_vec)[0]]
            candidates = medium_to_indices.get(predicted_label, [])
            if candidates:
                cand_embeds = train_embeds[candidates]
                cos = (cand_embeds @ q_vec.T).squeeze()
                if cos.ndim == 0:
                    cos = cos.reshape(1)
                best = (-cos).argsort()[:top_k]
                top_idx = [candidates[int(i)] for i in best]
            else:
                # Hybrid fallback for ~10% of questions whose medium label isn't in corpus
                bm25_scores = bm25.get_scores(question.lower().split())
                top_idx = _hybrid_top_k(q_vec, bm25_scores)
        else:
            bm25_scores = bm25.get_scores(question.lower().split())
            if mode == "bm25":
                top_idx = list(map(int, (-bm25_scores).argsort()[:top_k]))
            else:
                assert embed_model is not None and train_embeds is not None
                q_vec = embed_model.encode([question], normalize_embeddings=True)
                cosine = (train_embeds @ q_vec.T).squeeze()
                if mode == "embed":
                    top_idx = list(map(int, (-cosine).argsort()[:top_k]))
                else:
                    top_idx = _hybrid_top_k(q_vec, bm25_scores)

        lines = ["Similar examples:"]
        for idx in top_idx:
            lines.append(f"Q: {questions[idx]}")
            lines.append(f"SQL: {gold_sqls[idx]}")
        return "\n".join(lines)

    return _retrieve


# ---------------------------------------------------------------------------
# Model interface — swap in any callable (pipeline, agent, API)
# ---------------------------------------------------------------------------

ModelFn = Callable[[str], str]
"""A function that takes a natural-language question and returns predicted SQL
or the literal string "[ABSTAIN]"."""


def run_hf_baseline(
    model_name: str = config.INFERENCE_MODEL,
    few_shot_retriever: "Callable[[str], str] | None" = None,
) -> ModelFn:
    """Return a ModelFn for evaluation.

    Supports two loading paths:
      - Local path (starts with '.' or '/'): loaded via Unsloth FastLanguageModel
        (works for LoRA adapters and merged models, no bitsandbytes needed)
      - HF Hub name: loaded via transformers pipeline with 4-bit NF4 quantization

    few_shot_retriever: optional callable (question) -> str built by build_few_shot_retriever().
      When provided, retrieved examples are prepended to each user message.
    """
    import torch

    system_prompt = config.SYSTEM_PROMPT

    import os as _os
    is_local = (
        _os.path.isdir(model_name)
        or model_name.startswith("./")
        or model_name.startswith("../")
        or model_name.startswith("/")
        or model_name.startswith("checkpoints/")
        or model_name.startswith("models/")
    )

    if is_local:
        # Load via Unsloth — works for LoRA adapters without bitsandbytes at forward time
        from unsloth import FastLanguageModel  # type: ignore[import]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        class _UnslothPredictor:
            """Callable model function with repair and optional few-shot RAG support."""

            def _generate_with_entropy(self, msgs: list[dict], temperature: float = 0.0) -> tuple[str, float]:
                from transformers import LogitsProcessor

                class _EntropyCapture(LogitsProcessor):
                    """Intercept logits at each generation step to compute max token entropy.
                    Uses LogitsProcessor because Unsloth's patched generate() drops output_scores.
                    """
                    def __init__(self) -> None:
                        self.max_entropy: float = 0.0

                    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
                        probs = torch.softmax(scores, dim=-1)
                        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).max().item()
                        if ent > self.max_entropy:
                            self.max_entropy = ent
                        return scores

                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(prompt, return_tensors="pt").to(model.device)
                do_sample = temperature > 0.0
                capture = _EntropyCapture()
                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=256,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=[capture],
                    )
                text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                return text, capture.max_entropy

            def _generate(self, msgs: list[dict], temperature: float = 0.0) -> str:
                return self._generate_with_entropy(msgs, temperature)[0]

            def _user_content(self, question: str) -> str:
                if few_shot_retriever is None:
                    return question
                examples_block = few_shot_retriever(question)
                return f"{examples_block}\n\nQuestion: {question}"

            def __call__(self, question: str) -> str:
                return self._generate([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._user_content(question)},
                ])

            def predict_with_entropy(self, question: str) -> tuple[str, float]:
                return self._generate_with_entropy([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._user_content(question)},
                ])

            def repair(self, question: str, failed_sql: str, error: str) -> str:
                return self._generate([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._user_content(question)},
                    {"role": "assistant", "content": failed_sql},
                    {
                        "role": "user",
                        "content": (
                            f"SQLite error: {error}\n"
                            "Fix the SQL using only tables and columns in the schema above. "
                            "Output only the corrected SQL (no explanation)."
                        ),
                    },
                ])

            def vote(self, question: str, num_samples: int, temperature: float = 0.7) -> str:
                """Generate num_samples completions and return the majority-vote answer.

                Voting rule (RS-optimal):
                  - If ≥ ceil(num_samples/2) completions abstain → return [ABSTAIN]
                  - Else execute all non-abstain completions, return the SQL whose
                    result set appears most often (ties broken by first occurrence).
                """
                import math
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._user_content(question)},
                ]
                preds = [self._generate(msgs, temperature=temperature) for _ in range(num_samples)]

                abstain_count = sum(1 for p in preds if p == ABSTAIN_TOKEN or not p)
                if abstain_count >= math.ceil(num_samples / 2):
                    return ABSTAIN_TOKEN

                # Execute non-abstain predictions and vote on result sets.
                # dict maps result-set key → (count, first_sql_that_produced_it)
                result_counts: dict[str, tuple[int, str]] = {}
                for pred in preds:
                    if pred == ABSTAIN_TOKEN or not pred:
                        continue
                    rows, err = _exec_safe(pred)
                    if err is not None:
                        continue
                    key = _normalize_result(rows)
                    if key in result_counts:
                        cnt, first_sql = result_counts[key]
                        result_counts[key] = (cnt + 1, first_sql)
                    else:
                        result_counts[key] = (1, pred)

                if not result_counts:
                    # All non-abstain predictions failed execution — fall back to first non-abstain
                    for pred in preds:
                        if pred != ABSTAIN_TOKEN and pred:
                            return pred
                    return ABSTAIN_TOKEN

                # Return SQL from the most common result set (stable: first occurrence wins ties)
                best_key = max(result_counts, key=lambda k: result_counts[k][0])
                return result_counts[best_key][1]

        return _UnslothPredictor()

    # HF Hub model — load with AutoModelForCausalLM + 4-bit NF4 quantization.
    # Uses the same interface as _UnslothPredictor: supports few_shot_retriever,
    # repair(), vote(), and predict_with_entropy().
    import re as _re
    from transformers import (  # type: ignore[import]
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    hf_model.eval()

    def _strip_fences(text: str) -> str:
        """Extract SQL from markdown code fences if present."""
        m = _re.search(r'```(?:sql)?\s*(.*?)\s*```', text, _re.DOTALL | _re.IGNORECASE)
        return m.group(1).strip() if m else text

    class _HFPredictor:
        """Full-featured HF Hub model predictor with few-shot, repair, vote, and entropy."""

        def _generate_with_entropy(
            self, msgs: list[dict], temperature: float = 0.0
        ) -> tuple[str, float]:
            class _EntropyCapture(LogitsProcessor):
                def __init__(self) -> None:
                    self.max_entropy: float = 0.0

                def __call__(
                    self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor"
                ) -> "torch.FloatTensor":
                    probs = torch.softmax(scores, dim=-1)
                    ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).max().item()
                    if ent > self.max_entropy:
                        self.max_entropy = ent
                    return scores

            prompt = hf_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inp = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
            do_sample = temperature > 0.0
            capture = _EntropyCapture()
            with torch.no_grad():
                out = hf_model.generate(
                    **inp,
                    max_new_tokens=1024,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=hf_tokenizer.eos_token_id,
                    logits_processor=[capture],
                )
            text = hf_tokenizer.decode(
                out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            return _strip_fences(text), capture.max_entropy

        def _generate(self, msgs: list[dict], temperature: float = 0.0) -> str:
            return self._generate_with_entropy(msgs, temperature)[0]

        def _user_content(self, question: str) -> str:
            if few_shot_retriever is None:
                return question
            examples_block = few_shot_retriever(question)
            return f"{examples_block}\n\nQuestion: {question}"

        def __call__(self, question: str) -> str:
            return self._generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_content(question)},
            ])

        def predict_with_entropy(self, question: str) -> tuple[str, float]:
            return self._generate_with_entropy([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_content(question)},
            ])

        def repair(self, question: str, failed_sql: str, error: str) -> str:
            return self._generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_content(question)},
                {"role": "assistant", "content": failed_sql},
                {
                    "role": "user",
                    "content": (
                        f"SQLite error: {error}\n"
                        "Fix the SQL using only tables and columns in the schema above. "
                        "Output only the corrected SQL (no explanation)."
                    ),
                },
            ])

        def vote(self, question: str, num_samples: int, temperature: float = 0.7) -> str:
            import math
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_content(question)},
            ]
            preds = [self._generate(msgs, temperature=temperature) for _ in range(num_samples)]
            abstain_count = sum(1 for p in preds if p == ABSTAIN_TOKEN or not p)
            if abstain_count >= math.ceil(num_samples / 2):
                return ABSTAIN_TOKEN
            result_counts: dict[str, tuple[int, str]] = {}
            for pred in preds:
                if pred == ABSTAIN_TOKEN or not pred:
                    continue
                rows, err = _exec_safe(pred)
                if err is not None:
                    continue
                key = _normalize_result(rows)
                if key in result_counts:
                    cnt, first_sql = result_counts[key]
                    result_counts[key] = (cnt + 1, first_sql)
                else:
                    result_counts[key] = (1, pred)
            if not result_counts:
                for pred in preds:
                    if pred != ABSTAIN_TOKEN and pred:
                        return pred
                return ABSTAIN_TOKEN
            best_key = max(result_counts, key=lambda k: result_counts[k][0])
            return result_counts[best_key][1]

    return _HFPredictor()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    examples: Iterable[EHRSQLExample],
    model_fn: ModelFn,
    verbose: bool = False,
    progress_every: int = 25,
    max_repair_attempts: int = 0,
    num_samples: int = 1,
    predictions_path: "Path | None" = None,
    entropy_threshold: "float | None" = None,
    abstain_on_empty: bool = False,
    abstain_on_error: bool = False,
) -> EvalMetrics:
    """Run model_fn over examples, execute results, compute EX and RS(N).

    max_repair_attempts > 0: retry failed SQL up to N times (requires model_fn.repair()).
    num_samples > 1: self-consistency voting over K samples (requires model_fn.vote()).
    predictions_path: if set, write per-prediction JSONL to this file for debugging.
    entropy_threshold: if set, predictions with max token entropy above this value are
        converted to [ABSTAIN] before the repair loop (mirrors PLUQ winner's entropy filter).
    abstain_on_empty: if True, predictions that execute but return no rows are treated as
        [ABSTAIN] (mirrors PLUQ winner's execution filter).
    abstain_on_error: if True, predictions that fail SQL execution (after all repair
        attempts) are treated as [ABSTAIN] instead of counting as wrong SQL. This converts
        score -N to 0 in official RS(N), which is the biggest RS lever after model quality.
    """
    repair_fn = getattr(model_fn, "repair", None) if max_repair_attempts > 0 else None
    vote_fn = getattr(model_fn, "vote", None) if num_samples > 1 else None

    metrics = EvalMetrics()
    example_list = list(examples)
    total = len(example_list)
    _pred_fh = open(predictions_path, "w") if predictions_path else None  # noqa: SIM115  # closed at end

    for ex in example_list:
        metrics.total += 1
        if ex.is_answerable:
            metrics.answerable += 1
        else:
            metrics.unanswerable += 1

        t0 = time.monotonic()
        entropy = 0.0
        _capture_entropy = (entropy_threshold is not None or predictions_path is not None) \
                           and hasattr(model_fn, "predict_with_entropy")
        if vote_fn is not None:
            predicted_sql = vote_fn(ex.question, num_samples).strip()
        elif _capture_entropy:
            predicted_sql, entropy = model_fn.predict_with_entropy(ex.question)
            predicted_sql = predicted_sql.strip()
            metrics.entropy_samples.append(entropy)
        else:
            predicted_sql = model_fn(ex.question).strip()
        latency_ms = (time.monotonic() - t0) * 1000
        metrics.latency_samples.append(latency_ms)

        # Entropy gate: uncertain prediction → abstain (before repair)
        if entropy_threshold is not None and entropy > entropy_threshold:
            predicted_sql = ABSTAIN_TOKEN
            metrics.entropy_abstentions += 1

        abstained = predicted_sql == ABSTAIN_TOKEN or not predicted_sql

        if ex.is_answerable:
            if abstained:
                metrics.wrong_abstentions += 1
                if verbose:
                    print(f"[WRONG_ABSTAIN] {ex.id}: {ex.question[:60]}")
                if _pred_fh:
                    _pred_fh.write(json.dumps({
                        "id": ex.id, "question": ex.question,
                        "gold_sql": ex.gold_sql, "gold_err": None,
                        "gold_status": "unknown",
                        "predicted_sql": None, "pred_err": None,
                        "outcome": "WRONG_ABSTAIN", "latency_ms": round(latency_ms, 1),
                        "entropy": round(entropy, 4),
                    }) + "\n")
            else:
                gold_rows, gold_err = _exec_safe(post_process_sql(ex.gold_sql))

                # Track gold SQL validity — critical for understanding true EX signal.
                # On MIMIC-IV-Demo, 48.5% of gold SQL errors (missing tables/columns from
                # MIMIC-III) and 30.1% returns empty (patient not in 100-patient demo).
                # Only the 21.4% "gold_exec_valid" cases produce meaningful EX signal.
                if gold_err is not None:
                    metrics.gold_exec_error += 1
                elif gold_rows:
                    metrics.gold_exec_valid += 1
                else:
                    metrics.gold_exec_empty += 1

                pred_rows, pred_err = _exec_safe(post_process_sql(predicted_sql))

                # Execution filter: empty result → abstain (mirrors PLUQ winner)
                if abstain_on_empty and not pred_rows and pred_err is None:
                    predicted_sql = ABSTAIN_TOKEN
                    abstained = True
                    metrics.empty_abstentions += 1

                if abstained:
                    metrics.wrong_abstentions += 1
                    outcome = "WRONG_ABSTAIN_EMPTY"
                    if verbose:
                        print(f"[WRONG_ABSTAIN_EMPTY] {ex.id}: SQL executed empty → abstained")
                    if _pred_fh:
                        _pred_fh.write(json.dumps({
                            "id": ex.id, "question": ex.question,
                            "gold_sql": ex.gold_sql, "gold_err": None,
                            "gold_status": "unknown",
                            "predicted_sql": None, "pred_err": None,
                            "outcome": outcome, "latency_ms": round(latency_ms, 1),
                            "entropy": round(entropy, 4),
                        }) + "\n")
                    continue

                if pred_err is not None and repair_fn is not None:
                    for _attempt in range(max_repair_attempts):
                        metrics.repair_attempts += 1
                        repaired = repair_fn(ex.question, predicted_sql, pred_err).strip()
                        if repaired == ABSTAIN_TOKEN or not repaired:
                            break
                        pred_rows, pred_err = _exec_safe(repaired)
                        if pred_err is None:
                            metrics.repair_successes += 1
                            predicted_sql = repaired
                            if verbose:
                                print(f"  [REPAIRED attempt {_attempt+1}] {ex.id}")
                            break
                        predicted_sql = repaired

                # Execution error filter: abstain if SQL still fails after all repairs.
                # Converts score -N to 0 in official RS(N), saving N pts per failed query.
                if abstain_on_error and pred_err is not None:
                    predicted_sql = ABSTAIN_TOKEN
                    abstained = True
                    metrics.wrong_abstentions += 1
                    outcome = "WRONG_ABSTAIN_ERROR"
                    if verbose:
                        print(f"[WRONG_ABSTAIN_ERROR] {ex.id}: {pred_err[:60]}")
                    if _pred_fh:
                        _pred_fh.write(json.dumps({
                            "id": ex.id, "question": ex.question,
                            "gold_sql": ex.gold_sql, "gold_err": gold_err,
                            "gold_status": "error" if gold_err else ("valid" if gold_rows else "empty"),
                            "predicted_sql": None, "pred_err": pred_err,
                            "outcome": outcome, "latency_ms": round(latency_ms, 1),
                            "entropy": round(entropy, 4),
                        }) + "\n")
                    continue

                # Pass gold_err so results_match never counts error==error as correct.
                is_correct = pred_err is None and results_match(pred_rows, gold_rows, gold_err=gold_err)
                if is_correct:
                    metrics.correct_answers += 1
                    outcome = "CORRECT"
                    if verbose:
                        print(f"[CORRECT] {ex.id}")
                else:
                    outcome = "WRONG"
                    if verbose:
                        reason = pred_err or ("gold_error: " + str(gold_err)[:60]) or "result mismatch"
                        print(f"[WRONG] {ex.id}: {reason[:80]}")

                if _pred_fh:
                    _pred_fh.write(json.dumps({
                        "id": ex.id,
                        "question": ex.question,
                        "gold_sql": ex.gold_sql,
                        "gold_err": gold_err,
                        "gold_status": "valid" if (not gold_err and gold_rows) else ("empty" if not gold_err else "error"),
                        "predicted_sql": predicted_sql,
                        "pred_err": pred_err,
                        "outcome": outcome,
                        "latency_ms": round(latency_ms, 1),
                        "entropy": round(entropy, 4),
                    }) + "\n")
        else:
            # Unanswerable question
            if abstained:
                metrics.correct_abstentions += 1
                outcome = "CORRECT_ABSTAIN"
                if verbose:
                    print(f"[CORRECT_ABSTAIN] {ex.id}", flush=True)
            else:
                metrics.wrong_answers += 1
                outcome = "HALLUCINATED_SQL"
                if verbose:
                    print(f"[HALLUCINATED_SQL] {ex.id}: {predicted_sql[:80]}", flush=True)

            if _pred_fh:
                _pred_fh.write(json.dumps({
                    "id": ex.id,
                    "question": ex.question,
                    "gold_sql": None,
                    "gold_err": None,
                    "gold_status": "unanswerable",
                    "predicted_sql": predicted_sql if not abstained else None,
                    "pred_err": None,
                    "outcome": outcome,
                    "latency_ms": round(latency_ms, 1),
                    "entropy": round(entropy, 4),
                }) + "\n")

        if metrics.total % progress_every == 0:
            elapsed_ms = sum(metrics.latency_samples)
            avg_ms = elapsed_ms / metrics.total
            remaining = total - metrics.total
            eta_min = remaining * avg_ms / 60_000
            repair_str = (
                f" | repairs {metrics.repair_successes}/{metrics.repair_attempts}"
                if metrics.repair_attempts > 0 else ""
            )
            print(
                f"[{metrics.total}/{total}] EX so far: {metrics.ex:.3f} | "
                f"avg {avg_ms:.0f} ms/ex | ETA {eta_min:.0f} min{repair_str}",
                flush=True,
            )

    if _pred_fh:
        _pred_fh.close()

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import warnings

    # Silence the noisy transformers max_new_tokens/max_length warning per generate() call
    warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`")

    parser = argparse.ArgumentParser(description="EHRSQL evaluation harness")
    parser.add_argument("split", help="Path to EHRSQL split JSON (test or dev)")
    parser.add_argument(
        "--model",
        default=config.INFERENCE_MODEL,
        help="HuggingFace model name for baseline evaluation",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output", help="Write JSON metrics to this file", default=None
    )
    parser.add_argument(
        "--repair", action="store_true",
        help=f"Enable execution-guided repair loop (up to {config.MAX_REPAIR_ATTEMPTS} retries per question)",
    )
    parser.add_argument(
        "--few-shot", default=None, metavar="TRAIN_JSON",
        help="Path to EHRSQL train split; enables few-shot retrieval",
    )
    parser.add_argument(
        "--few-shot-k", type=int, default=2, metavar="K",
        help="Number of few-shot examples to retrieve per question (default: 2)",
    )
    parser.add_argument(
        "--retrieval-mode", default="hybrid",
        choices=["bm25", "embed", "hybrid", "template"],
        help=(
            "Retrieval mode: hybrid (default, BM25+embed RRF), bm25, embed, "
            "or template (medium-label LogReg → cosine within class, ~85-90%% Hit@1 with --retrieval-aug)"
        ),
    )
    parser.add_argument(
        "--retrieval-aug", default=None, metavar="AUG_DIR",
        help=(
            "Path to train_aug split directory. When set, aug examples are added to the retrieval "
            "corpus (~40K total). Dramatically improves template mode and hybrid diversity."
        ),
    )
    parser.add_argument(
        "--embed-cache", default=None,
        help="Path to precomputed embedding .npy cache (auto-derived if not set)",
    )
    parser.add_argument(
        "--classifier-cache", default=None,
        help="Path to template LogReg classifier .pkl (default: data/ehrsql2024/mimic_iv/template_classifier_medium[_combined].pkl)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Self-consistency voting: generate N completions and vote on majority result set (default: 1 = disabled)",
    )
    parser.add_argument(
        "--save-predictions", default=None, metavar="PRED_JSONL",
        help="Write per-prediction JSONL (id, question, gold_sql, gold_status, predicted_sql, outcome, entropy) for debugging",
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=None, metavar="FLOAT",
        help=(
            "Max token entropy threshold: predictions with entropy above this are converted to "
            "[ABSTAIN] before the repair loop. Tune on valid set — winner used top 7%% percentile. "
            "Use --save-predictions on valid set first to see the entropy distribution."
        ),
    )
    parser.add_argument(
        "--abstain-on-empty", action="store_true",
        help=(
            "Treat predictions that execute successfully but return no rows as [ABSTAIN]. "
            "Mirrors the PLUQ winner's execution filter. Safe for unanswerable questions; "
            "may slightly increase wrong_abstentions on answerable questions."
        ),
    )
    parser.add_argument(
        "--abstain-on-error", action="store_true",
        help=(
            "Treat predictions that fail SQL execution (after all repair attempts) as [ABSTAIN] "
            "instead of counting as wrong SQL. In official RS(N), this converts score -N to 0. "
            "Crucial for RS(10): each error avoided saves 10 points."
        ),
    )
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")

    print(f"Loading EHRSQL split: {split_path}")
    examples = load_ehrsql_split(split_path)
    print(f"  {len(examples)} examples loaded")

    few_shot_retriever = None
    if args.few_shot:
        train_path = Path(args.few_shot)
        mode = args.retrieval_mode
        embed_cache = Path(args.embed_cache) if args.embed_cache else None
        clf_cache = Path(args.classifier_cache) if args.classifier_cache else None
        aug_path = Path(args.retrieval_aug) if args.retrieval_aug else None
        print(f"Building {mode.upper()} few-shot index from: {train_path}")
        few_shot_retriever = build_few_shot_retriever(
            train_path, top_k=args.few_shot_k, mode=mode, embed_cache=embed_cache,
            classifier_cache=clf_cache, aug_path=aug_path,
        )
        print(f"  {mode.upper()} index built.")

    print(f"Loading model: {args.model}")
    model_fn = run_hf_baseline(args.model, few_shot_retriever=few_shot_retriever)

    max_repairs = config.MAX_REPAIR_ATTEMPTS if args.repair else 0
    if args.repair:
        has_repair = hasattr(model_fn, "repair")
        print(f"Repair loop: {'enabled' if has_repair else 'NOT supported for this model type'} "
              f"(max {max_repairs} attempts per question)")

    if args.num_samples > 1:
        has_vote = hasattr(model_fn, "vote")
        print(f"Self-consistency voting: {'enabled' if has_vote else 'NOT supported'} "
              f"(K={args.num_samples} samples per question, temperature=0.7)")

    pred_path = Path(args.save_predictions) if args.save_predictions else None
    if pred_path:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Per-prediction log: {pred_path}")

    if args.entropy_threshold is not None:
        print(f"Entropy filtering: threshold={args.entropy_threshold:.4f} (predictions above this → [ABSTAIN])")
    if args.abstain_on_empty:
        print("Execution filtering: empty SQL result → [ABSTAIN]")
    if args.abstain_on_error:
        print("Execution filtering: SQL error (after repairs) → [ABSTAIN]")

    print("Running evaluation...")
    metrics = evaluate(
        examples, model_fn,
        verbose=args.verbose,
        max_repair_attempts=max_repairs,
        num_samples=args.num_samples,
        predictions_path=pred_path,
        entropy_threshold=args.entropy_threshold,
        abstain_on_empty=args.abstain_on_empty,
        abstain_on_error=args.abstain_on_error,
    )

    summary = metrics.summary()
    print("\n=== Results ===")
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()
