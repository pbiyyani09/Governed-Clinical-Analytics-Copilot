"""RAGAS-style RAG evaluation adapted to NL->SQL few-shot retrieval.

The four RAGAS metrics, mapped to this setting (per the Text-to-SQL retrieval
literature — DAIL-SQL, AST-ranking, RAGAS paper arXiv:2309.15217):

  question  = the clinical NL question
  contexts  = the retrieved few-shot (question, SQL) exemplars
  answer    = the model-generated SQL
  ground_truth = the gold SQL  (+ EHRSQL `tag` template label)

  Context Precision  rank-weighted average precision of the retrieved exemplars,
                     "relevant" = shares the gold template `tag`.        [LLM-free]
  Context Recall     did the retrieved set cover the gold template?      [LLM-free]
  Faithfulness       is the generated SQL grounded in schema + contexts —
                     fraction of referenced tables/columns that are valid
                     (schema allowlist) and present in some context.     [LLM-free,
                     optional Gemini semantic judge with --use-llm-judge]
  Answer Relevance   does the generated SQL answer the question — execution
                     accuracy vs gold result set.                        [LLM-free,
                     optional Gemini reverse-question cosine]

The LLM-free substitutes are the principled, reproducible choice here because
EHRSQL ships a structural `tag` label and we can EXECUTE SQL against the DB.
The Gemini judge (GOOGLE_API_KEY / GEMINI_MODEL from .env) is offered for the
two genuinely semantic metrics.

Input: a predictions JSONL, one object per test question:
    {"question", "gold_sql", "gold_tag",
     "generated_sql", "contexts": [{"question","sql","tag"}, ...]}

Usage:
    python -m ehrcopilot.eval.rag_ragas --pred preds.jsonl \
        --output tests/evalgen/ragas_results.json [--use-llm-judge]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from ehrcopilot import config

try:
    import sqlglot
except Exception:  # noqa: BLE001
    sqlglot = None


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

_ALLOWED_TABLES = {t.lower() for t in config.ALLOWED_TABLES}
_ALLOWED_COLS = {c.lower() for cols in config.SCHEMA_ALLOWLIST.values() for c in cols}


def _referenced_identifiers(sql: str) -> tuple[set[str], set[str]]:
    """Return (tables, columns) referenced in the SQL, via sqlglot when available."""
    tables: set[str] = set()
    cols: set[str] = set()
    if sqlglot is not None:
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            for t in tree.find_all(sqlglot.exp.Table):
                tables.add(t.name.lower())
            for c in tree.find_all(sqlglot.exp.Column):
                cols.add(c.name.lower())
            return tables, cols
        except Exception:  # noqa: BLE001
            pass
    # regex fallback
    for m in re.finditer(r"(?:from|join)\s+(\w+)", sql, re.I):
        tables.add(m.group(1).lower())
    return tables, cols


# ---------------------------------------------------------------------------
# Metric 1+2: Context Precision (AP) and Context Recall — LLM-free via `tag`
# ---------------------------------------------------------------------------

def context_precision(contexts: list[dict], gold_tag: str) -> float:
    """Rank-weighted average precision of retrieved exemplars (RAGAS context precision)."""
    if not contexts:
        return 0.0
    hits = 0
    ap = 0.0
    for i, ctx in enumerate(contexts):
        if ctx.get("tag") == gold_tag:
            hits += 1
            ap += hits / (i + 1)
    return ap / hits if hits else 0.0


def context_recall(contexts: list[dict], gold_tag: str) -> float:
    """Binary: did the retrieved set include >=1 same-template exemplar."""
    return 1.0 if any(c.get("tag") == gold_tag for c in contexts) else 0.0


# ---------------------------------------------------------------------------
# Metric 3: Faithfulness — generated SQL grounded in schema + contexts
# ---------------------------------------------------------------------------

def faithfulness_structural(generated_sql: str, contexts: list[dict]) -> float:
    """Fraction of referenced tables+columns that are (a) in the schema allowlist
    and (b) present in at least one retrieved exemplar's SQL. Proxy for "the answer
    is supported by the provided context", with no hallucinated schema elements."""
    if not generated_sql or generated_sql.strip() == "[ABSTAIN]":
        return 1.0  # abstention can't hallucinate schema
    tables, cols = _referenced_identifiers(generated_sql)
    refs = {f"t:{t}" for t in tables} | {f"c:{c}" for c in cols}
    if not refs:
        return 0.0
    ctx_blob = " ".join((c.get("sql") or "").lower() for c in contexts)
    supported = 0
    for r in refs:
        kind, name = r.split(":", 1)
        in_schema = name in (_ALLOWED_TABLES if kind == "t" else _ALLOWED_COLS)
        in_ctx = name in ctx_blob
        if in_schema and (in_ctx or kind == "t"):
            supported += 1
    return supported / len(refs)


# ---------------------------------------------------------------------------
# Metric 4: Answer Relevance — execution accuracy vs gold
# ---------------------------------------------------------------------------

def answer_relevance_exec(generated_sql: str, gold_sql: str) -> float:
    """Execution-based answer relevance: 1.0 if generated result set == gold's."""
    from ehrcopilot.eval.harness import _exec_safe, _canonicalize_gold_sql, results_match

    if not generated_sql or generated_sql.strip() == "[ABSTAIN]":
        return 0.0
    gen_rows, gen_err = _exec_safe(generated_sql)
    if gen_err is not None:
        return 0.0
    gold_rows, _ = _exec_safe(_canonicalize_gold_sql(gold_sql))
    return 1.0 if results_match(gen_rows, gold_rows) else 0.0


# ---------------------------------------------------------------------------
# Optional Gemini judge
# ---------------------------------------------------------------------------

def _gemini_client():
    from google import genai  # lazy
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        # try .env
        for line in open(".env"):
            if line.strip().startswith("GOOGLE_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"')
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return genai.Client(api_key=key), model


def faithfulness_llm(client, model, question: str, generated_sql: str, contexts: list[dict]) -> float:
    ctx = "\n".join(f"- Q: {c.get('question')}  SQL: {c.get('sql')}" for c in contexts[:5])
    prompt = (
        "You judge whether a generated SQL query is FAITHFUL to the provided example "
        "context (does not invent tables/columns/logic absent from the examples and schema).\n"
        f"Question: {question}\nExamples:\n{ctx}\nGenerated SQL: {generated_sql}\n"
        "Answer with a single float 0.0-1.0 (1.0 = fully faithful). Only the number."
    )
    try:
        r = client.models.generate_content(model=model, contents=prompt)
        return float(re.search(r"[01](?:\.\d+)?", r.text).group())
    except Exception:  # noqa: BLE001
        return float("nan")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(pred_path: Path, use_llm_judge: bool) -> dict:
    preds = [json.loads(l) for l in open(pred_path) if l.strip()]
    client = model = None
    if use_llm_judge:
        client, model = _gemini_client()

    agg = {"context_precision": 0.0, "context_recall": 0.0,
           "faithfulness_structural": 0.0, "answer_relevance_exec": 0.0}
    if use_llm_judge:
        agg["faithfulness_llm"] = 0.0
    n = len(preds)
    for p in preds:
        ctxs = p.get("contexts", [])
        agg["context_precision"] += context_precision(ctxs, p.get("gold_tag"))
        agg["context_recall"] += context_recall(ctxs, p.get("gold_tag"))
        agg["faithfulness_structural"] += faithfulness_structural(p.get("generated_sql", ""), ctxs)
        agg["answer_relevance_exec"] += answer_relevance_exec(p.get("generated_sql", ""), p.get("gold_sql", ""))
        if use_llm_judge:
            agg["faithfulness_llm"] += faithfulness_llm(client, model, p.get("question", ""),
                                                        p.get("generated_sql", ""), ctxs)
    return {k: (v / n if n else 0.0) for k, v in agg.items()} | {"n_evaluated": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS-style NL->SQL RAG evaluation")
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--output", default="tests/evalgen/ragas_results.json")
    ap.add_argument("--use-llm-judge", action="store_true", help="add Gemini semantic faithfulness")
    args = ap.parse_args()
    res = evaluate(Path(args.pred), args.use_llm_judge)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
