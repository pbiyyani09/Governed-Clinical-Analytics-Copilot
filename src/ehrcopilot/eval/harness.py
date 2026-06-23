"""Evaluation harness: Execution Accuracy (EX) and Reliability Score RS(N).

Metrics mirror the EHRSQL 2024 shared task evaluation protocol.

EX  — fraction of answerable questions where the predicted SQL produces the same
      execution result as the gold SQL.

RS(N) — reliability score with penalty N for wrong answers on unanswerable questions:
         RS(N) = (#correct_answers - N * #wrong_abstentions_on_answerable
                  - N * #wrong_answers_on_unanswerable) / total_questions
         (any positive RS(N) is a publishable result — no 2024 shared-task team achieved it)

Reference: Lee et al., EHRSQL: A Practical Text-to-SQL Benchmark for EHRs, NeurIPS 2022.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ehrcopilot import config
from ehrcopilot.db.connection import execute_query

ABSTAIN_TOKEN = "[ABSTAIN]"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EHRSQLExample:
    id: str
    question: str
    gold_sql: str
    is_answerable: bool  # False for the ~1.9K unanswerable questions

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EHRSQLExample":
        sql = d.get("sql") or d.get("query") or ""
        # EHRSQL marks unanswerable questions with query="null" or is_impossible=True
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

    # EX components
    correct_answers: int = 0          # answerable + correct SQL result
    wrong_abstentions: int = 0        # answerable + model abstained
    wrong_answers: int = 0            # unanswerable + model gave SQL (any)
    correct_abstentions: int = 0      # unanswerable + model abstained

    # Latency
    total_latency_ms: float = 0.0
    latency_samples: list[float] = field(default_factory=list)

    @property
    def ex(self) -> float:
        if self.answerable == 0:
            return 0.0
        return self.correct_answers / self.answerable

    def rs(self, n: int) -> float:
        if self.total == 0:
            return 0.0
        penalty = n * (self.wrong_abstentions + self.wrong_answers)
        return (self.correct_answers - penalty) / self.total

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
        return {
            "total": self.total,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "EX": round(self.ex, 4),
            "RS(0)": round(self.rs(0), 4),
            "RS(5)": round(self.rs(5), 4),
            "RS(10)": round(self.rs(10), 4),
            "correct_answers": self.correct_answers,
            "wrong_abstentions": self.wrong_abstentions,
            "wrong_answers_on_unanswerable": self.wrong_answers,
            "correct_abstentions": self.correct_abstentions,
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Execution result comparison
# ---------------------------------------------------------------------------


def _normalize_result(rows: list[dict[str, Any]] | None) -> frozenset[tuple[Any, ...]]:
    """Order-independent comparison of SQL result sets."""
    if not rows:
        return frozenset()
    return frozenset(tuple(sorted(row.items())) for row in rows)


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
) -> bool:
    return _normalize_result(pred_rows) == _normalize_result(gold_rows)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_ehrsql_split(split_path: Path) -> list[EHRSQLExample]:
    """Load EHRSQL JSON split file. Handles both list and dict-keyed formats."""
    with open(split_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [EHRSQLExample.from_dict(d) for d in raw]
    # dict format: {"data": [...]}
    if "data" in raw:
        return [EHRSQLExample.from_dict(d) for d in raw["data"]]
    # fallback: treat as flat dict of id→example
    return [EHRSQLExample.from_dict({**v, "id": k}) for k, v in raw.items()]


# ---------------------------------------------------------------------------
# Model interface — swap in any callable (pipeline, agent, API)
# ---------------------------------------------------------------------------

ModelFn = Callable[[str], str]
"""A function that takes a natural-language question and returns predicted SQL
or the literal string "[ABSTAIN]"."""


def run_hf_baseline(model_name: str = config.INFERENCE_MODEL) -> ModelFn:
    """Return a ModelFn wrapping a HuggingFace text-generation pipeline.

    Uses 4-bit quantization (bitsandbytes NF4) to fit 7B models within 16 GB VRAM.
    """
    import torch
    from transformers import pipeline, BitsAndBytesConfig  # type: ignore[import]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    system_prompt = (
        "You are a clinical analytics assistant. Convert the user's question into "
        "a valid SQLite SELECT query over the MIMIC-IV-Demo database. "
        "If the question cannot be answered with the available data, output exactly: [ABSTAIN]\n\n"
        + config.schema_to_prompt()
    )

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        model_kwargs={"quantization_config": bnb_config},
        max_new_tokens=256,
        temperature=None,
        do_sample=False,
    )

    def _predict(question: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        out = pipe(messages)
        # HF chat pipeline returns list of dicts
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            # Grab the last assistant turn
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    return msg["content"].strip()
        return str(generated).strip()

    return _predict


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    examples: Iterable[EHRSQLExample],
    model_fn: ModelFn,
    verbose: bool = False,
    progress_every: int = 25,
) -> EvalMetrics:
    """Run model_fn over examples, execute results, compute EX and RS(N)."""
    import sys

    metrics = EvalMetrics()
    example_list = list(examples)
    total = len(example_list)

    for ex in example_list:
        metrics.total += 1
        if ex.is_answerable:
            metrics.answerable += 1
        else:
            metrics.unanswerable += 1

        t0 = time.monotonic()
        predicted_sql = model_fn(ex.question).strip()
        latency_ms = (time.monotonic() - t0) * 1000
        metrics.latency_samples.append(latency_ms)

        abstained = predicted_sql == ABSTAIN_TOKEN or not predicted_sql

        if ex.is_answerable:
            if abstained:
                metrics.wrong_abstentions += 1
                if verbose:
                    print(f"[WRONG_ABSTAIN] {ex.id}: {ex.question[:60]}")
            else:
                gold_rows, gold_err = _exec_safe(ex.gold_sql)
                pred_rows, pred_err = _exec_safe(predicted_sql)

                if pred_err is None and results_match(pred_rows, gold_rows):
                    metrics.correct_answers += 1
                    if verbose:
                        print(f"[CORRECT] {ex.id}")
                else:
                    if verbose:
                        reason = pred_err or "result mismatch"
                        print(f"[WRONG] {ex.id}: {reason[:80]}")
        else:
            # Unanswerable question
            if abstained:
                metrics.correct_abstentions += 1
                if verbose:
                    print(f"[CORRECT_ABSTAIN] {ex.id}", flush=True)
            else:
                metrics.wrong_answers += 1
                if verbose:
                    print(f"[HALLUCINATED_SQL] {ex.id}: {predicted_sql[:80]}", flush=True)

        if metrics.total % progress_every == 0:
            elapsed_ms = sum(metrics.latency_samples)
            avg_ms = elapsed_ms / metrics.total
            remaining = total - metrics.total
            eta_min = remaining * avg_ms / 60_000
            print(
                f"[{metrics.total}/{total}] EX so far: {metrics.ex:.3f} | "
                f"avg {avg_ms:.0f} ms/ex | ETA {eta_min:.0f} min",
                flush=True,
            )

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

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
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    print(f"Loading EHRSQL split: {split_path}")
    examples = load_ehrsql_split(split_path)
    print(f"  {len(examples)} examples loaded")

    print(f"Loading model: {args.model}")
    model_fn = run_hf_baseline(args.model)

    print("Running evaluation...")
    metrics = evaluate(examples, model_fn, verbose=args.verbose)

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
