"""Evaluation harness: Execution Accuracy (EX) and Reliability Score RS(N).

Metrics mirror the EHRSQL 2024 shared task evaluation protocol.

EX  — fraction of answerable questions where the predicted SQL produces the same
      execution result as the gold SQL.

RS(N) — reliability score with penalty N for wrong answers on unanswerable questions:
         RS(N) = (#correct_answers + #correct_abstentions
                  - N * #wrong_answers_on_unanswerable) / total_questions
         Wrong abstentions (abstaining on answerable questions) score 0, not −N.
         Reference: Lee et al., EHRSQL NeurIPS 2022 and EHRSQL 2024 shared task.

Reference: Lee et al., EHRSQL: A Practical Text-to-SQL Benchmark for EHRs, NeurIPS 2022.
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

# MIMIC-III column names that were renamed in MIMIC-IV.
# Applied to gold SQL before execution so the gold standard is valid against
# the MIMIC-IV-Demo database. Without this, 71% of gold SQL fails, causing
# frozenset()==frozenset() false positives that inflate EX.
_MIMIC_RENAMES: list[tuple[str, str]] = [
    (r"\bicustay_id\b", "stay_id"),
    (r"\bstartdate\b", "starttime"),
    (r"\bicd9_code\b", "icd_code"),
    (r"\bshort_title\b", "long_title"),
]


def _canonicalize_gold_sql(sql: str) -> str:
    for pattern, repl in _MIMIC_RENAMES:
        sql = re.sub(pattern, repl, sql, flags=re.IGNORECASE)
    return sql


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

    # Repair loop stats (non-zero only when repair is enabled)
    repair_attempts: int = 0
    repair_successes: int = 0

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
        return (self.correct_answers + self.correct_abstentions - n * self.wrong_answers) / self.total

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
        if self.repair_attempts > 0:
            d["repair_attempts"] = self.repair_attempts
            d["repair_successes"] = self.repair_successes
        return d


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
# BM25 few-shot retriever (RAG)
# ---------------------------------------------------------------------------


def build_few_shot_retriever(
    train_path: Path,
    top_k: int = 2,
    max_sql_chars: int = 120,
) -> "Callable[[str], str]":
    """Build a BM25 retriever over train examples.

    Returns a function (question) -> few_shot_block (str) ready to prepend
    into the user message. Caps each SQL at max_sql_chars to stay within the
    1536-token model context (schema uses ~630 tokens, leaving ~650 for
    question + few-shot + 256 generation tokens).
    """
    from rank_bm25 import BM25Okapi  # type: ignore[import]

    examples = [e for e in load_ehrsql_split(train_path) if e.is_answerable and e.gold_sql]
    tokenized = [e.question.lower().split() for e in examples]
    bm25 = BM25Okapi(tokenized)
    gold_sqls = [_canonicalize_gold_sql(e.gold_sql) for e in examples]
    questions = [e.question for e in examples]

    def _retrieve(question: str) -> str:
        scores = bm25.get_scores(question.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        lines = ["Similar examples:"]
        for idx in top_idx:
            sql = gold_sqls[idx]
            if len(sql) > max_sql_chars:
                sql = sql[:max_sql_chars] + "..."
            lines.append(f"Q: {questions[idx]}")
            lines.append(f"SQL: {sql}")
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

    system_prompt = (
        "You are a clinical analytics assistant. Convert the user's question into "
        "a valid SQLite SELECT query over the MIMIC-IV-Demo database. "
        "If the question cannot be answered with the available data, output exactly: [ABSTAIN]\n\n"
        + config.schema_to_prompt()
    )

    import os as _os
    use_unsloth = (
        _os.path.isdir(model_name)
        or model_name.startswith("./")
        or model_name.startswith("../")
        or model_name.startswith("/")
        or model_name.startswith("checkpoints/")
        or model_name.startswith("models/")
        # Gemma 3 is a Gemma3ForConditionalGeneration checkpoint that the plain
        # text-generation pipeline can't load; route hub Gemma/Unsloth repos
        # through Unsloth FastModel (which handles it) instead.
        or "gemma" in model_name.lower()
        or model_name.startswith("unsloth/")
    )

    if use_unsloth:
        # Load via Unsloth — works for LoRA adapters without bitsandbytes at forward time.
        # Gemma 3 is multimodal — load via FastModel (aliased for stable call sites).
        from unsloth import FastModel as FastLanguageModel  # type: ignore[import]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        class _UnslothPredictor:
            """Callable model function with repair and optional few-shot RAG support."""

            def _generate(self, msgs: list[dict], temperature: float = 0.0) -> str:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(prompt, return_tensors="pt").to(model.device)
                do_sample = temperature > 0.0
                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=256,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

                # Execute non-abstain predictions and vote on result sets
                result_counts: list[tuple[frozenset, str]] = []
                for pred in preds:
                    if pred == ABSTAIN_TOKEN or not pred:
                        continue
                    rows, err = _exec_safe(pred)
                    if err is not None:
                        continue
                    key = _normalize_result(rows)
                    # Find if this result set was seen before
                    for i, (existing_key, _) in enumerate(result_counts):
                        if existing_key == key:
                            result_counts[i] = (key, result_counts[i][1])
                            break
                    else:
                        result_counts.append((key, pred))

                if not result_counts:
                    # All non-abstain predictions failed execution — fall back to first non-abstain
                    for pred in preds:
                        if pred != ABSTAIN_TOKEN and pred:
                            return pred
                    return ABSTAIN_TOKEN

                # Return SQL from the most common result set (stable: first occurrence wins ties)
                best_key = max(
                    (k for k, _ in result_counts),
                    key=lambda k: sum(1 for rk, _ in result_counts if rk == k),
                )
                return next(sql for k, sql in result_counts if k == best_key)

        return _UnslothPredictor()

    # HF Hub model — use bitsandbytes pipeline
    from transformers import pipeline, BitsAndBytesConfig  # type: ignore[import]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
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
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
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
    max_repair_attempts: int = 0,
    num_samples: int = 1,
) -> EvalMetrics:
    """Run model_fn over examples, execute results, compute EX and RS(N).

    max_repair_attempts > 0: retry failed SQL up to N times (requires model_fn.repair()).
    num_samples > 1: self-consistency voting over K samples (requires model_fn.vote()).
    """
    import sys

    repair_fn = getattr(model_fn, "repair", None) if max_repair_attempts > 0 else None
    vote_fn = getattr(model_fn, "vote", None) if num_samples > 1 else None

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
        if vote_fn is not None:
            predicted_sql = vote_fn(ex.question, num_samples).strip()
        else:
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
                gold_rows, gold_err = _exec_safe(_canonicalize_gold_sql(ex.gold_sql))
                pred_rows, pred_err = _exec_safe(predicted_sql)

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
            repair_str = (
                f" | repairs {metrics.repair_successes}/{metrics.repair_attempts}"
                if metrics.repair_attempts > 0 else ""
            )
            print(
                f"[{metrics.total}/{total}] EX so far: {metrics.ex:.3f} | "
                f"avg {avg_ms:.0f} ms/ex | ETA {eta_min:.0f} min{repair_str}",
                flush=True,
            )

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
        help="Path to EHRSQL train.json; enables BM25 few-shot retrieval (top-2 similar examples per question)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Self-consistency voting: generate N completions and vote on majority result set (default: 1 = disabled)",
    )
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    print(f"Loading EHRSQL split: {split_path}")
    examples = load_ehrsql_split(split_path)
    print(f"  {len(examples)} examples loaded")

    few_shot_retriever = None
    if args.few_shot:
        train_path = Path(args.few_shot)
        print(f"Building BM25 few-shot index from: {train_path}")
        few_shot_retriever = build_few_shot_retriever(train_path)
        print("  BM25 index built.")

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

    print("Running evaluation...")
    metrics = evaluate(
        examples, model_fn,
        verbose=args.verbose,
        max_repair_attempts=max_repairs,
        num_samples=args.num_samples,
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
