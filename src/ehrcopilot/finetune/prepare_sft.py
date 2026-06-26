"""Prepare SFT training data from EHRSQL dataset.

Formats each example as a chat template:
  system  = full MIMIC-IV-Demo schema context (FK hints included)
  user    = NL question
  assistant = gold SQL (MIMIC-IV-canonicalized)  OR  [ABSTAIN]

Key changes vs v1:
  - Filters examples referencing MIMIC-III-only tables (cost, inputevents_cv, outputevents)
  - Applies 4 MIMIC-III→IV column renames to gold SQL before training
  - Uses full schema (not linked subset) to match eval harness context
  - Adds --valid for unanswerable source; oversamples to ~20% of dataset
  - Output: sft_train_v2.jsonl (default)

Usage:
    python -m ehrcopilot.finetune.prepare_sft \\
        --train data/ehrsql/ehrsql/mimic_iii/train.json \\
        --valid data/ehrsql/ehrsql/mimic_iii/valid.json \\
        --output data/ehrsql/sft_train_v2.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.harness import EHRSQLExample, load_ehrsql_split

ABSTAIN_TOKEN = "[ABSTAIN]"

_SYSTEM_TEMPLATE = """\
You are a clinical analytics SQL expert. Convert the user's question into a valid SQLite SELECT query.
If the question cannot be answered with the available data, output exactly: [ABSTAIN]

{schema}"""

# Tables in the MIMIC-III EHRSQL gold SQL that don't exist in MIMIC-IV-Demo.
# Training on these teaches the model to hallucinate non-existent tables.
_INCOMPATIBLE_TABLES: frozenset[str] = frozenset(
    {"cost", "inputevents_cv", "outputevents"}
)

# MIMIC-III column names renamed in MIMIC-IV. Must mirror harness._MIMIC_RENAMES.
_MIMIC_RENAMES: list[tuple[str, str]] = [
    (r"\bicustay_id\b", "stay_id"),
    (r"\bstartdate\b", "starttime"),
    (r"\bicd9_code\b", "icd_code"),
    (r"\bshort_title\b", "long_title"),
]

# Unanswerable target fraction of final SFT dataset (test split has 32.9%)
_UNANSWERABLE_TARGET_RATIO = 0.20


def _has_incompatible_tables(sql: str) -> bool:
    found = set(re.findall(r"(?:from|join)\s+(\w+)", sql.lower()))
    return bool(found & _INCOMPATIBLE_TABLES)


def _canonicalize_sql(sql: str) -> str:
    for pattern, repl in _MIMIC_RENAMES:
        sql = re.sub(pattern, repl, sql, flags=re.IGNORECASE)
    return sql


def format_example(
    ex: EHRSQLExample,
    max_tokens: int = config.MAX_SEQ_LENGTH,
    schema_text: str | None = None,
) -> dict | None:
    """Format a single EHRSQL example as a chat-style training dict.

    Returns None if the example should be filtered (incompatible tables,
    exceeds token budget).
    """
    if ex.is_answerable:
        if _has_incompatible_tables(ex.gold_sql):
            return None
        assistant_content = _canonicalize_sql(ex.gold_sql)
    else:
        assistant_content = ABSTAIN_TOKEN

    if schema_text is None:
        schema_text = config.schema_to_prompt()

    system_content = _SYSTEM_TEMPLATE.format(schema=schema_text)

    # Rough token budget: 1 token ≈ 4 chars
    total_chars = len(system_content) + len(ex.question) + len(assistant_content)
    if total_chars > max_tokens * 4:
        return None

    return {
        "id": ex.id,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": ex.question},
            {"role": "assistant", "content": assistant_content},
        ],
        "is_answerable": ex.is_answerable,
    }


def prepare(
    train_path: Path,
    output_path: Path,
    valid_path: Path | None = None,
    seed: int = 42,
) -> dict[str, int]:
    random.seed(seed)

    train_examples = load_ehrsql_split(train_path)
    print(f"Loaded {len(train_examples)} examples from {train_path}")

    # Collect unanswerable examples from valid split (if provided) + train split
    unanswerable_examples: list[EHRSQLExample] = [
        e for e in train_examples if not e.is_answerable
    ]
    if valid_path and valid_path.exists():
        valid_examples = load_ehrsql_split(valid_path)
        unanswerable_examples += [e for e in valid_examples if not e.is_answerable]
        print(f"  Added {len([e for e in valid_examples if not e.is_answerable])} unanswerable from {valid_path}")

    answerable_examples = [e for e in train_examples if e.is_answerable]
    print(f"  Answerable: {len(answerable_examples)}, Unanswerable unique: {len(unanswerable_examples)}")

    # Pre-compute full schema text once (same for all examples — matches eval harness)
    schema_text = config.schema_to_prompt()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {
        "total_input": 0,
        "kept": 0,
        "filtered_incompatible": 0,
        "filtered_too_long": 0,
        "answerable": 0,
        "unanswerable": 0,
    }

    formatted_answerable: list[dict] = []
    for ex in answerable_examples:
        stats["total_input"] += 1
        if _has_incompatible_tables(ex.gold_sql):
            stats["filtered_incompatible"] += 1
            continue
        formatted = format_example(ex, schema_text=schema_text)
        if formatted is None:
            stats["filtered_too_long"] += 1
            continue
        formatted_answerable.append(formatted)

    # Oversample unanswerable to reach target ratio
    target_unans_count = int(len(formatted_answerable) * _UNANSWERABLE_TARGET_RATIO / (1 - _UNANSWERABLE_TARGET_RATIO))
    formatted_unans_base: list[dict] = []
    for ex in unanswerable_examples:
        stats["total_input"] += 1
        formatted = format_example(ex, schema_text=schema_text)
        if formatted is not None:
            formatted_unans_base.append(formatted)

    if formatted_unans_base:
        repeat = (target_unans_count // len(formatted_unans_base)) + 1
        pool = (formatted_unans_base * repeat)[:target_unans_count]
    else:
        pool = []

    print(f"  Answerable formatted: {len(formatted_answerable)}")
    print(f"  Unanswerable unique formatted: {len(formatted_unans_base)} → oversampled to {len(pool)}")

    all_examples = formatted_answerable + pool
    random.shuffle(all_examples)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
            stats["kept"] += 1
            if ex["is_answerable"]:
                stats["answerable"] += 1
            else:
                stats["unanswerable"] += 1

    unans_pct = 100 * stats["unanswerable"] / stats["kept"] if stats["kept"] else 0
    print(f"\nSFT data written to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  unanswerable_pct: {unans_pct:.1f}%")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="EHRSQL train split JSON")
    parser.add_argument("--valid", default=None, help="EHRSQL valid split JSON (unanswerable source)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(
        train_path=Path(args.train),
        output_path=Path(args.output),
        valid_path=Path(args.valid) if args.valid else None,
        seed=args.seed,
    )
