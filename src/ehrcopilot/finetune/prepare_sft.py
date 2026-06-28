"""Prepare SFT training data from the EHRSQL-2024 dataset.

Formats each example as a chat sample:
  system    = canonical system prompt + full EHRSQL-2024 schema (config.system_prompt())
  user      = NL question
  assistant = gold SQL  OR  [ABSTAIN]

The input is an EHRSQL-2024 split *directory* (data.json + label.json) — e.g. the official
train, or the augmented train_aug produced by augment_ehrsql2024. Unanswerable examples are
oversampled toward ~20% (the valid/test unanswerable ratio) to strengthen abstention.

NB: this is the 2024 (MIMIC-IV) pipeline. The old MIMIC-III→IV table filtering / column
renames were REMOVED — the 2024 gold SQL is already written against the target schema
(cost / inputevents / outputevents are legitimate 2024 tables).

Usage:
    python -m ehrcopilot.finetune.prepare_sft \\
        --train data/ehrsql2024/mimic_iv/train_aug \\
        --valid data/ehrsql2024/mimic_iv/valid \\
        --output data/ehrsql2024/sft_train_aug.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.harness import EHRSQLExample, load_ehrsql_split

ABSTAIN_TOKEN = config.ABSTAIN_TOKEN

# Unanswerable target fraction of final SFT dataset (valid/test split is ~20%).
_UNANSWERABLE_TARGET_RATIO = 0.20


def format_example(
    ex: EHRSQLExample,
    system_text: str,
    max_tokens: int = config.MAX_SEQ_LENGTH,
) -> dict | None:
    """Format one example as a chat-style training dict, or None if over the token budget."""
    assistant_content = ex.gold_sql if ex.is_answerable else ABSTAIN_TOKEN

    # Rough token budget: 1 token ≈ 4 chars
    if (len(system_text) + len(ex.question) + len(assistant_content)) > max_tokens * 4:
        return None

    return {
        "id": ex.id,
        "messages": [
            {"role": "system", "content": system_text},
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

    unanswerable_examples: list[EHRSQLExample] = [e for e in train_examples if not e.is_answerable]
    if valid_path and Path(valid_path).exists():
        valid_examples = load_ehrsql_split(valid_path)
        added = [e for e in valid_examples if not e.is_answerable]
        unanswerable_examples += added
        print(f"  Added {len(added)} unanswerable from {valid_path}")

    answerable_examples = [e for e in train_examples if e.is_answerable]
    print(f"  Answerable: {len(answerable_examples)}, Unanswerable unique: {len(unanswerable_examples)}")

    system_text = config.system_prompt()  # identical to eval (train/inference parity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total_input": 0, "kept": 0, "filtered_too_long": 0, "answerable": 0, "unanswerable": 0}

    formatted_answerable: list[dict] = []
    for ex in answerable_examples:
        stats["total_input"] += 1
        formatted = format_example(ex, system_text)
        if formatted is None:
            stats["filtered_too_long"] += 1
            continue
        formatted_answerable.append(formatted)

    formatted_unans_base: list[dict] = []
    for ex in unanswerable_examples:
        stats["total_input"] += 1
        formatted = format_example(ex, system_text)
        if formatted is not None:
            formatted_unans_base.append(formatted)

    # Oversample unanswerable to reach the target ratio
    target_unans = int(len(formatted_answerable) * _UNANSWERABLE_TARGET_RATIO / (1 - _UNANSWERABLE_TARGET_RATIO))
    if formatted_unans_base:
        repeat = (target_unans // len(formatted_unans_base)) + 1
        pool = (formatted_unans_base * repeat)[:target_unans]
    else:
        pool = []

    print(f"  Answerable formatted: {len(formatted_answerable)}")
    print(f"  Unanswerable unique: {len(formatted_unans_base)} → oversampled to {len(pool)}")

    all_examples = formatted_answerable + pool
    random.shuffle(all_examples)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
            stats["kept"] += 1
            stats["answerable" if ex["is_answerable"] else "unanswerable"] += 1

    unans_pct = 100 * stats["unanswerable"] / stats["kept"] if stats["kept"] else 0
    print(f"\nSFT data written to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  unanswerable_pct: {unans_pct:.1f}%")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="EHRSQL-2024 train split DIR (data.json+label.json)")
    parser.add_argument("--valid", default=None, help="EHRSQL-2024 valid split DIR (extra unanswerable source)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(
        train_path=Path(args.train),
        output_path=Path(args.output),
        valid_path=Path(args.valid) if args.valid else None,
        seed=args.seed,
    )
