"""Build SFT training data for fine-tuning OmniSQL-7B on EHRSQL 2024 MIMIC-IV.

Loads train + valid + train_aug splits, formats each example as a chat message
(system + user + assistant), and oversamples unanswerable examples 3× to match
the ~20% unanswerable rate seen in test (vs 8.9% in raw combined data).

Output: data/sft/omnisql_sft_train.jsonl
  Each line: {"messages": [{"role": ..., "content": ...}, ...]}

Data composition (default):
  Answerable  SQL:     ~40,961 examples (train + valid + train_aug)
  Unanswerable [ABSTAIN]: 4,015 × 3 = 12,045 examples
  Total: ~53,006 examples  |  unanswerable ratio: 22.7%

Usage:
    python scripts/build_omnisql_sft_data.py
    python scripts/build_omnisql_sft_data.py --no-aug          # skip train_aug
    python scripts/build_omnisql_sft_data.py --oversample 5    # 5× unanswerable
    python scripts/build_omnisql_sft_data.py --max-aug 10000   # cap train_aug rows
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehrcopilot import config
from ehrcopilot.eval.harness import load_ehrsql_split

ABSTAIN = "[ABSTAIN]"
DATA_DIR = ROOT / "data" / "ehrsql2024" / "mimic_iv"


def format_example(question: str, answer: str, system_prompt: str) -> dict:
    """Format a single question+answer pair as a chat message dict."""
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_dataset(
    splits: list[Path],
    system_prompt: str,
    oversample: int = 3,
    max_aug: int | None = None,
    seed: int = 42,
) -> list[dict]:
    answerable:   list[dict] = []
    unanswerable: list[dict] = []

    for split_path in splits:
        examples = load_ehrsql_split(split_path)
        is_aug = "train_aug" in str(split_path)
        for ex in examples:
            answer = ABSTAIN if not ex.is_answerable else ex.gold_sql
            record = format_example(ex.question, answer, system_prompt)
            if ex.is_answerable:
                answerable.append(record)
            else:
                unanswerable.append(record)
        tag = split_path.name if split_path.is_dir() else split_path.stem
        print(f"  {tag}: {len(examples)} examples loaded")

    # Cap train_aug answerable rows if requested
    if max_aug is not None:
        rng = random.Random(seed)
        # We loaded train + valid first, then aug — trim the excess answerable from the end
        # (train_aug examples were appended last, so trim from the tail)
        n_non_aug = sum(
            len(load_ehrsql_split(p))
            for p in splits
            if "train_aug" not in str(p)
        )
        non_aug_ans = [r for r in answerable[:n_non_aug] if r]
        aug_ans = answerable[n_non_aug:]
        if len(aug_ans) > max_aug:
            aug_ans = rng.sample(aug_ans, max_aug)
            print(f"  train_aug capped to {max_aug} answerable examples")
        answerable = non_aug_ans + aug_ans

    # Oversample unanswerable
    rng = random.Random(seed)
    oversampled_unans = unanswerable * oversample
    rng.shuffle(oversampled_unans)

    combined = answerable + oversampled_unans
    rng.shuffle(combined)

    total = len(combined)
    n_unans = len(oversampled_unans)
    print(f"\nDataset stats:")
    print(f"  Answerable SQL:      {len(answerable):>7,}")
    print(f"  Unanswerable base:   {len(unanswerable):>7,}")
    print(f"  Unanswerable ×{oversample}:     {n_unans:>7,}")
    print(f"  Total:               {total:>7,}")
    print(f"  Unanswerable ratio:  {n_unans/total*100:.1f}%")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     default=str(DATA_DIR / "train"))
    parser.add_argument("--valid",     default=str(DATA_DIR / "valid"))
    parser.add_argument("--aug",       default=str(DATA_DIR / "train_aug"))
    parser.add_argument("--no-aug",    action="store_true", help="Skip train_aug")
    parser.add_argument("--max-aug",   type=int, default=None,
                        help="Max answerable rows from train_aug (default: all)")
    parser.add_argument("--oversample", type=int, default=3,
                        help="Unanswerable oversample factor (default: 3)")
    parser.add_argument("--output",    default=str(ROOT / "data" / "sft" / "omnisql_sft_train.jsonl"))
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    splits = [Path(args.train), Path(args.valid)]
    if not args.no_aug:
        splits.append(Path(args.aug))

    print("Loading splits:")
    for s in splits:
        if not s.exists():
            print(f"  ERROR: {s} does not exist")
            sys.exit(1)

    dataset = build_dataset(
        splits,
        system_prompt=config.SYSTEM_PROMPT,
        oversample=args.oversample,
        max_aug=args.max_aug,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {len(dataset):,} examples → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
