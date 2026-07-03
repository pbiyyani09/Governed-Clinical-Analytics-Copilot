"""Build ORPO preference pairs for fine-tuning OmniSQL-7B on EHRSQL 2024 MIMIC-IV.

Two sources of pairs:
  1. Existing predictions file (e.g. outputs/omnisql_7b_preds.jsonl from baseline eval)
     - Already have 492 pairs: 359 wrong-SQL + 133 hallucinations
  2. Run inference with SFT-OmniSQL on train split to get more pairs (~2,000-3,000)

Output format (same as existing orpo_v5 pairs, compatible with abstention_dpo.py):
  {
    "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    "chosen": [{"role": "assistant", "content": gold_sql_or_ABSTAIN}],
    "rejected": [{"role": "assistant", "content": model_wrong_prediction}],
    "is_answerable": true/false
  }

Usage:
  # Build from existing predictions only (no new inference):
  python scripts/build_omnisql_orpo_pairs.py \\
      --preds outputs/omnisql_7b_preds.jsonl \\
      --output data/pairs/omnisql_orpo_pairs.jsonl

  # Also run inference on train set with SFT adapter for more pairs:
  python scripts/build_omnisql_orpo_pairs.py \\
      --preds outputs/omnisql_7b_preds.jsonl \\
      --adapter checkpoints/omnisql_sft/adapter_final \\
      --train data/ehrsql2024/mimic_iv/train \\
      --output data/pairs/omnisql_orpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehrcopilot import config
from ehrcopilot.eval.harness import (
    ABSTAIN_TOKEN,
    load_ehrsql_split,
    post_process_sql,
    _exec_safe,
    results_match,
)

ABSTAIN = "[ABSTAIN]"


def load_predictions(preds_path: Path) -> dict[str, dict]:
    """Load prediction JSONL → dict keyed by question id."""
    preds = {}
    with open(preds_path) as f:
        for line in f:
            row = json.loads(line)
            preds[row["id"]] = row
    return preds


def make_prompt_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


def build_pairs_from_preds(
    examples,
    predictions: dict[str, dict],
    verify_execution: bool = True,
) -> list[dict]:
    """Build ORPO pairs from a predictions dict keyed by example id."""
    pairs = []
    skipped_correct = 0
    skipped_exec = 0

    for ex in examples:
        pred = predictions.get(ex.id)
        if pred is None:
            continue

        predicted_sql = pred.get("predicted_sql", "").strip()
        outcome = pred.get("outcome", "unknown")

        if not ex.is_answerable:
            # Unanswerable: only pair when model hallucinated SQL (not abstained)
            if outcome in ("hallucination",) or (
                predicted_sql and predicted_sql != ABSTAIN
            ):
                pairs.append({
                    "prompt":        make_prompt_messages(ex.question),
                    "chosen":        [{"role": "assistant", "content": ABSTAIN}],
                    "rejected":      [{"role": "assistant", "content": predicted_sql}],
                    "is_answerable": False,
                })
        else:
            # Answerable: only pair when model was wrong
            if outcome == "correct":
                skipped_correct += 1
                continue

            if not ex.gold_sql or predicted_sql == ABSTAIN or not predicted_sql:
                continue

            gold_sql = post_process_sql(ex.gold_sql)
            gold_rows, gold_err = _exec_safe(gold_sql)

            if gold_err or not gold_rows:
                skipped_exec += 1
                continue

            if verify_execution:
                pred_rows, pred_err = _exec_safe(post_process_sql(predicted_sql))
                if pred_err is None and results_match(pred_rows, gold_rows):
                    skipped_correct += 1
                    continue

            pairs.append({
                "prompt":        make_prompt_messages(ex.question),
                "chosen":        [{"role": "assistant", "content": ex.gold_sql}],
                "rejected":      [{"role": "assistant", "content": predicted_sql}],
                "is_answerable": True,
            })

    print(f"  Skipped (already correct): {skipped_correct}")
    print(f"  Skipped (gold empty/error): {skipped_exec}")
    return pairs


def run_inference_on_split(split_path: Path, adapter_path: Path) -> dict[str, dict]:
    """Run SFT-OmniSQL adapter inference on a split and return predictions dict."""
    import subprocess
    import tempfile

    preds_file = Path(tempfile.mktemp(suffix=".jsonl"))
    cmd = [
        sys.executable, "-m", "ehrcopilot.eval.harness",
        str(split_path),
        "--model", str(adapter_path),
        "--save-predictions", str(preds_file),
        "--output", str(preds_file.with_suffix(".json")),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    print(f"  Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print("  WARNING: inference returned non-zero exit code")

    if not preds_file.exists():
        print(f"  ERROR: predictions file not found at {preds_file}")
        return {}

    return load_predictions(preds_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds",    default=None,
                        help="Existing predictions JSONL (e.g. outputs/omnisql_7b_preds.jsonl)")
    parser.add_argument("--train",    default=str(ROOT / "data/ehrsql2024/mimic_iv/train"),
                        help="Train split dir for additional inference")
    parser.add_argument("--adapter",  default=None,
                        help="SFT adapter path; if set, run inference on --train to get more pairs")
    parser.add_argument("--output",   default=str(ROOT / "data/pairs/omnisql_orpo_pairs.jsonl"))
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip execution verification (faster, noisier pairs)")
    args = parser.parse_args()

    all_pairs: list[dict] = []

    # --- Source 1: existing predictions file ---
    if args.preds:
        preds_path = Path(args.preds)
        if not preds_path.exists():
            print(f"ERROR: --preds file not found: {preds_path}")
            sys.exit(1)

        print(f"\nLoading predictions from {preds_path}")
        preds = load_predictions(preds_path)

        # Need to match predictions to examples from the split they came from
        # Infer split from the 'id' prefix or try both test and valid
        test_path = ROOT / "data/ehrsql2024/mimic_iv/test"
        valid_path = ROOT / "data/ehrsql2024/mimic_iv/valid"

        for split_path in [test_path, valid_path]:
            if not split_path.exists():
                continue
            examples = load_ehrsql_split(split_path)
            split_ids = {ex.id for ex in examples}
            matching = [ex for ex in examples if ex.id in preds]
            if matching:
                print(f"  Matched {len(matching)} predictions to {split_path.name}")
                pairs = build_pairs_from_preds(
                    matching, preds, verify_execution=not args.no_verify
                )
                print(f"  → {len(pairs)} pairs from {split_path.name}")
                all_pairs.extend(pairs)

    # --- Source 2: inference on train split with SFT adapter ---
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            print(f"ERROR: --adapter not found: {adapter_path}")
            sys.exit(1)

        train_path = Path(args.train)
        print(f"\nRunning SFT-OmniSQL inference on {train_path.name} ...")
        train_preds = run_inference_on_split(train_path, adapter_path)

        if train_preds:
            train_examples = load_ehrsql_split(train_path)
            print(f"  Building pairs from {len(train_examples)} train examples ...")
            pairs = build_pairs_from_preds(
                train_examples, train_preds, verify_execution=not args.no_verify
            )
            print(f"  → {len(pairs)} pairs from train")
            all_pairs.extend(pairs)

    if not all_pairs:
        print("\nNo pairs generated. Provide --preds and/or --adapter.")
        sys.exit(1)

    # Deduplicate by (question, chosen, rejected)
    seen: set[tuple] = set()
    unique_pairs = []
    for p in all_pairs:
        key = (
            p["prompt"][-1]["content"],
            p["chosen"][0]["content"],
            p["rejected"][0]["content"],
        )
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    n_abstention = sum(1 for p in unique_pairs if not p["is_answerable"])
    n_sql        = sum(1 for p in unique_pairs if p["is_answerable"])

    print(f"\nTotal ORPO pairs: {len(unique_pairs):,}")
    print(f"  SQL quality (answerable wrong):   {n_sql:,}")
    print(f"  Abstention (unanswerable halluc): {n_abstention:,}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {len(unique_pairs):,} pairs → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
