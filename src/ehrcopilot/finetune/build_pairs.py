"""Build ORPO preference pairs for fine-tuning.

For UNANSWERABLE questions:
  chosen   = [ABSTAIN]
  rejected = model's own inference output (or random gold SQL as fallback)
  Teaches the model to prefer abstention over its own hallucinated SQL —
  stronger signal than random gold SQL since the model must unlearn its own wrong answer.

For ANSWERABLE questions (SQL quality):
  chosen   = gold SQL (already correct for EHRSQL 2024 MIMIC-IV schema)
  rejected = model's incorrect output (verified by execution when --verify-execution set)

  IMPORTANT: pairs are only generated when gold SQL executes with real data on the DB.
  Gold SQL that returns empty (patient not in 94-patient subset) is skipped — without a
  reference result we can't verify whether the model output is actually wrong.
  With --verify-execution, model output is also execution-checked so pred==gold (by
  result set) is skipped (no preference signal when model already gets it right).

Usage:
    python -m ehrcopilot.finetune.build_pairs \\
        --train data/ehrsql2024/mimic_iv/train \\
        --valid data/ehrsql2024/mimic_iv/valid \\
        --adapter checkpoints/orpo_v4_colab/adapter_final \\
        --output data/pairs/orpo_v5_pairs.jsonl \\
        --verify-execution \\
        --inference-rejected
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# bitsandbytes needs libnvJitLink.so.13 — must be set before any imports that pull it in
_conda_cu13 = os.path.join(
    os.path.dirname(os.__file__), "site-packages", "nvidia", "cu13", "lib"
)
if os.path.isdir(_conda_cu13) and _conda_cu13 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _conda_cu13 + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from ehrcopilot import config
from ehrcopilot.eval.harness import (
    load_ehrsql_split,
    _exec_safe,
    results_match,
)

ABSTAIN_TOKEN = "[ABSTAIN]"


def _build_messages(question: str) -> list[dict]:
    # Use the unified system prompt from config — identical to what eval harness uses.
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _normalize_sql(sql: str) -> str:
    """Lightweight SQL normalization for string comparison."""
    sql = sql.lower().strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql


def _sample_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    temperature: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    import torch

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build_pairs(
    train_path: Path,
    adapter_path: Path,
    output_path: Path,
    valid_path: Path | None = None,
    max_answerable: int = 500,
    unanswerable_only: bool = False,
    verify_execution: bool = False,
    inference_rejected: bool = False,
    num_samples: int = 1,
    seed: int = 42,
) -> dict[str, int]:
    """Build preference pairs for ORPO/DPO training.

    verify_execution=True: for answerable pairs, requires gold SQL to execute with
      actual data on our DB. Questions where gold errors or returns empty are skipped —
      broken gold SQL as "chosen" would teach the model to prefer SQL that cannot run.
      Model output is also execution-checked: if it already matches gold, skip (no signal).
    num_samples > 1: sample K model outputs and pick the first incorrect one as rejected.
    """
    random.seed(seed)

    train_examples = load_ehrsql_split(train_path)
    answerable_examples = [e for e in train_examples if e.is_answerable]

    # Pool of valid gold SQL used as fallback rejected for unanswerable pairs
    # (when model already abstains and we need a SQL to serve as the "wrong" response).
    gold_sql_pool = [e.gold_sql for e in answerable_examples if e.gold_sql and e.gold_sql.strip()]

    unanswerable_examples = []
    if valid_path and valid_path.exists():
        valid_examples = load_ehrsql_split(valid_path)
        unanswerable_examples = [e for e in valid_examples if not e.is_answerable]
        unanswerable_examples += [e for e in train_examples if not e.is_answerable]

    print(f"Gold SQL pool: {len(gold_sql_pool)} SQLs from answerable train examples")
    print(f"Unanswerable examples: {len(unanswerable_examples)}")
    print(f"Answerable examples (capped at {max_answerable}): {min(len(answerable_examples), max_answerable)}")
    print(f"Mode: {'unanswerable-only' if unanswerable_only else 'unanswerable + answerable'}")
    print(f"Verify execution: {verify_execution} (skip pairs where gold SQL errors/empty on our DB)")
    print(f"Unanswerable rejected: {'model inference (stronger signal)' if inference_rejected else 'random gold SQL (fast)'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {
        "total_processed": 0,
        "pairs_written": 0,
        "unanswerable_pairs": 0,
        "answerable_pairs": 0,
        "skipped_gold_error": 0,     # gold SQL failed on our DB
        "skipped_gold_empty": 0,     # gold SQL returned empty (patient not in demo)
        "skipped_no_diff": 0,        # model output identical to gold (string match)
        "skipped_model_abstained": 0,
        "skipped_exec_match": 0,     # model output execution-matches gold
    }

    # Pre-load model if needed for either inference-rejected unanswerable or answerable pairs
    model = None
    tokenizer = None
    needs_model = inference_rejected or not unanswerable_only
    if needs_model:
        try:
            import torch
            from unsloth import FastLanguageModel  # type: ignore[import]
            print(f"\nLoading adapter from {adapter_path} ...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_path),
                max_seq_length=config.MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            print("Model ready.")
        except ImportError as exc:
            print(f"Cannot load model: {exc}")
            if not unanswerable_only:
                print("Falling back to unanswerable-only mode with random rejected SQL.")
                unanswerable_only = True
                inference_rejected = False

    with open(output_path, "w") as f:

        # ── Unanswerable DPO pairs ────────────────────────────────────────────
        print(f"\nBuilding unanswerable DPO pairs ({len(unanswerable_examples)} examples)...")
        for i, ex in enumerate(unanswerable_examples):
            if i % 50 == 0:
                print(f"  Unanswerable {i}/{len(unanswerable_examples)} ...", flush=True)
            stats["total_processed"] += 1
            messages = _build_messages(ex.question)

            if inference_rejected and model is not None:
                # Use model's actual output as rejected — stronger training signal.
                # If the model already abstains, fall back to random SQL (pair still useful:
                # it reinforces [ABSTAIN] but counts as a trivially-won comparison).
                rejected_sql = _sample_one(model, tokenizer, messages, temperature=0.8)
                if rejected_sql == ABSTAIN_TOKEN or not rejected_sql.strip():
                    rejected_sql = random.choice(gold_sql_pool)
            else:
                # Fast fallback: random gold SQL from answerable set.
                rejected_sql = random.choice(gold_sql_pool)

            pair = {
                "id": ex.id,
                "prompt": messages,
                "chosen": [{"role": "assistant", "content": ABSTAIN_TOKEN}],
                "rejected": [{"role": "assistant", "content": rejected_sql}],
                "is_answerable": False,
            }
            f.write(json.dumps(pair) + "\n")
            stats["pairs_written"] += 1
            stats["unanswerable_pairs"] += 1

        print(f"  Unanswerable pairs written: {stats['unanswerable_pairs']}")

        if unanswerable_only:
            print("Skipping answerable pairs (--unanswerable-only)")
        else:
            # ── Answerable DPO pairs (verify gold SQL first, then model inference) ──
            if model is None:
                print("Model not loaded — skipping answerable pairs.")
            else:
                ans_subset = answerable_examples[:max_answerable]
                print(f"\nBuilding answerable DPO pairs ({len(ans_subset)} examples)...")

                for i, ex in enumerate(ans_subset):
                    stats["total_processed"] += 1
                    if i % 100 == 0:
                        print(
                            f"  Answerable {i}/{len(ans_subset)} — "
                            f"pairs: {stats['answerable_pairs']} "
                            f"gold_err: {stats['skipped_gold_error']} "
                            f"gold_empty: {stats['skipped_gold_empty']}",
                            flush=True,
                        )

                    canon_gold = ex.gold_sql or ""
                    gold_rows: list | None = None
                    gold_err: str | None = None

                    if verify_execution:
                        # Validate gold SQL before anything else. Skip if empty (patient not
                        # in DB subset) — without a reference result we can't determine whether
                        # the model's output is actually wrong, so there's no preference signal.
                        gold_rows, gold_err = _exec_safe(canon_gold)
                        if gold_err is not None:
                            stats["skipped_gold_error"] += 1
                            continue
                        if not gold_rows:
                            stats["skipped_gold_empty"] += 1
                            continue

                    messages = _build_messages(ex.question)

                    # Sample num_samples outputs; pick the first usable rejected candidate
                    model_out = None
                    for _s in range(max(1, num_samples)):
                        sample = _sample_one(model, tokenizer, messages, temperature=1.0)

                        if sample == ABSTAIN_TOKEN or not sample.strip():
                            stats["skipped_model_abstained"] += 1
                            continue

                        if verify_execution:
                            pred_rows, pred_err = _exec_safe(sample)
                            if pred_err is None and results_match(
                                pred_rows, gold_rows, gold_err=gold_err
                            ):
                                # Model already gets this right — no preference signal
                                stats["skipped_exec_match"] += 1
                                continue
                        else:
                            if _normalize_sql(sample) == _normalize_sql(canon_gold):
                                stats["skipped_no_diff"] += 1
                                continue

                        model_out = sample
                        break

                    if model_out is None:
                        continue

                    pair = {
                        "id": ex.id,
                        "prompt": messages,
                        "chosen": [{"role": "assistant", "content": canon_gold}],
                        "rejected": [{"role": "assistant", "content": model_out}],
                        "is_answerable": True,
                    }
                    f.write(json.dumps(pair) + "\n")
                    stats["pairs_written"] += 1
                    stats["answerable_pairs"] += 1

                print(f"  Answerable pairs written: {stats['answerable_pairs']}")

    print(f"\nPreference pairs written to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["unanswerable_pairs"] == 0:
        print("\nERROR: 0 unanswerable pairs generated — check valid_path and gold_sql_pool.")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="EHRSQL train.json")
    parser.add_argument("--valid", default=None, help="EHRSQL valid.json (unanswerable source)")
    parser.add_argument("--adapter", required=True, help="SFT adapter path")
    parser.add_argument("--output", required=True, help="Output DPO pairs JSONL")
    parser.add_argument("--max-answerable", type=int, default=500)
    parser.add_argument("--unanswerable-only", action="store_true",
                        help="Skip answerable pairs, focus on abstention DPO only")
    parser.add_argument("--inference-rejected", action="store_true",
                        help="For unanswerable pairs: run inference to get model's actual output "
                             "as the rejected response (stronger signal than random gold SQL). "
                             "Slower but produces cleaner abstention pairs.")
    parser.add_argument("--verify-execution", action="store_true",
                        help="For answerable pairs: skip if model output executes correctly "
                             "(execution-verified matching, cleaner signal than string-diff)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Sample K model outputs per answerable question and use first incorrect one")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pairs(
        train_path=Path(args.train),
        adapter_path=Path(args.adapter),
        output_path=Path(args.output),
        valid_path=Path(args.valid) if args.valid else None,
        max_answerable=args.max_answerable,
        unanswerable_only=args.unanswerable_only,
        inference_rejected=args.inference_rejected,
        verify_execution=args.verify_execution,
        num_samples=args.num_samples,
        seed=args.seed,
    )
