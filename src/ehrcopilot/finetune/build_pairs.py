"""Build DPO preference pairs for Abstention-DPO training.

Novel contribution: [ABSTAIN] as the DPO chosen response for unanswerable questions.
No published EHRSQL system uses this approach.

Two strategies, each addressing a different coverage problem:

For UNANSWERABLE questions (primary abstention contribution):
  chosen   = [ABSTAIN]
  rejected = randomly sampled gold SQL from an answerable question
  Rationale: the SFT model already abstains perfectly (temp 1.3 still gives [ABSTAIN]),
  so we cannot use the model to generate a SQL "rejected" sample. Instead we use a
  plausible-looking SQL from the answerable set — the DPO loss still correctly teaches
  the model to prefer [ABSTAIN] over any SQL for unanswerable questions.

For ANSWERABLE questions (SQL quality):
  chosen   = gold SQL
  rejected = model's single greedy output if string-different from gold
  Avoids execution-based matching (MIMIC-III gold SQL ≠ MIMIC-IV-Demo schema).
  String-normalized comparison catches most cases where model diverges from gold.

Usage:
    python -m ehrcopilot.finetune.build_pairs \\
        --train data/ehrsql/ehrsql/mimic_iii/train.json \\
        --valid data/ehrsql/ehrsql/mimic_iii/valid.json \\
        --adapter checkpoints/sft/adapter_final \\
        --output data/ehrsql/dpo_pairs.jsonl \\
        --max-answerable 500 \\
        --unanswerable-only      # (flag: skip answerable pairs, focus on abstention)
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
    _canonicalize_gold_sql,
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

    verify_execution=True: for answerable pairs, skip if model output executes
      correctly (no preference signal needed). Creates much cleaner pairs than
      string-diff matching.
    num_samples > 1: sample K model outputs and pick the first incorrect one
      as rejected (more diverse rejected candidates).
    """
    random.seed(seed)

    train_examples = load_ehrsql_split(train_path)
    answerable_examples = [e for e in train_examples if e.is_answerable]
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
    print(f"Unanswerable rejected: {'model inference (stronger signal)' if inference_rejected else 'random gold SQL (fast)'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {
        "total_processed": 0,
        "pairs_written": 0,
        "unanswerable_pairs": 0,
        "answerable_pairs": 0,
        "skipped_no_diff": 0,
        "skipped_model_abstained": 0,
        "skipped_exec_match": 0,
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
            # ── Answerable DPO pairs (1 model inference per question) ─────────
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
                            f"ans_pairs: {stats['answerable_pairs']}",
                            flush=True,
                        )

                    messages = _build_messages(ex.question)

                    # Sample num_samples outputs; pick the first usable rejected candidate
                    model_out = None
                    gold_rows = None
                    for _s in range(max(1, num_samples)):
                        sample = _sample_one(model, tokenizer, messages, temperature=1.0)

                        if sample == ABSTAIN_TOKEN or not sample.strip():
                            continue

                        if verify_execution:
                            if gold_rows is None:
                                gold_rows, _ = _exec_safe(_canonicalize_gold_sql(ex.gold_sql or ""))
                            pred_rows, pred_err = _exec_safe(sample)
                            if pred_err is None and results_match(pred_rows, gold_rows):
                                # Model already gets this right — no signal
                                continue
                        else:
                            if _normalize_sql(sample) == _normalize_sql(ex.gold_sql or ""):
                                continue

                        model_out = sample
                        break

                    if model_out is None:
                        # All samples were abstentions or correct outputs
                        if _sample_one(model, tokenizer, messages, temperature=1.0) == ABSTAIN_TOKEN:
                            stats["skipped_model_abstained"] += 1
                        elif verify_execution:
                            stats["skipped_exec_match"] += 1
                        else:
                            stats["skipped_no_diff"] += 1
                        continue

                    pair = {
                        "id": ex.id,
                        "prompt": messages,
                        "chosen": [{"role": "assistant", "content": ex.gold_sql}],
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
