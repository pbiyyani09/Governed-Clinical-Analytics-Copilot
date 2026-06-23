"""Build DPO preference pairs for abstention training.

Novel contribution: DPO with [ABSTAIN] as the chosen response for unanswerable questions.
No published EHRSQL system uses this approach.

For answerable questions:
  chosen   = SQL that executes correctly and matches gold result
  rejected = SQL that fails to execute, returns wrong results, or hallucinates schema

For unanswerable questions:
  chosen   = [ABSTAIN]
  rejected = any SQL that "answers" the unanswerable question (fabricated queries)

Usage:
    python -m ehrcopilot.finetune.build_pairs \
        --train data/ehrsql/train.json \
        --adapter checkpoints/sft/adapter_final \
        --output data/ehrsql/dpo_pairs.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ehrcopilot import config
from ehrcopilot.agents.nodes.schema_linker import link_schema
from ehrcopilot.db.connection import execute_query
from ehrcopilot.eval.harness import EHRSQLExample, load_ehrsql_split, _exec_safe, results_match

ABSTAIN_TOKEN = "[ABSTAIN]"


def _build_messages(question: str, linked_schema: dict) -> list[dict]:
    """Build chat messages list (system + user). No assistant turn — used as DPO prompt."""
    schema_text = config.schema_to_prompt(linked_schema)
    return [
        {
            "role": "system",
            "content": (
                "You are a clinical analytics SQL expert. Convert the user's question "
                "into a valid SQLite SELECT query.\n"
                f"If the question cannot be answered, output exactly: {ABSTAIN_TOKEN}\n\n"
                f"{schema_text}"
            ),
        },
        {"role": "user", "content": question},
    ]


def _sample_candidates(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    n: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list[str]:
    """Sample n candidate SQLs from the model at temperature > 0."""
    import torch

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    candidates = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(decoded.strip())
    return candidates


def build_pairs(
    train_path: Path,
    adapter_path: Path,
    output_path: Path,
    n_candidates: int = 8,
) -> dict[str, int]:
    try:
        import torch
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        sys.exit(1)

    print(f"Loading SFT model from {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    examples = load_ehrsql_split(train_path)
    print(f"Loaded {len(examples)} examples")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "pairs_written": 0, "answerable_pairs": 0, "unanswerable_pairs": 0}

    with open(output_path, "w") as f:
        for ex in examples:
            stats["total"] += 1
            linked = link_schema(ex.question, top_k=5)
            messages = _build_messages(ex.question, linked)

            candidates = _sample_candidates(model, tokenizer, messages, n=n_candidates)

            if ex.is_answerable:
                gold_rows, _ = _exec_safe(ex.gold_sql)

                chosen_list: list[str] = []
                rejected_list: list[str] = []

                for cand in candidates:
                    if cand == ABSTAIN_TOKEN:
                        rejected_list.append(cand)
                        continue
                    pred_rows, pred_err = _exec_safe(cand)
                    if pred_err is None and results_match(pred_rows, gold_rows):
                        chosen_list.append(cand)
                    else:
                        rejected_list.append(cand)

                if not chosen_list or not rejected_list:
                    continue

                # Conversational DPO format: prompt=list[msg], chosen/rejected=single assistant msg
                pair = {
                    "id": ex.id,
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": chosen_list[0]}],
                    "rejected": [{"role": "assistant", "content": rejected_list[0]}],
                    "is_answerable": True,
                }
                f.write(json.dumps(pair) + "\n")
                stats["pairs_written"] += 1
                stats["answerable_pairs"] += 1

            else:
                # Unanswerable: chosen = [ABSTAIN], rejected = any SQL the model generated
                sql_candidates = [c for c in candidates if c != ABSTAIN_TOKEN and c.strip()]
                if not sql_candidates:
                    continue

                pair = {
                    "id": ex.id,
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": ABSTAIN_TOKEN}],
                    "rejected": [{"role": "assistant", "content": sql_candidates[0]}],
                    "is_answerable": False,
                }
                f.write(json.dumps(pair) + "\n")
                stats["pairs_written"] += 1
                stats["unanswerable_pairs"] += 1

    print(f"Preference pairs written to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-candidates", type=int, default=8)
    args = parser.parse_args()

    build_pairs(Path(args.train), Path(args.adapter), Path(args.output), args.n_candidates)
