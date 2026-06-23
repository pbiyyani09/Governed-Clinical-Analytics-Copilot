"""Prepare SFT training data from EHRSQL dataset.

Formats each example as a chat template:
  system  = schema context (linked schema for this question)
  user    = NL question
  assistant = gold SQL  OR  [ABSTAIN] for unanswerable examples

Usage:
    python -m ehrcopilot.finetune.prepare_sft \
        --train data/ehrsql/train.json \
        --output data/ehrsql/sft_train.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.agents.nodes.schema_linker import link_schema
from ehrcopilot.eval.harness import EHRSQLExample, load_ehrsql_split

ABSTAIN_TOKEN = "[ABSTAIN]"

_SYSTEM_TEMPLATE = """\
You are a clinical analytics SQL expert. Convert the user's question into a valid SQLite SELECT query.
If the question cannot be answered with the available data, output exactly: [ABSTAIN]

{schema}"""


def format_example(ex: EHRSQLExample, max_tokens: int = config.MAX_SEQ_LENGTH) -> dict | None:
    """Format a single EHRSQL example as a chat-style training dict.

    Returns None if the formatted example exceeds max_tokens (rough char estimate).
    """
    assistant_content = ex.gold_sql if ex.is_answerable else ABSTAIN_TOKEN

    # Link schema to keep prompts short
    linked = link_schema(ex.question, top_k=5)
    schema_text = config.schema_to_prompt(linked)

    system_content = _SYSTEM_TEMPLATE.format(schema=schema_text)

    # Rough token estimate: 1 token ≈ 4 chars
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


def prepare(train_path: Path, output_path: Path) -> dict[str, int]:
    examples = load_ehrsql_split(train_path)
    print(f"Loaded {len(examples)} examples from {train_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "kept": 0, "truncated": 0, "answerable": 0, "unanswerable": 0}

    with open(output_path, "w") as f:
        for ex in examples:
            stats["total"] += 1
            formatted = format_example(ex)
            if formatted is None:
                stats["truncated"] += 1
                continue

            f.write(json.dumps(formatted) + "\n")
            stats["kept"] += 1
            if ex.is_answerable:
                stats["answerable"] += 1
            else:
                stats["unanswerable"] += 1

    print(f"SFT data written to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="EHRSQL train split JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    prepare(Path(args.train), Path(args.output))
