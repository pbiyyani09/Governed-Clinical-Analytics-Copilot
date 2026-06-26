"""GRPO fine-tuning with execution-based reward on MIMIC-IV-Demo.

Reward signal: actual SQL execution against data/mimic_iv_demo.db.
  +1.0  correct answer (SQL result matches gold on answerable)
  -0.2  SQL executes but result is wrong (valid syntax, wrong answer)
  -0.5  SQL fails to execute (syntax/schema error)
  -1.0  wrong abstention (abstained on answerable; opportunity cost)
  +1.0  correct abstention (abstained on unanswerable)
 -10.0  hallucinated SQL on unanswerable (mirrors RS(10) penalty)

The 3-tier reward for answerable questions breaks the reward_std=0 deadlock:
even when all K rollouts are wrong, execution-error vs wrong-result creates
variance within the group, keeping the gradient non-zero.

Usage:
    python -m ehrcopilot.finetune.grpo_train \\
        --data data/ehrsql/sft_train_v2.jsonl \\
        --adapter checkpoints/orpo_v3/adapter_final \\
        --output checkpoints/grpo_v2 \\
        --temperature 1.2
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# bitsandbytes / TRANSFORMERS_CACHE shim — must run before any TRL/transformers import
_conda_cu13 = os.path.join(
    os.path.dirname(os.__file__), "site-packages", "nvidia", "cu13", "lib"
)
if os.path.isdir(_conda_cu13) and _conda_cu13 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _conda_cu13 + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import transformers.utils.hub as _hub_shim
if not hasattr(_hub_shim, "TRANSFORMERS_CACHE"):
    _hub_shim.TRANSFORMERS_CACHE = os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        "transformers",
    )

from ehrcopilot import config
from ehrcopilot.eval.harness import (
    _canonicalize_gold_sql,
    _exec_safe,
    results_match,
    ABSTAIN_TOKEN,
)

SYSTEM_PROMPT = (
    "You are a clinical analytics assistant. Convert the user's question into "
    "a valid SQLite SELECT query over the MIMIC-IV-Demo database. "
    f"If the question cannot be answered with the available data, output exactly: {ABSTAIN_TOKEN}\n\n"
    + config.schema_to_prompt()
)


def _build_reward_fn():
    """Return a GRPO reward function that executes SQL against MIMIC-IV-Demo."""

    def reward_fn(
        prompts: list,
        completions: list,
        is_answerable: list[bool],
        gold_sql: list[str],
        **kwargs,
    ) -> list[float]:
        import statistics

        rewards = []
        for completion, is_ans, gold in zip(completions, is_answerable, gold_sql):
            # TRL passes completions as list[list[dict]] (one message per rollout)
            if isinstance(completion, list):
                pred = completion[0]["content"].strip() if completion else ""
            else:
                pred = str(completion).strip()

            abstained = pred == ABSTAIN_TOKEN or not pred

            if is_ans:
                if abstained:
                    rewards.append(-1.0)
                else:
                    pred_rows, pred_err = _exec_safe(pred)
                    gold_rows, _ = _exec_safe(_canonicalize_gold_sql(gold))
                    if pred_err is None and results_match(pred_rows, gold_rows):
                        rewards.append(1.0)   # correct result
                    elif pred_err is not None:
                        rewards.append(-0.5)  # SQL failed to execute (syntax/schema error)
                    else:
                        rewards.append(-0.2)  # executed but wrong result
            else:
                if abstained:
                    rewards.append(1.0)
                else:
                    rewards.append(-10.0)

        if len(rewards) > 1:
            std = statistics.stdev(rewards)
            if std < 1e-6:
                print("[WARN] reward_std=0 for this batch — no gradient update", flush=True)

        return rewards

    return reward_fn


def _load_grpo_dataset(data_path: Path) -> "Dataset":
    from datasets import Dataset  # type: ignore[import]

    raw: list[dict] = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            # sft_train_v2.jsonl has: messages, is_answerable, id
            # Extract gold_sql from the last assistant message
            messages = ex["messages"]
            gold_sql = ""
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    gold_sql = msg["content"]
                    break

            # GRPO prompt = all messages except the final assistant turn
            prompt_msgs = [m for m in messages if not (m["role"] == "assistant")]

            raw.append({
                "prompt": prompt_msgs,
                "is_answerable": ex.get("is_answerable", True),
                "gold_sql": gold_sql,
            })

    # Skip unanswerable examples whose gold_sql is [ABSTAIN] — reward fn handles them,
    # but we want the gold_sql column populated for answerable pairs too.
    print(f"  {len(raw)} GRPO examples loaded")
    ans = sum(1 for r in raw if r["is_answerable"])
    unans = sum(1 for r in raw if not r["is_answerable"])
    print(f"  Answerable: {ans} | Unanswerable: {unans}")
    return Dataset.from_list(raw)


def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SFT v2 JSONL training data")
    parser.add_argument("--adapter", required=True, help="SFT adapter to start from")
    parser.add_argument("--output", default="checkpoints/grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="K rollouts per question (reduce to 2 if OOM)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Subsample dataset to this many examples (default: use all)")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL penalty weight (keep low for exploration)")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Sampling temperature for rollout diversity (default 1.2)")
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    try:
        # Gemma 3 is multimodal — load via Unsloth FastModel (aliased).
        from unsloth import FastModel as FastLanguageModel  # type: ignore[import]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        sys.exit(1)

    print(f"Loading SFT adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Gemma 3 requires token_type_ids during training; default it to zeros.
    from ehrcopilot.finetune._gemma_compat import patch_token_type_ids
    patch_token_type_ids(model)

    print(f"Loading GRPO dataset: {args.data}")
    dataset = _load_grpo_dataset(Path(args.data))
    if args.max_examples and args.max_examples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.max_examples))
        print(f"  Subsampled to {len(dataset)} examples")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Total gradient steps = len(dataset) * epochs / (batch_size * grad_accum * num_generations)
    # With 9728 examples, K=4, bs=1, grad_accum=8: ~9728 / (1*8*4) = 304 steps
    grpo_config = GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=1280,
        # Rollout diversity: temperature > 0 is required for GRPO to work.
        # At temperature=0 (greedy default), all K rollouts are identical,
        # reward_std=0 within each group, advantages=0, grad_norm=0 → no learning.
        # 1.2 is higher than the previous 0.8 to force more diverse rollouts.
        temperature=args.temperature,
        top_p=0.95,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        beta=args.beta,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=True,
        output_dir=str(output_dir),
        report_to="none",
        logging_steps=10,
        save_steps=25,
    )

    reward_fn = _build_reward_fn()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training (K={args.num_generations} rollouts)...")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    adapter_path = output_dir / "adapter_final"
    print(f"Saving GRPO adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("GRPO training complete.")


if __name__ == "__main__":
    main()
