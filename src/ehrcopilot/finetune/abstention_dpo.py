"""Abstention-DPO fine-tuning on top of the SFT checkpoint.

Novel contribution: [ABSTAIN] as the DPO chosen response for unanswerable questions.
Uses the adapter-disable trick (ref_model=None) to avoid loading a second model copy,
making DPO fit within 16 GB VRAM on the RTX 4080 Super.

Usage:
    python -m ehrcopilot.finetune.abstention_dpo \
        --pairs data/ehrsql/dpo_pairs.jsonl \
        --adapter checkpoints/sft/adapter_final \
        --output checkpoints/dpo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="DPO pairs JSONL")
    parser.add_argument("--adapter", required=True, help="SFT adapter path")
    parser.add_argument("--output", default="checkpoints/dpo", help="Output dir")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    try:
        import os
        import transformers.utils.hub as _hub_shim
        if not hasattr(_hub_shim, "TRANSFORMERS_CACHE"):
            _hub_shim.TRANSFORMERS_CACHE = os.path.join(
                os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                "transformers",
            )
        import torch
        from unsloth import FastLanguageModel  # type: ignore[import]
        from trl import DPOConfig, DPOTrainer  # type: ignore[import]
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        sys.exit(1)

    print(f"Loading SFT adapter from {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1280,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    print(f"Loading DPO pairs from {args.pairs}")
    raw_pairs = []
    with open(args.pairs) as f:
        for line in f:
            raw_pairs.append(json.loads(line.strip()))

    dataset = Dataset.from_list(raw_pairs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=1280,
        max_prompt_length=768,
        beta=args.beta,
        loss_type="sigmoid",
        learning_rate=5e-6,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=True,
        output_dir=str(output_dir),
        report_to="none",
        logging_steps=25,
        save_steps=250,
    )

    # ref_model=None uses the adapter-disable trick:
    # TRL disables the LoRA adapters to compute the reference distribution
    # from the same model — no second model copy needed (saves ~8 GB VRAM).
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Abstention-DPO training...")
    trainer.train()

    adapter_path = output_dir / "adapter_final"
    print(f"Saving DPO adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("DPO training complete.")


if __name__ == "__main__":
    main()
