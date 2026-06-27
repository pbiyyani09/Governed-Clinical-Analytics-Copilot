"""Abstention preference fine-tuning on top of the SFT checkpoint using ORPO.

Novel contribution: [ABSTAIN] as the preferred response for unanswerable questions.
Uses ORPO (Odds-Ratio Preference Optimization) instead of DPO to eliminate the
reference model distribution mismatch that caused DPO to over-abstain.

ORPO advantages over DPO here:
  - No reference model needed (no second model copy, same VRAM as SFT)
  - Simultaneously applies SFT loss on chosen responses (prevents SQL quality regression)
  - Reference signal is implicit via the policy's own odds ratio (avoids base↔SFT mismatch)

Usage:
    python -m ehrcopilot.finetune.abstention_dpo \\
        --pairs data/ehrsql/dpo_pairs.jsonl \\
        --adapter checkpoints/sft/adapter_final \\
        --output checkpoints/dpo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import argparse
    import os

    # bitsandbytes needs libnvJitLink.so.13 from the nvidia-cu13 package
    _conda_cu13 = os.path.join(
        os.path.dirname(os.__file__), "site-packages", "nvidia", "cu13", "lib"
    )
    if os.path.isdir(_conda_cu13) and _conda_cu13 not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _conda_cu13 + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="DPO/ORPO pairs JSONL")
    parser.add_argument("--adapter", required=True, help="SFT adapter path")
    parser.add_argument("--output", default="checkpoints/dpo", help="Output dir")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max tokens for prompt+chosen/rejected (default 1024 — fits EHRSQL "
                             "schema-prompt+SQL and avoids Gemma-3 ORPO OOM on 24GB; raise to 1536 if needed)")
    parser.add_argument("--orpo-lambda", type=float, default=0.1,
                        help="ORPO lambda: weight of odds-ratio loss vs SFT loss")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Checkpoint dir to resume from (or 'true' for latest)")
    args = parser.parse_args()

    try:
        import transformers.utils.hub as _hub_shim
        if not hasattr(_hub_shim, "TRANSFORMERS_CACHE"):
            _hub_shim.TRANSFORMERS_CACHE = os.path.join(
                os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                "transformers",
            )
        import torch
        # Gemma 3 is multimodal — load via Unsloth FastModel (aliased).
        from unsloth import FastModel as FastLanguageModel  # type: ignore[import]
        from trl import ORPOConfig, ORPOTrainer  # type: ignore[import]
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        sys.exit(1)

    # Reduce CUDA fragmentation — ORPO's concatenated chosen+rejected forward OOMs
    # at step 2 from fragmentation otherwise on a 24GB card.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"Loading SFT adapter from {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1536,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Gemma 3 requires token_type_ids during training; TRL's ORPO collator omits
    # it (and passes input_ids positionally). Default it to zeros (all-text).
    from ehrcopilot.finetune._gemma_compat import patch_token_type_ids
    patch_token_type_ids(model)

    print(f"Loading preference pairs from {args.pairs}")
    raw_pairs = []
    with open(args.pairs) as f:
        for line in f:
            raw_pairs.append(json.loads(line.strip()))

    print(f"  {len(raw_pairs)} pairs loaded")
    n_unans = sum(1 for p in raw_pairs if not p.get("is_answerable", True))
    n_ans = sum(1 for p in raw_pairs if p.get("is_answerable", True))
    print(f"  Unanswerable (abstention): {n_unans}  |  Answerable (SQL quality): {n_ans}")

    dataset = Dataset.from_list(raw_pairs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    orpo_config = ORPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=args.max_length,
        max_prompt_length=min(896, args.max_length - 200),
        beta=args.orpo_lambda,   # TRL's ORPOConfig uses 'beta' as the lambda parameter name
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        output_dir=str(output_dir),
        report_to="none",
        logging_steps=10,
        save_steps=32,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Abstention-ORPO training...")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    adapter_path = output_dir / "adapter_final"
    print(f"Saving ORPO adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("ORPO training complete.")


if __name__ == "__main__":
    main()
