"""QLoRA SFT fine-tuning of Qwen2.5-Coder-7B-Instruct on EHRSQL.

Hardware: RTX 4080 Super (16 GB GDDR6X)
Key settings adapted for 16 GB from the project plan:
  - bs=1, effective batch=16 via gradient accumulation
  - max_seq_length=1536 (down from 2048)
  - Unsloth kernels for ~30% VRAM savings

Usage:
    python -m ehrcopilot.finetune.qlora_sft \
        --data data/ehrsql/sft_train.jsonl \
        --output checkpoints/sft
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import argparse
    import os

    # bitsandbytes needs the CUDA 13 nvJitLink library from the nvidia-cu13 package.
    # Add it to LD_LIBRARY_PATH so the paged_adamw_8bit optimizer can load.
    _nvidia_cu13 = os.path.expanduser(
        "~/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
    )
    _conda_cu13 = os.path.join(
        os.path.dirname(os.__file__), "site-packages", "nvidia", "cu13", "lib"
    )
    for _p in [_nvidia_cu13, _conda_cu13]:
        if os.path.isdir(_p) and _p not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = _p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SFT JSONL data path")
    parser.add_argument("--output", default="checkpoints/sft", help="Output dir")
    parser.add_argument("--base-model", default="unsloth/gemma-3-12b-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap optimizer steps (overrides epochs when > 0); use a small value to smoke-test.")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="per_device_train_batch_size (raise on a big GPU, e.g. 8-16 on a 95GB card)")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="gradient_accumulation_steps (lower it when raising --batch-size to keep effective batch)")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to checkpoint dir to resume from (or 'true' for latest)")
    args = parser.parse_args()

    # Lazy imports — only loaded when actually training
    try:
        import torch
        # Gemma 3 12B is a multimodal checkpoint; load via Unsloth FastModel
        # (FastLanguageModel is text-only). Aliased to keep call sites stable.
        from unsloth import FastModel as FastLanguageModel  # type: ignore[import]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import]
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        print("Install with: pip install -e '.[train]' unsloth")
        sys.exit(1)

    print(f"Loading base model: {args.base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Loading data from {args.data}")
    raw_data = []
    with open(args.data) as f:
        for line in f:
            raw_data.append(json.loads(line.strip()))

    def _format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data).map(_format_chat, remove_columns=["messages", "id", "is_answerable"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=25,
        save_steps=250,
        output_dir=str(output_dir),
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting SFT training...")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    adapter_path = output_dir / "adapter_final"
    print(f"Saving adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("SFT training complete.")


if __name__ == "__main__":
    main()
