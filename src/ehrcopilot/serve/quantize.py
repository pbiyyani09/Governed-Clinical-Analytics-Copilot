"""Merge DPO adapter and quantize for serving.

Produces two outputs:
  1. models/merged/        — full bf16 HF model (for vLLM with bitsandbytes INT4 load)
  2. models/gguf/q4_k_m   — GGUF Q4_K_M (for llama.cpp / ollama as fallback)

Usage:
    python -m ehrcopilot.serve.quantize \
        --adapter checkpoints/dpo/adapter_final \
        --output models
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="DPO adapter path (Unsloth format)")
    parser.add_argument("--output", default="models", help="Output base directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--gguf", action="store_true", help="Also export GGUF Q4_K_M")
    args = parser.parse_args()

    try:
        import torch
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        print(f"Training dependencies not installed: {exc}")
        sys.exit(1)

    out = Path(args.output)

    print(f"Loading DPO adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1536,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # 1. Merge LoRA weights into the base model and save as bf16 HF checkpoint
    merged_path = out / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"Merging adapter and saving merged model → {merged_path}")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved.")

    # 2. Optionally export GGUF Q4_K_M for ollama/llama.cpp
    if args.gguf:
        gguf_path = out / "gguf"
        gguf_path.mkdir(parents=True, exist_ok=True)
        print(f"Exporting GGUF Q4_K_M → {gguf_path}")
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("GGUF export complete.")

    print(
        f"\nDone. To serve with vLLM:\n"
        f"  vllm serve {merged_path} \\\n"
        f"      --quantization bitsandbytes \\\n"
        f"      --load-format bitsandbytes \\\n"
        f"      --max-model-len 1536 \\\n"
        f"      --tensor-parallel-size 1 \\\n"
        f"      --served-model-name qwen25coder7b-ehrsql"
    )


if __name__ == "__main__":
    main()
