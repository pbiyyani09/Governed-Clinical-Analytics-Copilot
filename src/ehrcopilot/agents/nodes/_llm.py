"""Thin LLM call wrapper — swappable between HF pipeline and vLLM."""

from __future__ import annotations

import os
from typing import Any

_pipeline: Any = None  # lazy-loaded


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from ehrcopilot import config
    from transformers import pipeline, BitsAndBytesConfig  # type: ignore[import]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    _pipeline = pipeline(
        "text-generation",
        model=config.INFERENCE_MODEL,
        device_map="auto",
        model_kwargs={"quantization_config": bnb},
        max_new_tokens=256,
        do_sample=False,
    )
    return _pipeline


def call_llm(
    system: str,
    user: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Call the configured LLM and return the assistant text response."""
    pipe = _get_pipeline()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    out = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
    )

    generated = out[0]["generated_text"]
    if isinstance(generated, list):
        for msg in reversed(generated):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg["content"]).strip()
    return str(generated).strip()
