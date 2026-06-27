"""Gemma 3 compatibility shim for the preference/RL trainers.

Gemma 3 is multimodal; its forward derives the causal-mask layout from
``token_type_ids`` (0 = text, 1 = image) and *requires* the field during training:

    ValueError: `token_type_ids` is required as a model input when training

Unsloth's SFTTrainer collator supplies it, but TRL's ORPO/GRPO trainers tokenize
prompt/chosen/rejected through the plain tokenizer (and pass input_ids
positionally), which omits it. For text-only finetuning the correct value is all
zeros (every token is text → standard causal mask). This shim injects that default
at the model's forward boundary so the preference trainers run unmodified.
"""

from __future__ import annotations

from typing import Any


def patch_token_type_ids(model: Any) -> Any:
    """Wrap ``model.forward`` so a missing ``token_type_ids`` defaults to zeros.

    Idempotent; reads input_ids from kwargs OR the first positional arg (TRL's
    concatenated_forward passes it positionally). Returns the model for chaining.
    """
    import torch

    if getattr(model, "_ehrcopilot_tti_patched", False):
        return model
    original_forward = model.forward

    def forward_with_token_type_ids(*args: Any, **kwargs: Any):
        if kwargs.get("token_type_ids", None) is None:
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and args and torch.is_tensor(args[0]):
                input_ids = args[0]
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
        return original_forward(*args, **kwargs)

    model.forward = forward_with_token_type_ids
    model._ehrcopilot_tti_patched = True
    return model
