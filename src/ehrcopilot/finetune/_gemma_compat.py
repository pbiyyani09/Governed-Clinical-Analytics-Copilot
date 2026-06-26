"""Gemma 3 compatibility shims for the preference/RL trainers.

Gemma 3 is a multimodal checkpoint. Its forward derives the causal-mask layout
from ``token_type_ids`` (0 = text token, 1 = image token) and *requires* the
field during training:

    ValueError: `token_type_ids` is required as a model input when training

Unsloth's SFTTrainer collator supplies it, but TRL's ORPO/GRPO trainers tokenize
prompt/chosen/rejected through the plain tokenizer, which omits it. For our
text-only finetune the correct value is all zeros (every token is text), which
yields the standard causal mask. This shim injects that default at the model's
forward boundary so the preference trainers run unmodified.
"""

from __future__ import annotations

from typing import Any


def patch_token_type_ids(model: Any) -> Any:
    """Wrap ``model.forward`` so a missing ``token_type_ids`` defaults to zeros.

    Idempotent and a no-op for non-Gemma models (they simply never pass the
    field down). Returns the same model for chaining.
    """
    import torch

    if getattr(model, "_ehrcopilot_tti_patched", False):
        return model

    original_forward = model.forward

    def forward_with_token_type_ids(*args: Any, **kwargs: Any):
        if kwargs.get("token_type_ids", None) is None:
            # TRL's ORPO/GRPO concatenated_forward passes input_ids positionally
            # (model(concatenated_input_ids, ...)), so check args[0] as well.
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and args and torch.is_tensor(args[0]):
                input_ids = args[0]
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
        return original_forward(*args, **kwargs)

    model.forward = forward_with_token_type_ids
    model._ehrcopilot_tti_patched = True
    return model
