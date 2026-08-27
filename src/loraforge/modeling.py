"""GPU-only model loading and efficient four-code class scoring."""

from __future__ import annotations

from numbers import Integral
from typing import Iterable

import numpy as np

from .config import ExperimentConfig
from .prompts import class_code_token_ids, prompt_token_ids


def _left_pad_token_rows(
    rows: list[list[int]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Pad already-tokenized prompts without depending on tokenizer.pad internals."""
    if not rows:
        raise ValueError("cannot pad an empty prompt batch")
    width = max(len(row) for row in rows)
    input_ids = []
    attention_mask = []
    for row in rows:
        padding = width - len(row)
        input_ids.append([pad_token_id] * padding + row)
        attention_mask.append([0] * padding + [1] * len(row))
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def resolve_last_logit_kwargs(model) -> dict[str, int]:
    """Find the kwarg that limits the LM head to the final position, if it exists.

    Only the last position's logits are ever read, but by default the head is
    applied to all 512, producing a [batch, 512, 32768] tensor. Transformers
    calls the limiter `logits_to_keep` (older versions: `num_logits_to_keep`),
    and a PEFT-wrapped model hides it behind pass-through `**kwargs`, so walk the
    wrapper chain. Returning `{}` is safe: the result is identical either way.
    """
    import inspect

    current = model
    for _ in range(5):
        if current is None:
            break
        try:
            parameters = inspect.signature(current.forward).parameters
        except (AttributeError, TypeError, ValueError):
            parameters = {}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in parameters:
                return {name: 1}
        current = getattr(current, "base_model", None) or getattr(current, "model", None)
    return {}


def load_quantized_base(config: ExperimentConfig):
    """Load the frozen base model in 4-bit NF4. Requires a CUDA runtime."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA; use the T4 notebook")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        config.quantization.compute_dtype
    ]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=config.quantization.use_double_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.model_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    class_code_token_ids(tokenizer)  # fail before evaluation if the contract changed
    return model, tokenizer


def attach_saved_adapter(model, adapter_dir):
    """Load a frozen adapter for inference; the base weights stay quantized and frozen."""
    from peft import PeftModel

    adapted = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    adapted.eval()
    return adapted


def score_class_codes(
    model,
    tokenizer,
    texts: Iterable[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Return A-D next-token logits for each prompt in one forward pass per batch."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    batch_size, max_length = int(batch_size), int(max_length)

    token_rows = [prompt_token_ids(tokenizer, text) for text in texts]
    if not token_rows:
        raise ValueError("cannot score an empty text collection")
    too_long = [len(row) for row in token_rows if len(row) > max_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts exceed frozen max_length={max_length}; "
            "do not silently truncate evaluation text"
        )

    import torch

    code_ids = torch.tensor(class_code_token_ids(tokenizer), device=model.device)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad token ID")
    last_logit_kwargs = resolve_last_logit_kwargs(model)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(token_rows), batch_size):
            padded = _left_pad_token_rows(
                token_rows[start : start + batch_size], int(tokenizer.pad_token_id)
            )
            encoded = {
                key: torch.tensor(value, device=model.device)
                for key, value in padded.items()
            }
            # [:, -1, :] is correct whether the head ran on one position or all.
            next_token_logits = model(**encoded, **last_logit_kwargs).logits[:, -1, :]
            chunks.append(next_token_logits.index_select(1, code_ids).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def peak_cuda_memory_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return float(torch.cuda.max_memory_allocated() / 1024**3)
