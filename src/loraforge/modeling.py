"""GPU-only model loading and efficient four-code class scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import ExperimentConfig
from .prompts import class_code_token_ids, prompt_token_ids


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
    import torch

    token_rows = [prompt_token_ids(tokenizer, text) for text in texts]
    too_long = [len(row) for row in token_rows if len(row) > max_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts exceed frozen max_length={max_length}; "
            "do not silently truncate evaluation text"
        )
    code_ids = torch.tensor(class_code_token_ids(tokenizer), device=model.device)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(token_rows), batch_size):
            encoded = tokenizer.pad(
                [
                    {"input_ids": row, "attention_mask": [1] * len(row)}
                    for row in token_rows[start : start + batch_size]
                ],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            next_token_logits = model(**encoded).logits[:, -1, :]
            chunks.append(next_token_logits.index_select(1, code_ids).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def peak_cuda_memory_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return float(torch.cuda.max_memory_allocated() / 1024**3)
