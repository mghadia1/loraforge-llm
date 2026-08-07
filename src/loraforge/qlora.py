"""Attach and audit the frozen LoRA adapter configuration."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig


def attach_lora(model, config: ExperimentConfig):
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    prepared = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.training.gradient_checkpointing
    )
    peft_config = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(prepared, peft_config)


def effective_numel(parameter) -> int:
    """True parameter count, undoing bitsandbytes' 4-bit packing.

    A `Params4bit` tensor stores two 4-bit weights per byte, so `numel()` returns
    the number of *stored elements*, not parameters. Counting it raw halves the
    denominator and doubles the reported trainable percentage. Storage dtype can
    be wider than a byte, so scale by the element size as PEFT does.
    """
    count = parameter.numel()
    if type(parameter).__name__ != "Params4bit":
        return count
    element_size = getattr(parameter, "element_size", None)
    bytes_per_element = element_size() if callable(element_size) else 1
    return count * 2 * bytes_per_element


def parameter_report(model) -> dict[str, Any]:
    total = sum(effective_numel(parameter) for parameter in model.parameters())
    trainable = sum(
        effective_numel(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    stored_elements = sum(parameter.numel() for parameter in model.parameters())
    if total == 0 or trainable == 0:
        raise ValueError("model has no total or trainable parameters")
    non_adapter_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if non_adapter_trainable:
        raise ValueError(f"unexpected trainable non-adapter parameters: {non_adapter_trainable[:5]}")
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(100 * trainable / total),
        "stored_tensor_elements": int(stored_elements),
        "counting_note": (
            "total_parameters unpacks 4-bit weights (two per stored byte); "
            "stored_tensor_elements is the raw numel() sum and is not a parameter count"
        ),
        "only_lora_parameters_trainable": True,
    }
