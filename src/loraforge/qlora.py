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


def parameter_report(model) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
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
        "only_lora_parameters_trainable": True,
    }
