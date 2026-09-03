"""Attach and audit the frozen LoRA adapter configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .provenance import EvidenceError


def verify_saved_adapter_config(
    adapter_dir: Path, config: ExperimentConfig
) -> dict[str, Any]:
    """Bind a saved PEFT adapter's executable protocol to the experiment config.

    Directory hashes prove that an adapter did not change after it was recorded,
    but a self-consistent snapshot could still have been created with the wrong
    base model or LoRA settings. Validate the stable, protocol-defining PEFT
    fields while allowing unrelated metadata added by different PEFT versions.
    """
    path = Path(adapter_dir) / "adapter_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read saved adapter config {path}: {error}") from error
    if type(payload) is not dict:
        raise EvidenceError("saved adapter config must be a JSON object")

    expected_scalars = {
        "base_model_name_or_path": config.model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
    }
    for field, expected in expected_scalars.items():
        actual = payload.get(field)
        if type(expected) is int:
            matches = type(actual) is int and actual == expected
        elif type(expected) is float:
            matches = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and actual == expected
            )
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise EvidenceError(
                f"saved adapter {field}={actual!r} does not match the frozen "
                f"protocol value {expected!r}"
            )

    targets = payload.get("target_modules")
    if (
        type(targets) is not list
        or not targets
        or any(type(target) is not str or not target for target in targets)
        or len(set(targets)) != len(targets)
        or set(targets) != set(config.lora.target_modules)
    ):
        raise EvidenceError(
            "saved adapter target_modules do not match the frozen LoRA targets"
        )

    # These optional PEFT features override or extend the simple frozen LoRA
    # contract. Missing fields are equivalent to their disabled defaults.
    disabled_features = {
        "alpha_pattern": ({}, None),
        "rank_pattern": ({}, None),
        "exclude_modules": (None,),
        "layer_replication": (None,),
        "layers_to_transform": (None,),
        "modules_to_save": (None,),
        "target_parameters": (None,),
        "trainable_token_indices": (None,),
        "use_dora": (False, None),
        "use_qalora": (False, None),
        "use_rslora": (False, None),
    }
    for field, allowed in disabled_features.items():
        if field in payload and not any(
            type(payload[field]) is type(expected) and payload[field] == expected
            for expected in allowed
        ):
            raise EvidenceError(
                f"saved adapter enables unsupported PEFT feature {field}"
            )
    if payload.get("lora_bias", False) is not False:
        raise EvidenceError("saved adapter enables unsupported PEFT feature lora_bias")

    revision = payload.get("revision")
    if revision is not None and revision != config.model_revision:
        raise EvidenceError(
            "saved adapter revision does not match the frozen base-model revision"
        )
    return payload


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
