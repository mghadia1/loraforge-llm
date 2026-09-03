from __future__ import annotations

import json

import pytest

from loraforge.config import default_config
from loraforge.provenance import EvidenceError
from loraforge.qlora import parameter_report, verify_saved_adapter_config


class Parameter:
    def __init__(self, size: int, requires_grad: bool):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class Model:
    def __init__(self, values):
        self.values = values

    def parameters(self):
        return [parameter for _, parameter in self.values]

    def named_parameters(self):
        return self.values


def test_parameter_report_requires_only_lora_trainable() -> None:
    model = Model(
        [
            ("base.weight", Parameter(10_000, False)),
            ("layer.lora_A.weight", Parameter(100, True)),
            ("layer.lora_B.weight", Parameter(100, True)),
        ]
    )
    report = parameter_report(model)
    assert report["trainable_parameters"] == 200
    assert report["trainable_percent"] == pytest.approx(200 / 10_200 * 100)


class Params4bit(Parameter):
    """Stands in for the bitsandbytes tensor that packs two weights per byte."""

    def element_size(self):
        return 1


def test_four_bit_weights_are_unpacked_before_the_percentage_is_computed() -> None:
    model = Model(
        [
            ("base.weight", Params4bit(3_500_000_000, False)),
            ("layer.lora_A.weight", Parameter(20_000_000, True)),
            ("layer.lora_B.weight", Parameter(20_000_000, True)),
        ]
    )
    report = parameter_report(model)
    # 3.5e9 stored bytes hold 7e9 real parameters, so the denominator doubles
    assert report["total_parameters"] == 7_040_000_000
    assert report["stored_tensor_elements"] == 3_540_000_000
    assert report["trainable_percent"] == pytest.approx(100 * 40_000_000 / 7_040_000_000)
    # counting the packed tensor raw would have inflated this to ~1.13%
    assert report["trainable_percent"] < 0.6


def test_parameter_report_rejects_accidentally_trainable_base_weight() -> None:
    model = Model(
        [
            ("base.weight", Parameter(10_000, True)),
            ("layer.lora_A.weight", Parameter(100, True)),
        ]
    )
    with pytest.raises(ValueError, match="non-adapter"):
        parameter_report(model)


def write_saved_config(path, **updates) -> None:
    config = default_config()
    payload = {
        "base_model_name_or_path": config.model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
        "target_modules": list(reversed(config.lora.target_modules)),
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
    }
    payload.update(updates)
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_saved_adapter_config_matches_the_frozen_protocol(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    write_saved_config(adapter)

    verified = verify_saved_adapter_config(adapter, default_config())

    assert verified["r"] == 16


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"base_model_name_or_path": "unrelated/model"}, "base_model_name_or_path"),
        ({"r": 4}, "saved adapter r"),
        ({"lora_alpha": 16}, "lora_alpha"),
        ({"lora_dropout": 0.0}, "lora_dropout"),
        ({"target_modules": ["q_proj"]}, "target_modules"),
        ({"rank_pattern": {"q_proj": 4}}, "rank_pattern"),
        ({"use_dora": True}, "use_dora"),
        ({"use_dora": 0}, "use_dora"),
        ({"revision": "0" * 40}, "revision"),
    ],
)
def test_saved_adapter_config_rejects_protocol_drift(tmp_path, updates, message) -> None:
    adapter = tmp_path / "adapter"
    write_saved_config(adapter, **updates)

    with pytest.raises(EvidenceError, match=message):
        verify_saved_adapter_config(adapter, default_config())
