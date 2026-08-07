from __future__ import annotations

import pytest

from loraforge.qlora import parameter_report


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
