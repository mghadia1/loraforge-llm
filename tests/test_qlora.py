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


def test_parameter_report_rejects_accidentally_trainable_base_weight() -> None:
    model = Model(
        [
            ("base.weight", Parameter(10_000, True)),
            ("layer.lora_A.weight", Parameter(100, True)),
        ]
    )
    with pytest.raises(ValueError, match="non-adapter"):
        parameter_report(model)
