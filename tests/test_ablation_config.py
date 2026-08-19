from __future__ import annotations

from pathlib import Path

import pytest

from loraforge.config import load_config


FROZEN = Path("configs/experiment.json")
RANK4 = Path("configs/experiment-rank4.json")


def flatten(payload: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_rank4_ablation_changes_only_capacity_and_test_budget() -> None:
    """A capacity ablation must vary capacity and nothing else."""
    frozen = flatten(load_config(FROZEN).to_dict())
    ablation = flatten(load_config(RANK4).to_dict())
    differences = {
        key: (frozen[key], ablation[key])
        for key in frozen
        if frozen[key] != ablation[key]
    }
    assert set(differences) == {"lora.rank", "lora.alpha", "test_evaluations_allowed"}, (
        f"the ablation drifted from the frozen run: {differences}"
    )


def test_alpha_scales_with_rank_so_update_scaling_is_unchanged() -> None:
    """LoRA scales its update by alpha/rank.

    Holding alpha fixed while lowering rank would multiply the update by four and
    confound capacity with effective step size, so the comparison could not
    attribute any change to rank alone.
    """
    frozen = load_config(FROZEN).lora
    ablation = load_config(RANK4).lora
    assert ablation.rank < frozen.rank
    assert ablation.alpha / ablation.rank == pytest.approx(frozen.alpha / frozen.rank)


def test_the_ablation_cannot_spend_a_test_budget_that_is_already_used() -> None:
    assert load_config(RANK4).test_evaluations_allowed == 0
    assert load_config(FROZEN).test_evaluations_allowed == 1
