"""Compare two training runs, and refuse to call the comparison controlled if it isn't.

An ablation only attributes a difference to the variable under test when nothing
else moved. Both failures this project hit are checked here: a library stack that
silently changed between runs, and a config that drifted in more than the field
being studied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .provenance import EvidenceError, read_json


def flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = set(left) | set(right)
    return {
        key: (left.get(key), right.get(key))
        for key in sorted(keys)
        if left.get(key) != right.get(key)
    }


def compare_runs(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    *,
    expected_config_changes: set[str] | None = None,
) -> dict[str, Any]:
    """Diff two training reports and judge whether the comparison is controlled."""
    expected = expected_config_changes or set()

    package_drift = differences(
        baseline["environment"]["packages"], variant["environment"]["packages"]
    )
    config_drift = differences(flatten(baseline["config"]), flatten(variant["config"]))
    unexpected_config = {k: v for k, v in config_drift.items() if k not in expected}
    missing_expected = sorted(expected - set(config_drift))

    # torch comes from the host image and is reported rather than policed; every
    # other library is installed by us and must match for the comparison to hold.
    blocking_packages = {k: v for k, v in package_drift.items() if k != "torch"}
    controlled = not blocking_packages and not unexpected_config and not missing_expected

    baseline_best = max(e["validation"]["macro_f1"] for e in baseline["epochs"])
    variant_best = max(e["validation"]["macro_f1"] for e in variant["epochs"])
    return {
        "controlled": controlled,
        "package_differences": package_drift,
        "blocking_package_differences": blocking_packages,
        "config_differences": config_drift,
        "unexpected_config_differences": unexpected_config,
        "expected_changes_not_found": missing_expected,
        "gpu": [
            baseline["environment"].get("gpu_name"),
            variant["environment"].get("gpu_name"),
        ],
        "validation_macro_f1": {
            "baseline": baseline_best,
            "variant": variant_best,
            "delta": variant_best - baseline_best,
        },
        "trainable_parameters": {
            "baseline": baseline.get("parameters", {}).get("trainable_parameters"),
            "variant": variant.get("parameters", {}).get("trainable_parameters"),
        },
        "wall_time_seconds": {
            "baseline": baseline.get("wall_time_seconds"),
            "variant": variant.get("wall_time_seconds"),
        },
    }


def require_controlled(comparison: dict[str, Any]) -> None:
    """Raise unless the only thing that changed is what the ablation meant to change."""
    if comparison["controlled"]:
        return
    reasons = []
    if comparison["blocking_package_differences"]:
        reasons.append(f"library versions changed: {comparison['blocking_package_differences']}")
    if comparison["unexpected_config_differences"]:
        reasons.append(f"unexpected config changes: {comparison['unexpected_config_differences']}")
    if comparison["expected_changes_not_found"]:
        reasons.append(
            f"the fields under test did not actually change: {comparison['expected_changes_not_found']}"
        )
    raise EvidenceError(
        "this comparison is not controlled, so a difference cannot be attributed to the "
        "variable under test — " + "; ".join(reasons)
    )


def compare_report_files(
    baseline_path: Path, variant_path: Path, *, expected_config_changes: set[str] | None = None
) -> dict[str, Any]:
    return compare_runs(
        read_json(baseline_path),
        read_json(variant_path),
        expected_config_changes=expected_config_changes,
    )
