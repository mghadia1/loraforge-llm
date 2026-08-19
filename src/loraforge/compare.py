"""Compare two training runs, and refuse to call the comparison controlled if it isn't.

An ablation only attributes a difference to the variable under test when nothing
else moved. Both failures this project hit are checked here: a library stack that
silently changed between runs, and a config that drifted in more than the field
being studied.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .provenance import EvidenceError, read_json


MISSING = {"present": False}


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
    output = {}
    for key in sorted(keys):
        left_present = key in left
        right_present = key in right
        left_value = left[key] if left_present else MISSING
        right_value = right[key] if right_present else MISSING
        if left_present != right_present or left_value != right_value:
            output[key] = (left_value, right_value)
    return output


def _selected_macro_f1(report: dict[str, Any], name: str) -> float:
    """Use the declared checkpoint only after reapplying the selection rule."""
    from .training import select_checkpoint

    epochs = report.get("epochs")
    selection = report.get("selection")
    if not isinstance(epochs, list) or not isinstance(selection, dict):
        raise EvidenceError(f"{name} report has no checkpoint selection evidence")
    selected = select_checkpoint(epochs)
    if selection.get("selected_epoch") != selected.get("epoch"):
        raise EvidenceError(f"{name} report selected epoch does not follow the selection rule")
    try:
        value = float(selected["validation"]["macro_f1"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{name} selected validation macro-F1 is invalid") from error
    if not math.isfinite(value):
        raise EvidenceError(f"{name} selected validation macro-F1 is not finite")
    return value


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
    gpu = [
        baseline["environment"].get("gpu_name"),
        variant["environment"].get("gpu_name"),
    ]
    gpu_matches = gpu[0] is not None and gpu[0] == gpu[1]
    validation_hashes = [
        baseline.get("validation_label_sha256"),
        variant.get("validation_label_sha256"),
    ]
    validation_data_matches = (
        isinstance(validation_hashes[0], str)
        and validation_hashes[0]
        and validation_hashes[0] == validation_hashes[1]
    )
    controlled = (
        not blocking_packages
        and not unexpected_config
        and not missing_expected
        and gpu_matches
        and validation_data_matches
    )

    baseline_best = _selected_macro_f1(baseline, "baseline")
    variant_best = _selected_macro_f1(variant, "variant")
    return {
        "controlled": controlled,
        "package_differences": package_drift,
        "blocking_package_differences": blocking_packages,
        "config_differences": config_drift,
        "unexpected_config_differences": unexpected_config,
        "expected_changes_not_found": missing_expected,
        "gpu": gpu,
        "gpu_matches": gpu_matches,
        "validation_label_sha256": validation_hashes,
        "validation_data_matches": validation_data_matches,
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
    if not comparison["gpu_matches"]:
        reasons.append(f"GPU model changed or is missing: {comparison['gpu']}")
    if not comparison["validation_data_matches"]:
        reasons.append(
            "validation label digest changed or is missing: "
            f"{comparison['validation_label_sha256']}"
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
