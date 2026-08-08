"""Step 5 gate: freeze the selected adapter and its temperatures before test exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .metrics import evaluation_block, fit_temperature, softmax
from .provenance import (
    EvidenceError,
    load_logits,
    read_json,
    sha256_directory,
    utc_now,
    write_json,
)


METRIC_TOLERANCE = 1e-9
TRAINING_REPORT = Path("outputs/training-report.json")
FROZEN_SELECTION = Path("outputs/frozen-selection.json")


def _assert_close(name: str, recorded: float, recomputed: float) -> None:
    if abs(float(recorded) - float(recomputed)) > METRIC_TOLERANCE:
        raise EvidenceError(
            f"{name} recorded as {recorded} but its own logits recompute to {recomputed}"
        )


def recompute_block(logits: np.ndarray, labels: list[int], recorded: dict[str, Any], name: str) -> None:
    """Recompute a metrics block from raw logits and reject any hand-edited number."""
    fresh = evaluation_block(logits, labels, temperature=recorded.get("temperature", 1.0))
    _assert_close(f"{name}.macro_f1", recorded["macro_f1"], fresh["macro_f1"])
    _assert_close(f"{name}.accuracy", recorded["accuracy"], fresh["accuracy"])
    _assert_close(f"{name}.nll", recorded["nll"], fresh["nll"])
    _assert_close(
        f"{name}.calibration.ece", recorded["calibration"]["ece"], fresh["calibration"]["ece"]
    )
    if recorded["confusion_matrix"] != fresh["confusion_matrix"]:
        raise EvidenceError(f"{name}.confusion_matrix does not match its own logits")


def verify_training_report(
    report: dict[str, Any],
    *,
    root: Path = Path("."),
    labels: list[int] | None = None,
    verify_adapters: bool = True,
) -> dict[str, Any]:
    """Re-derive selection and every validation metric from the saved logits."""
    from .training import select_checkpoint

    labels = _validation_labels(report, root, labels)
    base_logits = load_logits(report["base_validation_logits"], root=root)
    recompute_block(base_logits, labels, report["base_validation_metrics"], "base_validation")

    records = []
    for entry in report["epochs"]:
        logits = load_logits(entry["validation_logits"], root=root)
        recompute_block(logits, labels, entry["validation"], f"epoch-{entry['epoch']}")
        if verify_adapters:
            stored = sha256_directory(root / entry["adapter_dir"])["combined_sha256"]
            if stored != entry["adapter_hashes"]["combined_sha256"]:
                raise EvidenceError(f"epoch-{entry['epoch']} adapter files changed since training")
        records.append(entry)

    selected = select_checkpoint(records)
    if selected["epoch"] != report["selection"]["selected_epoch"]:
        raise EvidenceError(
            f"report selects epoch {report['selection']['selected_epoch']} but the rule "
            f"selects epoch {selected['epoch']}"
        )
    if report["selection"]["selected_adapter_hashes"] != selected["adapter_hashes"]:
        raise EvidenceError("selected adapter hashes do not match the selected epoch")
    if verify_adapters:
        selected_dir = root / report["selection"]["selected_adapter_dir"]
        if (
            sha256_directory(selected_dir)["combined_sha256"]
            != selected["adapter_hashes"]["combined_sha256"]
        ):
            raise EvidenceError("adapters/selected does not match the selected epoch checkpoint")
    return selected


def _validation_labels(
    report: dict[str, Any], root: Path, labels: list[int] | None = None
) -> list[int]:
    """Use the pinned validation labels, reloading them unless a caller supplies them."""
    from .config import DataConfig
    from .data import load_dataset
    from .provenance import sha256_labels

    if labels is None:
        bundle = load_dataset(allow_test=False, config=DataConfig(**report["config"]["data"]))
        labels = bundle.validation.labels
    if sha256_labels(labels) != report["validation_label_sha256"]:
        raise EvidenceError("validation labels no longer match the training report")
    return labels


def build_frozen_selection(
    *, root: Path = Path("."), labels: list[int] | None = None
) -> dict[str, Any]:
    """Fit both temperatures on validation only and write the pre-test gate file."""
    root = Path(root)
    target = root / FROZEN_SELECTION
    if target.exists():
        raise EvidenceError(f"{target} already exists; the selection is already frozen")
    report = read_json(root / TRAINING_REPORT)
    if report.get("test_evaluated"):
        raise EvidenceError("training report claims test was evaluated; selection is not clean")
    selected = verify_training_report(report, root=root, labels=labels)

    labels = _validation_labels(report, root, labels)
    base_logits = load_logits(report["base_validation_logits"], root=root)
    tuned_logits = load_logits(selected["validation_logits"], root=root)
    base_temperature = fit_temperature(base_logits, labels)
    tuned_temperature = fit_temperature(tuned_logits, labels)
    for name, logits, temperature in (
        ("base", base_logits, base_temperature),
        ("tuned", tuned_logits, tuned_temperature),
    ):
        if not np.array_equal(
            softmax(logits).argmax(1), softmax(logits / temperature).argmax(1)
        ):
            raise EvidenceError(f"{name} temperature changed argmax predictions on validation")

    frozen = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "test_evaluated": False,
        "model": report["model"],
        "model_revision": report["model_revision"],
        "selected_epoch": selected["epoch"],
        "selected_adapter_dir": report["selection"]["selected_adapter_dir"],
        "selected_adapter_hashes": report["selection"]["selected_adapter_hashes"],
        "selection_rule": report["selection"]["rule"],
        "validation_label_sha256": report["validation_label_sha256"],
        "validation": {
            "base": {
                "logits": report["base_validation_logits"],
                "metrics": report["base_validation_metrics"],
                "temperature": base_temperature,
                "metrics_after_temperature": evaluation_block(
                    base_logits, labels, temperature=base_temperature
                ),
            },
            "tuned": {
                "logits": selected["validation_logits"],
                "metrics": selected["validation"],
                "temperature": tuned_temperature,
                "metrics_after_temperature": evaluation_block(
                    tuned_logits, labels, temperature=tuned_temperature
                ),
            },
        },
    }
    write_json(frozen, target)
    return frozen


def require_frozen_selection(*, root: Path = Path(".")) -> dict[str, Any]:
    """Load the gate file and refuse to continue unless the adapter still matches it."""
    root = Path(root)
    frozen = read_json(root / FROZEN_SELECTION)
    adapter_dir = root / frozen["selected_adapter_dir"]
    if not adapter_dir.is_dir():
        raise EvidenceError(f"selected adapter {adapter_dir} is missing")
    actual = sha256_directory(adapter_dir)["combined_sha256"]
    if actual != frozen["selected_adapter_hashes"]["combined_sha256"]:
        raise EvidenceError(
            "the selected adapter changed after selection was frozen; test evaluation refused"
        )
    if frozen.get("test_evaluated"):
        raise EvidenceError("frozen selection is already marked as test-evaluated")
    return frozen
