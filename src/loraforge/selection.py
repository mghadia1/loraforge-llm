"""Step 5 gate: freeze the selected adapter and its temperatures before test exists."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import evaluation_block, fit_temperature, softmax
from .provenance import (
    EvidenceError,
    load_logits,
    read_json,
    resolve_adapter_directory,
    utc_now,
    verify_directory_snapshot,
    write_json,
)


METRIC_TOLERANCE = 1e-9
TRAINING_REPORT = Path("outputs/training-report.json")
FROZEN_SELECTION = Path("outputs/frozen-selection.json")
SELECTION_RULE = "max validation macro_f1; exact tie resolved to the earlier epoch"


def _assert_close(name: str, recorded: float, recomputed: float) -> None:
    recorded_value = float(recorded)
    recomputed_value = float(recomputed)
    if (
        not math.isfinite(recorded_value)
        or not math.isfinite(recomputed_value)
        or abs(recorded_value - recomputed_value) > METRIC_TOLERANCE
    ):
        raise EvidenceError(
            f"{name} recorded as {recorded} but its own logits recompute to {recomputed}"
        )


def _assert_metric_tree(name: str, recorded: Any, recomputed: Any) -> None:
    """Compare a complete metric tree, using tolerance only for finite floats."""
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict):
            raise EvidenceError(f"{name} must be a metrics object")
        missing = recomputed.keys() - recorded.keys()
        unexpected = recorded.keys() - recomputed.keys()
        if missing or unexpected:
            raise EvidenceError(
                f"{name} metric fields differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for key, value in recomputed.items():
            _assert_metric_tree(f"{name}.{key}", recorded[key], value)
        return
    if isinstance(recomputed, list):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            raise EvidenceError(
                f"{name} recorded length does not match its own logits"
            )
        for index, value in enumerate(recomputed):
            _assert_metric_tree(f"{name}[{index}]", recorded[index], value)
        return
    if isinstance(recomputed, float):
        try:
            _assert_close(name, recorded, recomputed)
        except (TypeError, ValueError) as error:
            raise EvidenceError(f"{name} must be a finite number") from error
        return
    if recorded != recomputed:
        raise EvidenceError(
            f"{name} recorded as {recorded!r} but its own logits recompute to {recomputed!r}"
        )


def recompute_block(logits: np.ndarray, labels: list[int], recorded: dict[str, Any], name: str) -> None:
    """Recompute a metrics block from raw logits and reject any hand-edited number."""
    fresh = evaluation_block(logits, labels, temperature=recorded.get("temperature", 1.0))
    _assert_metric_tree(name, recorded, fresh)


def training_report_config(report: dict[str, Any]):
    """Parse the embedded config with the same strict schema used before training."""
    from .config import config_from_dict

    if not isinstance(report, dict):
        raise EvidenceError("training report must be a JSON object")
    try:
        return config_from_dict(report.get("config"))
    except (TypeError, ValueError) as error:
        raise EvidenceError(
            f"training report config does not match the typed experiment schema: {error}"
        ) from error


def _verify_training_protocol(report: dict[str, Any]):
    """Bind copied training-report provenance and counts to its validated config."""
    from .data import CLASS_NAMES

    config = training_report_config(report)
    expected = {
        "schema_version": 1,
        "model": config.model_name,
        "model_revision": config.model_revision,
        "test_evaluated": False,
        "train_rows": len(CLASS_NAMES) * config.data.train_per_class,
        "validation_rows": len(CLASS_NAMES) * config.data.validation_per_class,
    }
    for field, value in expected.items():
        recorded = report.get(field)
        if type(recorded) is not type(value) or recorded != value:
            raise EvidenceError(
                f"training report {field} does not match its validated experiment config"
            )

    epochs = report.get("epochs")
    expected_epochs = list(range(1, config.training.epochs + 1))
    if not isinstance(epochs, list) or any(not isinstance(item, dict) for item in epochs):
        raise EvidenceError("training report epochs must be a list of checkpoint objects")
    recorded_epochs = [item.get("epoch") for item in epochs]
    if any(type(epoch) is not int for epoch in recorded_epochs) or recorded_epochs != expected_epochs:
        raise EvidenceError(
            "training report epoch sequence does not match config.training.epochs"
        )

    selection = report.get("selection")
    if not isinstance(selection, dict) or selection.get("rule") != SELECTION_RULE:
        raise EvidenceError("training report selection rule does not match the frozen protocol")
    return config


def verify_training_report(
    report: dict[str, Any],
    *,
    root: Path = Path("."),
    labels: list[int] | None = None,
    verify_adapters: bool = True,
) -> dict[str, Any]:
    """Re-derive selection and every validation metric from the saved logits."""
    from .qlora import verify_saved_adapter_config
    from .training import select_checkpoint

    config = _verify_training_protocol(report)
    labels = _validation_labels(report, root, labels, config=config)
    base_logits = load_logits(report["base_validation_logits"], root=root)
    recompute_block(base_logits, labels, report["base_validation_metrics"], "base_validation")

    records = []
    for entry in report["epochs"]:
        logits = load_logits(entry["validation_logits"], root=root)
        recompute_block(logits, labels, entry["validation"], f"epoch-{entry['epoch']}")
        try:
            adapter_dir = resolve_adapter_directory(root, entry["adapter_dir"])
        except EvidenceError as error:
            raise EvidenceError(
                f"epoch-{entry['epoch']} adapter files changed since training: {error}"
            ) from error
        if verify_adapters:
            try:
                verify_directory_snapshot(adapter_dir, entry["adapter_hashes"])
                verify_saved_adapter_config(adapter_dir, config)
            except EvidenceError as error:
                raise EvidenceError(
                    f"epoch-{entry['epoch']} adapter files changed since training: {error}"
                ) from error

        records.append(entry)

    selected = select_checkpoint(records)
    if selected["epoch"] != report["selection"]["selected_epoch"]:
        raise EvidenceError(
            f"report selects epoch {report['selection']['selected_epoch']} but the rule "
            f"selects epoch {selected['epoch']}"
        )
    if report["selection"]["selected_adapter_hashes"] != selected["adapter_hashes"]:
        raise EvidenceError("selected adapter hashes do not match the selected epoch")
    try:
        selected_dir = resolve_adapter_directory(
            root, report["selection"]["selected_adapter_dir"]
        )
    except EvidenceError as error:
        raise EvidenceError(
            f"adapters/selected does not match the selected epoch checkpoint: {error}"
        ) from error
    if verify_adapters:
        try:
            verify_directory_snapshot(
                selected_dir,
                selected["adapter_hashes"],
                mutable_files=frozenset({"README.md"}),
            )
            verify_saved_adapter_config(selected_dir, config)
        except EvidenceError as error:
            raise EvidenceError(
                "adapters/selected does not match the selected epoch checkpoint: "
                f"{error}"
            ) from error
    return selected


def _validation_labels(
    report: dict[str, Any],
    root: Path,
    labels: list[int] | None = None,
    *,
    config=None,
) -> list[int]:
    """Use the pinned validation labels, reloading them unless a caller supplies them."""
    from .data import load_dataset
    from .provenance import sha256_labels

    config = config or _verify_training_protocol(report)
    if labels is None:
        bundle = load_dataset(allow_test=False, config=config.data)
        labels = bundle.validation.labels
    if len(labels) != report["validation_rows"]:
        raise EvidenceError(
            "validation label count does not match the training report and experiment config"
        )
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


def verify_frozen_selection(
    *, root: Path = Path("."), labels: list[int] | None = None
) -> dict[str, float]:
    """Re-fit validation temperatures and verify the complete frozen calibration gate."""
    root = Path(root)
    report = read_json(root / TRAINING_REPORT)
    labels = _validation_labels(report, root, labels)
    selected = verify_training_report(
        report, root=root, labels=labels, verify_adapters=False
    )
    frozen = read_json(root / FROZEN_SELECTION)
    expected_gate = {
        "model": report["model"],
        "model_revision": report["model_revision"],
        "selected_epoch": selected["epoch"],
        "selected_adapter_dir": report["selection"]["selected_adapter_dir"],
        "selected_adapter_hashes": report["selection"]["selected_adapter_hashes"],
        "selection_rule": report["selection"]["rule"],
        "validation_label_sha256": report["validation_label_sha256"],
    }
    for field, expected in expected_gate.items():
        if frozen.get(field) != expected:
            raise EvidenceError(f"frozen selection {field} does not match training evidence")

    sources = {
        "base": (report["base_validation_logits"], report["base_validation_metrics"]),
        "tuned": (selected["validation_logits"], selected["validation"]),
    }
    temperatures = {}
    for name, (logits_reference, metrics) in sources.items():
        recorded = frozen["validation"][name]
        if recorded["logits"] != logits_reference:
            raise EvidenceError(
                f"frozen {name} validation logits do not match training evidence"
            )
        _assert_metric_tree(f"frozen.{name}.metrics", recorded["metrics"], metrics)
        logits = load_logits(logits_reference, root=root)
        fitted_temperature = fit_temperature(logits, labels)
        _assert_close(
            f"frozen.{name}.temperature",
            recorded["temperature"],
            fitted_temperature,
        )
        _assert_close(
            f"frozen.{name}.metrics_after_temperature.temperature",
            recorded["metrics_after_temperature"]["temperature"],
            fitted_temperature,
        )
        recompute_block(
            logits,
            labels,
            recorded["metrics_after_temperature"],
            f"frozen.{name}.metrics_after_temperature",
        )
        temperatures[name] = fitted_temperature
    return temperatures


def require_frozen_selection(*, root: Path = Path(".")) -> dict[str, Any]:
    """Load the gate file and refuse to continue unless the adapter still matches it."""
    root = Path(root)
    frozen = read_json(root / FROZEN_SELECTION)
    adapter_dir = resolve_adapter_directory(root, frozen["selected_adapter_dir"])
    if not adapter_dir.is_dir():
        raise EvidenceError(f"selected adapter {adapter_dir} is missing")
    try:
        verify_directory_snapshot(
            adapter_dir,
            frozen["selected_adapter_hashes"],
            mutable_files=frozenset({"README.md"}),
        )
    except EvidenceError as error:
        raise EvidenceError(
            "the selected adapter changed after selection was frozen; test evaluation refused"
        ) from error
    if frozen.get("test_evaluated"):
        raise EvidenceError("frozen selection is already marked as test-evaluated")
    return frozen
