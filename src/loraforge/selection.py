"""Step 5 gate: freeze the selected adapter and its temperatures before test exists."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import evaluation_block, fit_temperature, softmax
from .provenance import (
    AUDITED_PACKAGES,
    EvidenceError,
    PRETRAINING_MANIFEST_PATH,
    load_logits,
    read_json,
    resolve_adapter_directory,
    resolve_evidence_file,
    sha256_file,
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


def _manifest_object(manifest: dict[str, Any], field: str) -> dict[str, Any]:
    value = manifest.get(field)
    if type(value) is not dict:
        raise EvidenceError(f"pre-training manifest {field} must be a JSON object")
    return value


def _verify_manifest_environment(manifest: dict[str, Any]) -> None:
    recorded = _manifest_object(manifest, "environment")
    required = {"python", "platform", "packages", "gpu_name", "cuda_available"}
    optional = {"cuda_version"}
    missing = required - recorded.keys()
    unexpected = recorded.keys() - required - optional
    if missing or unexpected:
        raise EvidenceError(
            "pre-training manifest environment fields differ from schema: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for field in ("python", "platform"):
        if type(recorded[field]) is not str or not recorded[field]:
            raise EvidenceError(
                f"pre-training manifest environment.{field} must be a nonempty string"
            )
    packages = recorded["packages"]
    if type(packages) is not dict or set(packages) != set(AUDITED_PACKAGES):
        raise EvidenceError(
            "pre-training manifest environment.packages must contain exactly the "
            "audited package versions"
        )
    if any(type(value) is not str or not value for value in packages.values()):
        raise EvidenceError(
            "pre-training manifest environment package versions must be nonempty strings"
        )
    cuda_available = recorded["cuda_available"]
    if type(cuda_available) is not bool:
        raise EvidenceError(
            "pre-training manifest environment.cuda_available must be a JSON boolean"
        )
    if cuda_available:
        for field in ("gpu_name", "cuda_version"):
            if type(recorded.get(field)) is not str or not recorded[field]:
                raise EvidenceError(
                    f"pre-training manifest environment.{field} must be a nonempty "
                    "string when CUDA is available"
                )
    elif recorded["gpu_name"] is not None or "cuda_version" in recorded:
        raise EvidenceError(
            "pre-training manifest environment cannot claim GPU/CUDA details when "
            "CUDA is unavailable"
        )


def _verify_manifest_training_arguments(manifest: dict[str, Any], config) -> None:
    from .data import CLASS_NAMES

    recorded = _manifest_object(manifest, "training_arguments")
    required = {
        "output_dir",
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "optim",
        "fp16",
        "seed",
    }
    optional = {
        "logging_steps",
        "save_strategy",
        "report_to",
        "remove_unused_columns",
        "warmup_ratio",
        "warmup_steps",
    }
    missing = required - recorded.keys()
    unexpected = recorded.keys() - required - optional
    if missing or unexpected:
        raise EvidenceError(
            "pre-training manifest training_arguments fields differ from schema: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    expected = {
        "num_train_epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "optim": config.training.optimizer,
        "fp16": config.quantization.compute_dtype == "float16",
        "seed": config.data.seed,
    }
    for field, value in expected.items():
        actual = recorded[field]
        if type(actual) is not type(value) or actual != value:
            raise EvidenceError(
                f"pre-training manifest training_arguments.{field} does not match config"
            )
    if type(recorded["output_dir"]) is not str or not recorded["output_dir"]:
        raise EvidenceError(
            "pre-training manifest training_arguments.output_dir must be a nonempty string"
        )
    warmup_fields = {"warmup_ratio", "warmup_steps"} & recorded.keys()
    if len(warmup_fields) != 1:
        raise EvidenceError(
            "pre-training manifest training_arguments must contain exactly one warmup control"
        )
    if "warmup_ratio" in recorded:
        if (
            type(recorded["warmup_ratio"]) not in (int, float)
            or not math.isfinite(recorded["warmup_ratio"])
            or recorded["warmup_ratio"] != config.training.warmup_ratio
        ):
            raise EvidenceError(
                "pre-training manifest training_arguments.warmup_ratio does not match config"
            )
    else:
        effective_batch = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        )
        train_rows = len(CLASS_NAMES) * config.data.train_per_class
        optimizer_steps = math.ceil(train_rows / effective_batch) * config.training.epochs
        expected_steps = max(1, round(config.training.warmup_ratio * optimizer_steps))
        if type(recorded["warmup_steps"]) is not int or recorded["warmup_steps"] != expected_steps:
            raise EvidenceError(
                "pre-training manifest training_arguments.warmup_steps does not match config"
            )
    cosmetic = {
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
    }
    for field, value in cosmetic.items():
        if field in recorded and (
            type(recorded[field]) is not type(value) or recorded[field] != value
        ):
            raise EvidenceError(
                f"pre-training manifest training_arguments.{field} is invalid"
            )


def _verify_manifest_parameters(manifest: dict[str, Any]) -> None:
    recorded = _manifest_object(manifest, "parameters")
    expected_fields = {
        "total_parameters",
        "trainable_parameters",
        "trainable_percent",
        "stored_tensor_elements",
        "counting_note",
        "only_lora_parameters_trainable",
    }
    if set(recorded) != expected_fields:
        raise EvidenceError(
            "pre-training manifest parameters fields differ from schema"
        )
    for field in ("total_parameters", "trainable_parameters", "stored_tensor_elements"):
        if type(recorded[field]) is not int or recorded[field] <= 0:
            raise EvidenceError(
                f"pre-training manifest parameters.{field} must be a positive integer"
            )
    total = recorded["total_parameters"]
    trainable = recorded["trainable_parameters"]
    stored = recorded["stored_tensor_elements"]
    if trainable > total or stored > total:
        raise EvidenceError(
            "pre-training manifest parameter counts are internally inconsistent"
        )
    percentage = recorded["trainable_percent"]
    if (
        type(percentage) is not float
        or not math.isfinite(percentage)
        or abs(percentage - (100 * trainable / total)) > 1e-12
    ):
        raise EvidenceError(
            "pre-training manifest parameters.trainable_percent does not match its counts"
        )
    if type(recorded["counting_note"]) is not str or not recorded["counting_note"]:
        raise EvidenceError(
            "pre-training manifest parameters.counting_note must be a nonempty string"
        )
    if recorded["only_lora_parameters_trainable"] is not True:
        raise EvidenceError(
            "pre-training manifest must record that only LoRA parameters are trainable"
        )


def _verify_pretraining_manifest(report: dict[str, Any], root: Path, config) -> None:
    """Bind schema-v2 training controls to the file captured before training."""
    reference = report.get("pretraining_manifest")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise EvidenceError(
            "schema-v2 training report must contain an exact pretraining_manifest reference"
        )
    if reference.get("path") != PRETRAINING_MANIFEST_PATH:
        raise EvidenceError(
            "pre-training manifest must use the canonical outputs path"
        )
    target = resolve_evidence_file(
        root,
        reference["path"],
        label="pre-training manifest",
        suffix=".json",
    )
    if not target.is_file():
        raise EvidenceError(f"required pre-training manifest {target} does not exist")
    actual_sha256 = sha256_file(target)
    if reference.get("sha256") != actual_sha256:
        raise EvidenceError(
            "pre-training manifest hash does not match the training report reference"
        )

    manifest = read_json(target)
    expected_fields = {
        "schema_version",
        "created_at_utc",
        "stage",
        "test_loaded",
        "config",
        "environment",
        "training_arguments",
        "parameters",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise EvidenceError("pre-training manifest fields do not match schema version 1")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise EvidenceError("pre-training manifest schema_version must be integer 1")
    if (
        not isinstance(manifest.get("created_at_utc"), str)
        or not manifest["created_at_utc"].endswith("Z")
    ):
        raise EvidenceError("pre-training manifest created_at_utc must be a UTC timestamp")
    if manifest.get("stage") != "before_optimizer_training":
        raise EvidenceError("pre-training manifest stage is not before optimizer training")
    if manifest.get("test_loaded") is not False:
        raise EvidenceError("pre-training manifest must keep publisher test locked")
    _verify_manifest_environment(manifest)
    _verify_manifest_training_arguments(manifest, config)
    _verify_manifest_parameters(manifest)
    for field in ("config", "environment", "training_arguments", "parameters"):
        if manifest.get(field) != report.get(field):
            raise EvidenceError(
                f"training report {field} does not match its hash-bound pre-training manifest"
            )


def _verify_training_protocol(report: dict[str, Any], *, root: Path = Path(".")):
    """Bind copied training-report provenance and counts to its validated config."""
    from .data import CLASS_NAMES

    config = training_report_config(report)
    schema_version = report.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise EvidenceError("training report schema_version must be integer 1 or 2")
    if schema_version == 2:
        _verify_pretraining_manifest(report, root, config)
    elif "pretraining_manifest" in report:
        raise EvidenceError(
            "schema-v1 training report cannot claim a pre-training manifest"
        )
    expected = {
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

    config = _verify_training_protocol(report, root=root)
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
