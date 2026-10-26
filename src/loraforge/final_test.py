"""Step 5: the one and only publisher-test evaluation, base versus selected adapter."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import Split
from .metrics import evaluation_block, softmax
from .provenance import (
    AUDITED_PACKAGES,
    EvidenceError,
    environment,
    load_logits,
    read_json,
    resolve_adapter_directory,
    save_logits,
    sha256_array,
    sha256_labels,
    utc_now,
    write_json,
)
from .selection import (
    METRIC_TOLERANCE,
    TRAINING_REPORT,
    recompute_block,
    require_frozen_selection,
    training_report_config,
    verify_frozen_selection,
)

FINAL_REPORT = Path("outputs/final-test-report.json")
FINAL_ATTEMPT = Path("outputs/final-test-attempt.json")
CONFIRMATION = "i-am-running-the-single-final-test"
DELTA_NOTE = "positive macro-F1 delta means QLoRA helped; a negative delta is reported as-is"
DECODING_NOTE = (
    "constrained: argmax over the four class-code token logits, so an unparseable "
    "free-text answer is impossible by construction and invalid_prediction_rate is 0"
)
FINAL_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "test_evaluated",
        "test_evaluations_run",
        "split",
        "rows",
        "test_label_sha256",
        "test_row_ids_sha256",
        "model",
        "model_revision",
        "selected_epoch",
        "selected_adapter_hashes",
        "config",
        "environment",
        "decoding",
        "systems",
        "delta",
    }
)
FINAL_SYSTEM_FIELDS = frozenset(
    {
        "metrics_before_temperature",
        "validation_fitted_temperature",
        "metrics_after_temperature",
        "prediction_sha256",
        "scoring_seconds",
        "seconds_per_row",
        "logits",
    }
)
FINAL_ENVIRONMENT_FIELDS = frozenset(
    {"python", "platform", "packages", "gpu_name", "cuda_available", "cuda_version"}
)


def _require_exact_fields(owner: str, value: Any, expected: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise EvidenceError(f"{owner} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise EvidenceError(
            f"{owner} fields differ from schema: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return value


def _verify_final_report_envelope(report: Any) -> None:
    """Reject fields that the final-test protocol does not produce or verify."""
    report = _require_exact_fields("final report", report, FINAL_REPORT_FIELDS)
    if report["decoding"] != DECODING_NOTE:
        raise EvidenceError("final report decoding does not match the frozen protocol")

    recorded_environment = _require_exact_fields(
        "final report environment", report["environment"], FINAL_ENVIRONMENT_FIELDS
    )
    for field in ("python", "platform", "gpu_name", "cuda_version"):
        if type(recorded_environment[field]) is not str or not recorded_environment[field]:
            raise EvidenceError(f"final report environment.{field} must be a nonempty string")
    if recorded_environment["cuda_available"] is not True:
        raise EvidenceError("final report environment.cuda_available must be true")
    packages = recorded_environment["packages"]
    if type(packages) is not dict or set(packages) != set(AUDITED_PACKAGES):
        raise EvidenceError(
            "final report environment.packages must contain exactly the audited packages"
        )
    if any(type(version) is not str or not version for version in packages.values()):
        raise EvidenceError(
            "final report environment package versions must be nonempty strings"
        )

    systems = _require_exact_fields(
        "final report systems", report["systems"], frozenset({"base", "tuned"})
    )
    for name, system in systems.items():
        _require_exact_fields(
            f"final report systems.{name}", system, FINAL_SYSTEM_FIELDS
        )


def _positive_finite_number(owner: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise EvidenceError(f"{owner} must be a positive finite number")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise EvidenceError(f"{owner} must be a positive finite number")
    return float(value)


def _system_block(
    logits: np.ndarray, labels: list[int], temperature: float, seconds: float
) -> dict[str, Any]:
    before = evaluation_block(logits, labels)
    after = evaluation_block(logits, labels, temperature=temperature)
    if not np.array_equal(
        softmax(logits).argmax(1), softmax(logits / temperature).argmax(1)
    ):
        raise EvidenceError("temperature scaling changed argmax predictions on test")
    if before["macro_f1"] != after["macro_f1"]:
        raise EvidenceError("temperature changed macro-F1; calibration must not move predictions")
    return {
        "metrics_before_temperature": before,
        "validation_fitted_temperature": temperature,
        "metrics_after_temperature": after,
        "prediction_sha256": sha256_labels(softmax(logits).argmax(1).tolist()),
        "scoring_seconds": seconds,
        "seconds_per_row": seconds / len(labels),
    }


def _previous_attempt_error(path: Path) -> EvidenceError:
    return EvidenceError(
        f"{path} records a previous final-test attempt; the frozen protocol "
        "forbids an automatic retry"
    )


def _claim_final_test_attempt(root: Path, selected_epoch: int) -> tuple[Path, dict[str, Any]]:
    """Atomically spend the automatic attempt before publisher-test access."""
    path = root / FINAL_ATTEMPT
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "in_progress",
        "stage": "publisher-test loading, scoring, metrics, and final report",
        "selected_epoch": selected_epoch,
        "note": (
            "this durable claim prevents concurrent or automatic retries; preserve it "
            "for protocol review even if the attempt fails"
        ),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(attempt, handle, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise _previous_attempt_error(path) from error
    return path, attempt


def run_final_test(
    config: ExperimentConfig,
    *,
    confirmation: str,
    root: Path = Path("."),
) -> dict[str, Any]:
    """Score the locked 7,600-row publisher test once with base and tuned systems."""
    from .data import load_dataset
    from .modeling import attach_saved_adapter, load_quantized_base, score_class_codes
    from .qlora import verify_saved_adapter_config

    config.validate()
    root = Path(root)
    if config.test_evaluations_allowed != 1:
        raise EvidenceError(
            "this experiment is validation-only or invalid; publisher-test evaluation is disabled"
        )
    if confirmation != CONFIRMATION:
        raise EvidenceError(
            f"the final test evaluation requires the explicit confirmation {CONFIRMATION!r}"
        )
    if (root / FINAL_REPORT).exists():
        raise EvidenceError(
            f"{FINAL_REPORT} already exists; the protocol allows exactly one test evaluation"
        )
    if (root / FINAL_ATTEMPT).exists():
        raise _previous_attempt_error(root / FINAL_ATTEMPT)
    frozen = require_frozen_selection(root=root)
    recorded_config = training_report_config(read_json(root / TRAINING_REPORT))
    if config.to_dict() != recorded_config.to_dict():
        raise EvidenceError(
            "final-test config does not match the complete frozen training config"
        )

    adapter_dir = resolve_adapter_directory(root, frozen["selected_adapter_dir"])
    verify_saved_adapter_config(adapter_dir, config)
    # Re-derive the complete validation-only gate before GPU setup or publisher-test
    # access. A self-consistent adapter snapshot is not enough if the frozen
    # temperature or calibration evidence was edited after selection.
    verify_frozen_selection(root=root)
    base_model, tokenizer = load_quantized_base(config)
    model = attach_saved_adapter(base_model, adapter_dir)
    attempt_path, attempt = _claim_final_test_attempt(root, frozen["selected_epoch"])

    try:
        bundle = load_dataset(allow_test=True, config=config.data)
        test = bundle.require_test()
        if len(test) != config.data.publisher_test_rows:
            raise EvidenceError(
                f"publisher test has {len(test)} rows, expected {config.data.publisher_test_rows}"
            )

        def score(disable_adapter: bool) -> tuple[np.ndarray, float]:
            started = time.perf_counter()
            if disable_adapter:
                with model.disable_adapter():
                    logits = score_class_codes(
                        model,
                        tokenizer,
                        test.texts,
                        batch_size=config.training.per_device_eval_batch_size,
                        max_length=config.training.max_sequence_length,
                    )
            else:
                logits = score_class_codes(
                    model,
                    tokenizer,
                    test.texts,
                    batch_size=config.training.per_device_eval_batch_size,
                    max_length=config.training.max_sequence_length,
                )
            return logits, time.perf_counter() - started

        labels = test.labels
        base_logits, base_seconds = score(True)
        base_reference = save_logits(base_logits, root, "outputs/logits/base-test.npy")
        tuned_logits, tuned_seconds = score(False)
        tuned_reference = save_logits(tuned_logits, root, "outputs/logits/tuned-test.npy")
        base = _system_block(
            base_logits, labels, frozen["validation"]["base"]["temperature"], base_seconds
        )
        tuned = _system_block(
            tuned_logits, labels, frozen["validation"]["tuned"]["temperature"], tuned_seconds
        )
        report = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "test_evaluated": True,
            "test_evaluations_run": 1,
            "split": "publisher test",
            "rows": len(test),
            "test_label_sha256": sha256_labels(labels),
            "test_row_ids_sha256": test.id_sha256(),
            "model": config.model_name,
            "model_revision": config.model_revision,
            "selected_epoch": frozen["selected_epoch"],
            "selected_adapter_hashes": frozen["selected_adapter_hashes"],
            "config": config.to_dict(),
            "environment": environment(),
            "decoding": DECODING_NOTE,
            "systems": {
                "base": {**base, "logits": base_reference},
                "tuned": {**tuned, "logits": tuned_reference},
            },
            "delta": {
                "macro_f1": tuned["metrics_before_temperature"]["macro_f1"]
                - base["metrics_before_temperature"]["macro_f1"],
                "accuracy": tuned["metrics_before_temperature"]["accuracy"]
                - base["metrics_before_temperature"]["accuracy"],
                "ece_after_temperature": tuned["metrics_after_temperature"]["calibration"]["ece"]
                - base["metrics_after_temperature"]["calibration"]["ece"],
                "note": DELTA_NOTE,
            },
        }
        write_json(report, root / FINAL_REPORT)
        frozen["test_evaluated"] = True
        frozen["test_evaluated_at_utc"] = report["created_at_utc"]
        write_json(frozen, root / "outputs" / "frozen-selection.json")
    except Exception as error:  # preserve and block retries after the claimed attempt
        write_json(
            {
                **attempt,
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "note": (
                    "the only automatic final-test attempt is blocked from retry; any logits "
                    "already written under outputs/logits are from this attempt"
                ),
            },
            attempt_path,
        )
        raise

    write_json(
        {
            **attempt,
            "status": "completed",
            "completed_at_utc": report["created_at_utc"],
            "final_report": str(FINAL_REPORT),
        },
        attempt_path,
    )
    return report


def verify_final_report(
    *,
    root: Path = Path("."),
    labels: list[int] | None = None,
    validation_labels: list[int] | None = None,
    test_split: Split | None = None,
) -> dict[str, Any]:
    """Recompute test metrics and prove calibration came from validation."""
    root = Path(root)
    if labels is not None and test_split is not None:
        raise ValueError("provide either labels or test_split, not both")
    report = read_json(root / FINAL_REPORT)
    _verify_final_report_envelope(report)
    if (
        type(report.get("test_evaluations_run")) is not int
        or report["test_evaluations_run"] != 1
    ):
        raise EvidenceError("the final report must record exactly one test evaluation")
    training_report = read_json(root / TRAINING_REPORT)
    frozen = read_json(root / "outputs" / "frozen-selection.json")
    if frozen.get("test_evaluated") is not True:
        raise EvidenceError("frozen selection must record that the test budget was consumed")
    report_timestamp = report.get("created_at_utc")
    if not isinstance(report_timestamp, str) or not report_timestamp:
        raise EvidenceError("final report created_at_utc must be a timestamp string")
    if frozen.get("test_evaluated_at_utc") != report_timestamp:
        raise EvidenceError(
            "frozen selection test-consumption timestamp does not match the final report"
        )
    fitted_temperatures = verify_frozen_selection(root=root, labels=validation_labels)

    expected_provenance = {
        "schema_version": 1,
        "test_evaluated": True,
        "split": "publisher test",
        "model": frozen["model"],
        "model_revision": frozen["model_revision"],
        "selected_epoch": frozen["selected_epoch"],
        "selected_adapter_hashes": frozen["selected_adapter_hashes"],
        "config": training_report["config"],
    }
    for field, expected in expected_provenance.items():
        actual = report.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise EvidenceError(
                f"final report {field} does not match the validated training and frozen evidence"
            )
    systems = report["systems"]

    if test_split is not None:
        labels = test_split.labels
        if test_split.id_sha256() != report.get("test_row_ids_sha256"):
            raise EvidenceError("publisher-test row IDs do not match the final report")
    elif labels is None:
        from .config import DataConfig
        from .data import load_dataset

        bundle = load_dataset(
            allow_test=True,
            config=DataConfig(**training_report["config"]["data"]),
        )
        test = bundle.require_test()
        labels = test.labels
        if test.id_sha256() != report.get("test_row_ids_sha256"):
            raise EvidenceError("publisher-test row IDs do not match the final report")
    if report.get("rows") != len(labels):
        raise EvidenceError("final report row count does not match the publisher-test labels")
    if sha256_labels(labels) != report["test_label_sha256"]:
        raise EvidenceError("test labels no longer match the final report")

    checked = {}
    for name, system in systems.items():
        scoring_seconds = _positive_finite_number(
            f"{name}.scoring_seconds", system["scoring_seconds"]
        )
        seconds_per_row = _positive_finite_number(
            f"{name}.seconds_per_row", system["seconds_per_row"]
        )
        if not math.isclose(
            seconds_per_row,
            scoring_seconds / len(labels),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise EvidenceError(
                f"{name}.seconds_per_row does not match scoring_seconds / rows"
            )
        for field, recorded_temperature in (
            ("validation_fitted_temperature", system["validation_fitted_temperature"]),
            (
                "metrics_after_temperature.temperature",
                system["metrics_after_temperature"]["temperature"],
            ),
        ):
            try:
                difference = abs(
                    float(recorded_temperature) - fitted_temperatures[name]
                )
            except (TypeError, ValueError) as error:
                raise EvidenceError(f"{name}.{field} must be a finite number") from error
            if (
                not np.isfinite(float(recorded_temperature))
                or difference > METRIC_TOLERANCE
            ):
                raise EvidenceError(
                    f"{name}.{field} does not match the temperature fitted on validation"
                )
        logits = load_logits(system["logits"], root=root)
        recompute_block(logits, labels, system["metrics_before_temperature"], f"{name}.before")
        recompute_block(logits, labels, system["metrics_after_temperature"], f"{name}.after")
        predictions = softmax(logits).argmax(1).tolist()
        if sha256_labels(predictions) != system["prediction_sha256"]:
            raise EvidenceError(f"{name} predictions do not match their recorded hash")
        checked[name] = {
            "macro_f1": system["metrics_before_temperature"]["macro_f1"],
            "logits_sha256": sha256_array(logits),
        }

    expected_delta = {
        "macro_f1": (
            systems["tuned"]["metrics_before_temperature"]["macro_f1"]
            - systems["base"]["metrics_before_temperature"]["macro_f1"]
        ),
        "accuracy": (
            systems["tuned"]["metrics_before_temperature"]["accuracy"]
            - systems["base"]["metrics_before_temperature"]["accuracy"]
        ),
        "ece_after_temperature": (
            systems["tuned"]["metrics_after_temperature"]["calibration"]["ece"]
            - systems["base"]["metrics_after_temperature"]["calibration"]["ece"]
        ),
        "note": DELTA_NOTE,
    }
    recorded_delta = report.get("delta")
    if not isinstance(recorded_delta, dict) or set(recorded_delta) != set(expected_delta):
        raise EvidenceError("final report delta fields do not match the frozen protocol")
    for field, expected in expected_delta.items():
        recorded = recorded_delta[field]
        if isinstance(expected, str):
            if recorded != expected:
                raise EvidenceError(f"final report delta.{field} does not match the protocol")
        elif (
            type(recorded) not in (int, float)
            or not np.isfinite(recorded)
            or abs(recorded - expected) > METRIC_TOLERANCE
        ):
            raise EvidenceError(
                f"final report delta.{field} does not match its own system metrics"
            )
    return {
        "verified": True,
        "systems": checked,
        "macro_f1_delta": expected_delta["macro_f1"],
    }
