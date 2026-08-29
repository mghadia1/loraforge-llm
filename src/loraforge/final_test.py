"""Step 5: the one and only publisher-test evaluation, base versus selected adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import Split
from .metrics import evaluation_block, softmax
from .provenance import (
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
    verify_frozen_selection,
)

FINAL_REPORT = Path("outputs/final-test-report.json")
CONFIRMATION = "i-am-running-the-single-final-test"
DELTA_NOTE = "positive macro-F1 delta means QLoRA helped; a negative delta is reported as-is"


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
    frozen = require_frozen_selection(root=root)

    adapter_dir = resolve_adapter_directory(root, frozen["selected_adapter_dir"])
    verify_saved_adapter_config(adapter_dir, config)
    base_model, tokenizer = load_quantized_base(config)
    model = attach_saved_adapter(base_model, adapter_dir)
    bundle = load_dataset(allow_test=True, config=config.data)
    test = bundle.require_test()
    if len(test) != config.data.publisher_test_rows:
        raise EvidenceError(
            f"publisher test has {len(test)} rows, expected {config.data.publisher_test_rows}"
        )

    try:
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
    except Exception as error:  # preserve the failed attempt rather than retrying silently
        write_json(
            {
                "created_at_utc": utc_now(),
                "stage": "test scoring and metrics",
                "error": f"{type(error).__name__}: {error}",
                "selected_epoch": frozen["selected_epoch"],
                "note": "any logits already written under outputs/logits are from this attempt",
            },
            root / "outputs" / "failed-attempts" / f"final-test-{int(time.time())}.json",
        )
        raise

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
        "decoding": (
            "constrained: argmax over the four class-code token logits, so an unparseable "
            "free-text answer is impossible by construction and invalid_prediction_rate is 0"
        ),
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
    systems = report.get("systems")
    if not isinstance(systems, dict) or set(systems) != {"base", "tuned"}:
        raise EvidenceError("final report must contain exactly the base and tuned systems")

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
