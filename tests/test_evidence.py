from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from loraforge.config import default_config
from loraforge.data import Example, Split
from loraforge.final_test import (
    CONFIRMATION,
    _system_block,
    run_final_test,
    verify_final_report,
)
from loraforge.metrics import evaluation_block, fit_temperature
from loraforge.provenance import (
    EvidenceError,
    load_logits,
    read_json,
    save_logits,
    sha256_directory,
    sha256_labels,
    write_json,
)
from loraforge.selection import (
    build_frozen_selection,
    require_frozen_selection,
    verify_training_report,
)


LABELS = [index % 4 for index in range(40)]


def logits_for(labels, margin: float, seed: int) -> np.ndarray:
    """Noisy logits nudged toward the truth; a larger margin means a better system."""
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=1.5, size=(len(labels), 4))
    values[np.arange(len(labels)), np.asarray(labels)] += margin
    return values.astype(np.float32)


def write_adapter(root: Path, relative: str, payload: str) -> str:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_text(payload, encoding="utf-8")
    return relative


def make_training_run(
    root: Path, *, epoch_margins=(1.5, 2.5), epoch_seeds=(11, 12), base_margin: float = 0.4
) -> dict:
    """Write a complete synthetic training run whose numbers match its own logits."""
    base_logits = logits_for(LABELS, base_margin, seed=1)
    epochs = []
    for index, (margin, seed) in enumerate(zip(epoch_margins, epoch_seeds), start=1):
        logits = logits_for(LABELS, margin, seed=seed)
        relative = write_adapter(root, f"adapters/epoch-{index}", f"weights-{index}")
        epochs.append(
            {
                "epoch": index,
                "adapter_dir": relative,
                "adapter_hashes": sha256_directory(root / relative),
                "validation": evaluation_block(logits, LABELS),
                "validation_logits": save_logits(
                    logits, root, f"outputs/logits/tuned-validation-epoch-{index}.npy"
                ),
            }
        )
    best = max(epochs, key=lambda entry: entry["validation"]["macro_f1"])
    selected = write_adapter(
        root, "adapters/selected", f"weights-{best['epoch']}"
    )
    report = {
        "schema_version": 1,
        "model": default_config().model_name,
        "model_revision": default_config().model_revision,
        "config": default_config().to_dict(),
        "test_evaluated": False,
        "validation_label_sha256": sha256_labels(LABELS),
        "base_validation_metrics": evaluation_block(base_logits, LABELS),
        "base_validation_logits": save_logits(
            base_logits, root, "outputs/logits/base-validation.npy"
        ),
        "epochs": epochs,
        "selection": {
            "rule": "max validation macro_f1; exact tie resolved to the earlier epoch",
            "selected_epoch": best["epoch"],
            "selected_adapter_dir": selected,
            "selected_adapter_hashes": sha256_directory(root / selected),
        },
    }
    write_json(report, root / "outputs" / "training-report.json")
    return report


def make_final_report(root: Path, *, tuned_margin: float = 2.5) -> dict:
    frozen = read_json(root / "outputs" / "frozen-selection.json")
    base_logits = logits_for(LABELS, 0.4, seed=21)
    tuned_logits = logits_for(LABELS, tuned_margin, seed=22)
    base = _system_block(base_logits, LABELS, frozen["validation"]["base"]["temperature"], 3.0)
    tuned = _system_block(tuned_logits, LABELS, frozen["validation"]["tuned"]["temperature"], 3.2)
    report = {
        "schema_version": 1,
        "test_evaluated": True,
        "test_evaluations_run": 1,
        "split": "publisher test",
        "rows": len(LABELS),
        "test_label_sha256": sha256_labels(LABELS),
        "model": frozen["model"],
        "model_revision": frozen["model_revision"],
        "selected_epoch": frozen["selected_epoch"],
        "config": default_config().to_dict(),
        "selected_adapter_hashes": frozen["selected_adapter_hashes"],
        "systems": {
            "base": {**base, "logits": save_logits(base_logits, root, "outputs/logits/base-test.npy")},
            "tuned": {
                **tuned,
                "logits": save_logits(tuned_logits, root, "outputs/logits/tuned-test.npy"),
            },
        },
        "delta": {
            "macro_f1": tuned["metrics_before_temperature"]["macro_f1"]
            - base["metrics_before_temperature"]["macro_f1"]
        },
    }
    write_json(report, root / "outputs" / "final-test-report.json")
    return report


def test_frozen_selection_records_two_separate_validation_temperatures(tmp_path) -> None:
    make_training_run(tmp_path)
    frozen = build_frozen_selection(root=tmp_path, labels=LABELS)
    assert frozen["selected_epoch"] == 2
    assert frozen["test_evaluated"] is False
    base_temperature = frozen["validation"]["base"]["temperature"]
    tuned_temperature = frozen["validation"]["tuned"]["temperature"]
    assert base_temperature != tuned_temperature
    assert base_temperature == pytest.approx(
        fit_temperature(logits_for(LABELS, 0.4, seed=1), LABELS)
    )
    # calibration may not move the decision, only the confidence
    assert (
        frozen["validation"]["tuned"]["metrics_after_temperature"]["macro_f1"]
        == frozen["validation"]["tuned"]["metrics"]["macro_f1"]
    )


def test_selection_tie_break_is_honoured_end_to_end(tmp_path) -> None:
    make_training_run(tmp_path, epoch_margins=(2.5, 2.5), epoch_seeds=(12, 12))
    report = read_json(tmp_path / "outputs" / "training-report.json")
    assert (
        report["epochs"][0]["validation"]["macro_f1"]
        == report["epochs"][1]["validation"]["macro_f1"]
    )
    assert verify_training_report(report, root=tmp_path, labels=LABELS)["epoch"] == 1


def test_freezing_twice_is_refused(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    with pytest.raises(EvidenceError, match="already frozen"):
        build_frozen_selection(root=tmp_path, labels=LABELS)


def test_hand_edited_macro_f1_in_the_training_report_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    path = tmp_path / "outputs" / "training-report.json"
    report = read_json(path)
    report["epochs"][1]["validation"]["macro_f1"] = 0.99
    write_json(report, path)
    with pytest.raises(EvidenceError, match="recompute"):
        verify_training_report(read_json(path), root=tmp_path, labels=LABELS)


def test_hand_edited_per_class_metric_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    path = tmp_path / "outputs" / "training-report.json"
    report = read_json(path)
    report["epochs"][1]["validation"]["per_class"]["World"]["recall"] = 0.99

    with pytest.raises(EvidenceError, match=r"epoch-2\.per_class\.World\.recall"):
        verify_training_report(report, root=tmp_path, labels=LABELS)


def test_swapped_logits_file_is_rejected_by_its_hash(tmp_path) -> None:
    make_training_run(tmp_path)
    np.save(
        tmp_path / "outputs" / "logits" / "tuned-validation-epoch-2.npy",
        logits_for(LABELS, 9.0, seed=99),
    )
    with pytest.raises(EvidenceError, match="does not match the recorded"):
        verify_training_report(
            read_json(tmp_path / "outputs" / "training-report.json"), root=tmp_path, labels=LABELS
        )


def test_logit_reference_shape_is_verified(tmp_path) -> None:
    reference = save_logits(
        logits_for(LABELS, 1.0, seed=9),
        tmp_path,
        "outputs/logits/probe.npy",
    )
    reference["shape"] = [1, 4]

    with pytest.raises(EvidenceError, match="shape .* does not match the recorded"):
        load_logits(reference, root=tmp_path)


@pytest.mark.parametrize("replacement", ["/tmp/probe.npy", "../probe.npy"])
def test_logit_reference_must_stay_under_its_root(tmp_path, replacement) -> None:
    reference = save_logits(
        logits_for(LABELS, 1.0, seed=10),
        tmp_path,
        "outputs/logits/probe.npy",
    )
    reference["path"] = replacement

    with pytest.raises(EvidenceError, match="repo-relative"):
        load_logits(reference, root=tmp_path)


def test_saving_logits_cannot_escape_the_evidence_root(tmp_path) -> None:
    with pytest.raises(EvidenceError, match="repo-relative"):
        save_logits(logits_for(LABELS, 1.0, seed=11), tmp_path, "../probe.npy")


def test_saved_logit_path_requires_the_npy_suffix(tmp_path) -> None:
    with pytest.raises(EvidenceError, match="must end in .npy"):
        save_logits(logits_for(LABELS, 1.0, seed=12), tmp_path, "outputs/logits/probe")


def test_modified_adapter_file_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    (tmp_path / "adapters" / "epoch-2" / "adapter_model.safetensors").write_text("tampered")
    with pytest.raises(EvidenceError, match="adapter files changed"):
        verify_training_report(
            read_json(tmp_path / "outputs" / "training-report.json"), root=tmp_path, labels=LABELS
        )


def test_selected_adapter_model_card_is_mutable_but_payload_is_not(tmp_path) -> None:
    make_training_run(tmp_path)
    report = read_json(tmp_path / "outputs" / "training-report.json")
    selected = tmp_path / "adapters" / "selected"
    (selected / "README.md").write_text("a better model card", encoding="utf-8")

    assert verify_training_report(report, root=tmp_path, labels=LABELS)["epoch"] == 2

    (selected / "unexpected.bin").write_bytes(b"not recorded")
    with pytest.raises(EvidenceError, match="selected epoch checkpoint"):
        verify_training_report(report, root=tmp_path, labels=LABELS)


def test_reports_only_verification_does_not_require_adapter_directories(tmp_path) -> None:
    make_training_run(tmp_path)
    for directory in ("epoch-1", "epoch-2", "selected"):
        for item in (tmp_path / "adapters" / directory).iterdir():
            item.unlink()
        (tmp_path / "adapters" / directory).rmdir()

    selected = verify_training_report(
        read_json(tmp_path / "outputs" / "training-report.json"),
        root=tmp_path,
        labels=LABELS,
        verify_adapters=False,
    )
    assert selected["epoch"] == 2


def test_reports_only_rejects_edited_selected_adapter_hashes(tmp_path) -> None:
    make_training_run(tmp_path)
    path = tmp_path / "outputs" / "training-report.json"
    report = read_json(path)
    report["selection"]["selected_adapter_hashes"]["combined_sha256"] = "0" * 64

    with pytest.raises(EvidenceError, match="selected adapter hashes"):
        verify_training_report(
            report,
            root=tmp_path,
            labels=LABELS,
            verify_adapters=False,
        )


def test_report_claiming_the_wrong_epoch_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    path = tmp_path / "outputs" / "training-report.json"
    report = read_json(path)
    report["selection"]["selected_epoch"] = 1
    write_json(report, path)
    with pytest.raises(EvidenceError, match="the rule"):
        verify_training_report(read_json(path), root=tmp_path, labels=LABELS)


def test_mismatched_validation_labels_are_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    with pytest.raises(EvidenceError, match="labels no longer match"):
        verify_training_report(
            read_json(tmp_path / "outputs" / "training-report.json"),
            root=tmp_path,
            labels=list(reversed(LABELS)),
        )


def test_adapter_changed_after_freezing_blocks_the_test_run(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    (tmp_path / "adapters" / "selected" / "adapter_model.safetensors").write_text("swapped")
    with pytest.raises(EvidenceError, match="changed after selection was frozen"):
        require_frozen_selection(root=tmp_path)


def test_final_test_requires_the_explicit_confirmation(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    with pytest.raises(EvidenceError, match="explicit confirmation"):
        run_final_test(default_config(), confirmation="yes", root=tmp_path)


def test_validation_only_experiment_cannot_access_publisher_test(tmp_path) -> None:
    config = replace(default_config(), test_evaluations_allowed=0)
    config.validate()
    with pytest.raises(EvidenceError, match="validation-only"):
        run_final_test(config, confirmation=CONFIRMATION, root=tmp_path)


def test_boolean_test_budget_cannot_access_publisher_test(tmp_path) -> None:
    config = replace(default_config(), test_evaluations_allowed=True)
    with pytest.raises(ValueError, match="JSON booleans are not a test budget"):
        run_final_test(config, confirmation=CONFIRMATION, root=tmp_path)


def test_second_final_test_run_is_refused(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    with pytest.raises(EvidenceError, match="exactly one test evaluation"):
        run_final_test(default_config(), confirmation=CONFIRMATION, root=tmp_path)


def test_final_report_verifies_against_its_own_logits(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    verified = verify_final_report(
        root=tmp_path, labels=LABELS, validation_labels=LABELS
    )
    assert verified["verified"] is True
    assert verified["macro_f1_delta"] > 0


def test_preloaded_test_split_still_checks_ordered_row_ids(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    test = Split(
        "test",
        tuple(
            Example(
                row_id=f"test-{index}",
                text=f"article {index}",
                label=label,
                source_index=index,
            )
            for index, label in enumerate(LABELS)
        ),
    )
    path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(path)
    report["test_row_ids_sha256"] = test.id_sha256()
    write_json(report, path)

    assert verify_final_report(
        root=tmp_path, validation_labels=LABELS, test_split=test
    )["verified"] is True

    reordered = Split("test", tuple(reversed(test.examples)))
    with pytest.raises(EvidenceError, match="row IDs"):
        verify_final_report(
            root=tmp_path, validation_labels=LABELS, test_split=reordered
        )


def test_edited_final_test_number_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(path)
    report["systems"]["tuned"]["metrics_before_temperature"]["macro_f1"] = 0.999
    write_json(report, path)
    with pytest.raises(EvidenceError, match="tuned.before.macro_f1"):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


def test_final_report_temperature_must_match_the_validation_gate(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(path)
    report["systems"]["tuned"]["validation_fitted_temperature"] = 2.0
    write_json(report, path)

    with pytest.raises(EvidenceError, match="temperature fitted on validation"):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("model",), "other/model", "model"),
        (("model_revision",), "0" * 40, "model_revision"),
        (("selected_epoch",), 1, "selected_epoch"),
        (("selected_adapter_hashes", "total_bytes"), 0, "selected_adapter_hashes"),
        (("config", "training", "epochs"), 99, "config"),
        (("rows",), len(LABELS) - 1, "row count"),
    ],
)
def test_final_report_provenance_must_match_frozen_evidence(
    tmp_path, path, replacement, message
) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    report_path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(report_path)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    write_json(report, report_path)

    with pytest.raises(EvidenceError, match=message):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


def test_frozen_temperature_is_refitted_from_validation_logits(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    path = tmp_path / "outputs" / "frozen-selection.json"
    frozen = read_json(path)
    frozen["validation"]["base"]["temperature"] = 2.0
    write_json(frozen, path)

    with pytest.raises(EvidenceError, match=r"frozen\.base\.temperature"):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


def test_edited_calibration_bin_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(path)
    bins = report["systems"]["tuned"]["metrics_after_temperature"]["calibration"]["bins"]
    populated = next(item for item in bins if item["average_confidence"] is not None)
    populated["average_confidence"] += 0.1
    write_json(report, path)

    with pytest.raises(EvidenceError, match=r"tuned\.after\.calibration\.bins"):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


def test_inflated_delta_is_rejected(tmp_path) -> None:
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(path)
    report["delta"]["macro_f1"] = 0.5
    write_json(report, path)
    with pytest.raises(EvidenceError, match="delta does not match"):
        verify_final_report(root=tmp_path, labels=LABELS, validation_labels=LABELS)


def test_a_negative_result_still_verifies(tmp_path) -> None:
    """If QLoRA loses, the pipeline must report it rather than reject it."""
    make_training_run(tmp_path)
    build_frozen_selection(root=tmp_path, labels=LABELS)
    make_final_report(tmp_path, tuned_margin=-0.5)
    verified = verify_final_report(
        root=tmp_path, labels=LABELS, validation_labels=LABELS
    )
    assert verified["macro_f1_delta"] < 0
    assert json.loads(Path(tmp_path / "outputs" / "final-test-report.json").read_text())[
        "test_evaluations_run"
    ] == 1
