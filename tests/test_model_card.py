from __future__ import annotations

import json

import pytest

import loraforge.selection as selection_module
from loraforge.config import default_config
from loraforge.model_card import PLACEHOLDER, build_model_card, write_model_card
from loraforge.provenance import (
    EvidenceError,
    sha256_directory,
    sha256_file,
    write_json,
)


@pytest.fixture(autouse=True)
def isolate_card_formatting_from_dataset_loading(monkeypatch):
    """Formatting tests use minimal reports; verifier behavior has focused tests below."""
    monkeypatch.setattr(
        selection_module,
        "verify_training_report",
        lambda report, *, root, verify_adapters: report["epochs"][-1],
    )


def make_run(tmp_path, *, macro_f1=0.9360, rank=4, with_test=False):
    config = default_config().to_dict()
    config["lora"]["rank"] = rank
    config["lora"]["alpha"] = rank * 2
    adapter = tmp_path / "adapters" / "selected"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text('{"r": 4}')
    (adapter / "adapter_model.safetensors").write_text("weights")
    adapter_hashes = sha256_directory(adapter)

    write_json(
        {
            "model": config["model_name"],
            "model_revision": config["model_revision"],
            "config": config,
            "environment": {"gpu_name": "Tesla T4", "packages": {}},
            "train_rows": 8_000,
            "validation_rows": 2_000,
            "base_validation_metrics": {"accuracy": 0.7475, "macro_f1": 0.7299},
            "epochs": [
                {
                    "epoch": 1,
                    "validation": {
                        "accuracy": macro_f1 - 0.02,
                        "macro_f1": macro_f1 - 0.02,
                    },
                },
                {"epoch": 2, "validation": {"accuracy": macro_f1, "macro_f1": macro_f1}},
            ],
            "selection": {
                "selected_epoch": 2,
                "selected_adapter_dir": "adapters/selected",
                "selected_adapter_hashes": adapter_hashes,
            },
            "parameters": {
                "trainable_parameters": 10_485_760,
                "total_parameters": 7_258_509_312,
                "trainable_percent": 100 * 10_485_760 / 7_258_509_312,
            },
            "wall_time_seconds": 14_487.9,
            "peak_cuda_memory_gib": 5.48,
        },
        tmp_path / "outputs" / "training-report.json",
    )
    if with_test:
        final_path = tmp_path / "outputs" / "final-test-report.json"
        write_json(
            {
                "test_evaluated": True,
                "test_evaluations_run": 1,
                "rows": 7_600,
                "model": config["model_name"],
                "model_revision": config["model_revision"],
                "selected_epoch": 2,
                "selected_adapter_hashes": adapter_hashes,
                "config": config,
                "systems": {
                    "base": {"metrics_before_temperature": {"accuracy": 0.7428, "macro_f1": 0.7262}},
                    "tuned": {"metrics_before_temperature": {"accuracy": 0.9333, "macro_f1": 0.9333}},
                },
            },
            final_path,
        )
        write_json(
            {
                "source_report": "outputs/final-test-report.json",
                "source_report_sha256": sha256_file(final_path),
            },
            tmp_path / "outputs" / "test-intervals-v2.json",
        )


def test_a_validation_only_run_refuses_to_quote_held_out_numbers(tmp_path) -> None:
    make_run(tmp_path, with_test=False)
    card = build_model_card(root=tmp_path)
    assert "No held-out test result is reported" in card
    assert "held-out test (" not in card
    assert "0.9333" not in card  # the other run's test score must not leak in


def test_a_run_that_evaluated_test_reports_it(tmp_path) -> None:
    make_run(tmp_path, with_test=True)
    card = build_model_card(root=tmp_path)
    assert "held-out test (7,600 rows)" in card
    assert "0.9333" in card
    assert "evaluated **once**" in card


def test_every_placeholder_is_filled(tmp_path) -> None:
    make_run(tmp_path)
    assert PLACEHOLDER not in build_model_card(root=tmp_path)


def test_the_card_reports_the_run_it_was_generated_from(tmp_path) -> None:
    make_run(tmp_path, macro_f1=0.8123, rank=8)
    card = build_model_card(root=tmp_path)
    assert "0.8123" in card
    assert "rank-8 QLoRA adapter" in card
    assert "10,485,760" in card and "0.1445%" in card


def test_usage_pins_the_recorded_base_and_tokenizer_revision(tmp_path) -> None:
    make_run(tmp_path)
    card = build_model_card(root=tmp_path)
    revision = default_config().model_revision

    assert f"revision = '{revision}'" in card
    assert (
        "AutoTokenizer.from_pretrained("
        "'mistralai/Mistral-7B-Instruct-v0.3', revision=revision)" in card
    )
    assert (
        "'mistralai/Mistral-7B-Instruct-v0.3', revision=revision, "
        "quantization_config=quantization" in card
    )


def test_limitations_do_not_invent_unrecorded_comparisons_or_error_analysis(tmp_path) -> None:
    make_run(tmp_path)
    card = build_model_card(root=tmp_path)
    assert "training variance across seeds has not been measured" in card
    assert "separately measured classical baseline" in card
    assert "no annotated error-analysis artifact" in card
    assert "not a result" in card  # the constrained-decoding caveat
    assert "0.887" not in card
    assert "about a second" not in card
    assert "roughly half" not in card


def test_writing_the_card_never_touches_the_adapter_directory(tmp_path) -> None:
    """Writing into the adapter would invalidate the hash the whole chain rests on."""
    from loraforge.provenance import sha256_directory

    make_run(tmp_path)
    adapter = tmp_path / "adapters" / "selected"
    before = sha256_directory(adapter)["combined_sha256"]

    target = write_model_card(root=tmp_path, repo_url="https://example.invalid/repo")

    assert target == tmp_path / "outputs" / "model-card.md"
    assert "https://example.invalid/repo" in target.read_text()
    assert sha256_directory(adapter)["combined_sha256"] == before


def test_missing_parameter_measurement_is_refused(tmp_path) -> None:
    make_run(tmp_path)
    path = tmp_path / "outputs" / "training-report.json"
    report = json.loads(path.read_text())
    del report["parameters"]
    write_json(report, path)
    with pytest.raises(EvidenceError, match="refusing to invent"):
        build_model_card(root=tmp_path)


def test_unrelated_final_report_is_refused(tmp_path) -> None:
    make_run(tmp_path, with_test=True)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = json.loads(path.read_text())
    report["selected_epoch"] = 1
    write_json(report, path)
    with pytest.raises(EvidenceError, match="does not belong"):
        build_model_card(root=tmp_path)


def test_model_card_runs_training_evidence_verification(tmp_path, monkeypatch) -> None:
    make_run(tmp_path)
    calls = []

    def reject_unverified(report, *, root, verify_adapters):
        calls.append((root, verify_adapters))
        raise EvidenceError("validation metrics are not verified")

    monkeypatch.setattr(selection_module, "verify_training_report", reject_unverified)
    with pytest.raises(EvidenceError, match="validation metrics are not verified"):
        build_model_card(root=tmp_path)
    assert calls == [(tmp_path, False)]


def test_edited_final_metrics_are_not_quoted_without_their_hash_binding(tmp_path) -> None:
    make_run(tmp_path, with_test=True)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = json.loads(path.read_text())
    report["systems"]["tuned"]["metrics_before_temperature"]["macro_f1"] = 0.9999
    write_json(report, path)

    with pytest.raises(EvidenceError, match="not hash-bound"):
        build_model_card(root=tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("test_evaluated", 1), ("test_evaluations_run", True)],
)
def test_model_card_rejects_boolean_integer_protocol_aliases(
    tmp_path, field, replacement
) -> None:
    make_run(tmp_path, with_test=True)
    path = tmp_path / "outputs" / "final-test-report.json"
    report = json.loads(path.read_text())
    report[field] = replacement
    write_json(report, path)

    with pytest.raises(EvidenceError, match=field):
        build_model_card(root=tmp_path)


def test_a_missing_report_is_refused(tmp_path) -> None:
    with pytest.raises(EvidenceError, match="training-report.json"):
        build_model_card(root=tmp_path)
