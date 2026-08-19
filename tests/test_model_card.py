from __future__ import annotations

import json

import pytest

from loraforge.config import default_config
from loraforge.model_card import PLACEHOLDER, build_model_card, write_model_card
from loraforge.provenance import write_json


def make_run(tmp_path, *, macro_f1=0.9360, rank=4, with_test=False):
    config = default_config().to_dict()
    config["lora"]["rank"] = rank
    config["lora"]["alpha"] = rank * 2
    adapter = tmp_path / "adapters" / "selected"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_text("weights")

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
                {"epoch": 1, "validation": {"accuracy": 0.9182, "macro_f1": 0.9182}},
                {"epoch": 2, "validation": {"accuracy": macro_f1, "macro_f1": macro_f1}},
            ],
            "selection": {"selected_epoch": 2, "selected_adapter_dir": "adapters/selected"},
            "parameters": {
                "trainable_parameters": 10_485_760,
                "total_parameters": 7_258_509_312,
                "trainable_percent": 0.1445,
            },
            "wall_time_seconds": 14_487.9,
            "peak_cuda_memory_gib": 5.48,
        },
        tmp_path / "outputs" / "training-report.json",
    )
    if with_test:
        write_json(
            {
                "rows": 7_600,
                "systems": {
                    "base": {"metrics_before_temperature": {"accuracy": 0.7428, "macro_f1": 0.7262}},
                    "tuned": {"metrics_before_temperature": {"accuracy": 0.9333, "macro_f1": 0.9333}},
                },
            },
            tmp_path / "outputs" / "final-test-report.json",
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


def test_limitations_name_the_cheap_baseline_and_the_missing_seed_variance(tmp_path) -> None:
    make_run(tmp_path)
    card = build_model_card(root=tmp_path)
    assert "0.887" in card  # the TF-IDF comparison a reader deserves to see
    assert "training variance across seeds has not been measured" in card
    assert "not a result" in card  # the constrained-decoding caveat


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


def test_an_unmeasured_parameter_count_is_stated_not_invented(tmp_path) -> None:
    """A missing measurement once rendered as "Trainable parameters: **0**"."""
    make_run(tmp_path)
    report_path = tmp_path / "outputs" / "training-report.json"
    report = json.loads(report_path.read_text())
    del report["parameters"]
    report_path.write_text(json.dumps(report))

    card = build_model_card(root=tmp_path)
    assert "Trainable parameters: **not recorded by this run" in card
    assert "Trainable parameters: **0**" not in card


def test_a_missing_report_is_refused(tmp_path) -> None:
    from loraforge.provenance import EvidenceError

    with pytest.raises(EvidenceError, match="training-report.json"):
        build_model_card(root=tmp_path)
