from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from loraforge.intervals import (
    bootstrap_delta,
    macro_f1,
    mcnemar,
    paired_disagreement,
)


LABELS = np.array([index % 4 for index in range(400)])


def predictions_with_accuracy(labels: np.ndarray, correct: int, seed: int) -> np.ndarray:
    """Build predictions that are right on exactly `correct` rows."""
    generator = np.random.default_rng(seed)
    output = labels.copy()
    wrong = generator.choice(len(labels), len(labels) - correct, replace=False)
    output[wrong] = (labels[wrong] + 1) % 4
    return output


def test_fast_macro_f1_matches_sklearn() -> None:
    generator = np.random.default_rng(5)
    for seed in range(5):
        predictions = generator.integers(0, 4, len(LABELS))
        assert macro_f1(LABELS, predictions) == pytest.approx(
            f1_score(LABELS, predictions, labels=range(4), average="macro", zero_division=0)
        )


def test_macro_f1_handles_a_class_that_is_never_predicted() -> None:
    predictions = np.where(LABELS == 3, 0, LABELS)  # class 3 never predicted
    assert macro_f1(LABELS, predictions) == pytest.approx(
        f1_score(LABELS, predictions, labels=range(4), average="macro", zero_division=0)
    )


def test_identical_systems_produce_an_interval_containing_zero() -> None:
    same = predictions_with_accuracy(LABELS, 300, seed=1)
    result = bootstrap_delta(LABELS, same, same, resamples=200, seed=73)
    assert result["delta"]["delta"] == pytest.approx(0.0)
    assert result["delta"]["ci_lower"] <= 0.0 <= result["delta"]["ci_upper"]
    assert result["resamples_without_improvement"] == 200


def test_a_clearly_better_system_produces_an_interval_above_zero() -> None:
    worse = predictions_with_accuracy(LABELS, 200, seed=2)
    better = predictions_with_accuracy(LABELS, 380, seed=3)
    result = bootstrap_delta(LABELS, worse, better, resamples=300, seed=73)
    assert result["delta"]["ci_lower"] > 0
    assert result["resamples_without_improvement"] == 0
    for name in ("base", "tuned", "delta"):
        block = result[name]
        point = block.get("macro_f1", block.get("delta"))
        assert block["ci_lower"] <= point <= block["ci_upper"]
        assert block["ci_width"] == pytest.approx(block["ci_upper"] - block["ci_lower"])


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    worse = predictions_with_accuracy(LABELS, 250, seed=4)
    better = predictions_with_accuracy(LABELS, 350, seed=5)
    first = bootstrap_delta(LABELS, worse, better, resamples=100, seed=73)
    second = bootstrap_delta(LABELS, worse, better, resamples=100, seed=73)
    different = bootstrap_delta(LABELS, worse, better, resamples=100, seed=74)
    assert first == second
    assert first["delta"]["ci_lower"] != different["delta"]["ci_lower"]


def test_bootstrap_rejects_bad_settings() -> None:
    same = predictions_with_accuracy(LABELS, 300, seed=6)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_delta(LABELS, same, same, alpha=0.0, resamples=10)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_delta(LABELS, same, same, resamples=0)
    with pytest.raises(ValueError, match="same nonzero length"):
        bootstrap_delta(LABELS, same[:10], same, resamples=10)
    with pytest.raises(ValueError, match="nonzero"):
        bootstrap_delta(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
    with pytest.raises(ValueError, match="class IDs"):
        bootstrap_delta(LABELS, same, np.full_like(LABELS, 4), resamples=10)


def test_paired_counts_are_exact() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1])
    base = np.array([0, 1, 0, 0, 1, 1])  # correct on rows 0, 1, 5
    tuned = np.array([0, 2, 2, 3, 0, 1])  # correct on rows 0, 2, 3, 4, 5
    result = paired_disagreement(labels, base, tuned)
    assert result["tuned_fixed_base_error"] == 3  # rows 2, 3, 4
    assert result["tuned_broke_base_success"] == 1  # row 1
    assert result["both_correct"] == 2  # rows 0, 5
    assert result["both_wrong"] == 0
    assert (
        result["both_correct"]
        + result["both_wrong"]
        + result["tuned_fixed_base_error"]
        + result["tuned_broke_base_success"]
        == result["rows"]
    )


def test_mcnemar_finds_no_effect_when_fixes_and_breaks_are_equal() -> None:
    result = mcnemar(50, 50)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["chi_square"] == pytest.approx(0.01)


def test_mcnemar_is_decisive_for_a_lopsided_split() -> None:
    result = mcnemar(1577, 129)  # the measured LoRAForge test counts
    assert result["discordant_pairs"] == 1706
    assert result["log10_p_value"] < -250
    assert result["chi_square"] > 1000
    assert result["p_value"] is not None and result["p_value"] > 0
    assert result["p_value_scientific"].endswith("e-317")


def test_mcnemar_handles_perfect_agreement_and_rejects_negatives() -> None:
    agreed = mcnemar(0, 0)
    assert agreed["discordant_pairs"] == 0
    assert agreed["chi_square"] is None
    assert agreed["log10_p_value"] is None
    with pytest.raises(ValueError, match="negative"):
        mcnemar(-1, 3)
    with pytest.raises(ValueError, match="integers"):
        mcnemar(1.5, 3)


def test_small_lopsided_split_has_a_representable_p_value() -> None:
    result = mcnemar(10, 0)
    assert result["p_value"] == pytest.approx(2 / 2**10)


def make_final_test_report(tmp_path, base, tuned, labels):
    """Minimal stand-in for the artifact that build_intervals reads."""
    from loraforge.provenance import save_logits, sha256_labels, write_json

    def one_hot(predictions):
        logits = np.full((len(predictions), 4), -3.0, dtype=np.float32)
        logits[np.arange(len(predictions)), predictions] = 3.0
        return logits

    systems = {}
    for name, predictions in (("base", base), ("tuned", tuned)):
        logits = one_hot(predictions)
        systems[name] = {
            "logits": save_logits(logits, tmp_path, f"outputs/logits/{name}-test.npy"),
            "prediction_sha256": sha256_labels(predictions.tolist()),
        }
    write_json(
        {
            "config": {"data": {}},
            "test_evaluated": True,
            "test_evaluations_run": 1,
            "test_label_sha256": sha256_labels(list(labels)),
            "systems": systems,
        },
        tmp_path / "outputs" / "final-test-report.json",
    )


def test_intervals_round_trip_and_spend_no_test_budget(tmp_path) -> None:
    from loraforge.intervals import build_intervals, verify_intervals

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=7)
    tuned = predictions_with_accuracy(labels, 370, seed=8)
    make_final_test_report(tmp_path, base, tuned, labels)

    built = build_intervals(root=tmp_path, labels=list(labels))
    assert built["new_test_evaluations"] == 0
    assert built["bootstrap"]["delta"]["ci_lower"] > 0
    assert verify_intervals(root=tmp_path, labels=list(labels))["verified"] is True


def test_edited_confidence_interval_is_rejected(tmp_path) -> None:
    from loraforge.intervals import INTERVALS_REPORT, build_intervals, verify_intervals
    from loraforge.provenance import EvidenceError, read_json, write_json

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=9)
    tuned = predictions_with_accuracy(labels, 370, seed=10)
    make_final_test_report(tmp_path, base, tuned, labels)
    build_intervals(root=tmp_path, labels=list(labels))

    stored = read_json(tmp_path / INTERVALS_REPORT)
    stored["bootstrap"]["delta"]["ci_lower"] = 0.99
    write_json(stored, tmp_path / INTERVALS_REPORT)
    with pytest.raises(EvidenceError, match="recomputes to"):
        verify_intervals(root=tmp_path, labels=list(labels))


def test_edited_paired_counts_are_rejected(tmp_path) -> None:
    from loraforge.intervals import INTERVALS_REPORT, build_intervals, verify_intervals
    from loraforge.provenance import EvidenceError, read_json, write_json

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=11)
    tuned = predictions_with_accuracy(labels, 370, seed=12)
    make_final_test_report(tmp_path, base, tuned, labels)
    build_intervals(root=tmp_path, labels=list(labels))

    stored = read_json(tmp_path / INTERVALS_REPORT)
    stored["paired"]["tuned_broke_base_success"] = 0
    write_json(stored, tmp_path / INTERVALS_REPORT)
    with pytest.raises(EvidenceError, match="paired.tuned_broke_base_success"):
        verify_intervals(root=tmp_path, labels=list(labels))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("new_test_evaluations",), 99, "new_test_evaluations"),
        (("scope",), "training variance measured", "scope"),
        (("bootstrap", "resamples_without_improvement"), 99, "resamples_without_improvement"),
        (("bootstrap", "settings", "resamples"), 2_000.0, "must be an integer"),
        (("test_label_sha256",), "0" * 64, "test_label_sha256"),
    ],
)
def test_every_headline_interval_claim_is_verified(
    tmp_path, path, replacement, message
) -> None:
    from loraforge.intervals import INTERVALS_REPORT, build_intervals, verify_intervals
    from loraforge.provenance import EvidenceError, read_json, write_json

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=15)
    tuned = predictions_with_accuracy(labels, 370, seed=16)
    make_final_test_report(tmp_path, base, tuned, labels)
    build_intervals(root=tmp_path, labels=list(labels))
    stored = read_json(tmp_path / INTERVALS_REPORT)
    target = stored
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    write_json(stored, tmp_path / INTERVALS_REPORT)

    with pytest.raises(EvidenceError, match=message):
        verify_intervals(root=tmp_path, labels=list(labels))


def test_intervals_are_bound_to_the_exact_final_report(tmp_path) -> None:
    from loraforge.intervals import build_intervals, verify_intervals
    from loraforge.provenance import EvidenceError, read_json, write_json

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=17)
    tuned = predictions_with_accuracy(labels, 370, seed=18)
    make_final_test_report(tmp_path, base, tuned, labels)
    build_intervals(root=tmp_path, labels=list(labels))

    report_path = tmp_path / "outputs" / "final-test-report.json"
    report = read_json(report_path)
    report["unrelated_edit"] = True
    write_json(report, report_path)

    with pytest.raises(EvidenceError, match="not bound"):
        verify_intervals(root=tmp_path, labels=list(labels))


def test_swapped_logits_file_is_rejected_by_prediction_hash(tmp_path) -> None:
    from loraforge.intervals import build_intervals
    from loraforge.provenance import EvidenceError

    labels = LABELS
    base = predictions_with_accuracy(labels, 220, seed=13)
    tuned = predictions_with_accuracy(labels, 370, seed=14)
    make_final_test_report(tmp_path, base, tuned, labels)
    np.save(tmp_path / "outputs" / "logits" / "tuned-test.npy",
            np.zeros((len(labels), 4), dtype=np.float32))
    with pytest.raises(EvidenceError):
        build_intervals(root=tmp_path, labels=list(labels))
