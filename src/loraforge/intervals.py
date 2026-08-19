"""Uncertainty for the single test run, recomputed from its stored logits.

Nothing here touches a GPU or spends test budget: it resamples and re-analyzes
the one evaluation that already happened. That bounds *sampling* uncertainty —
how the measured gap would move if a different sample of articles had been drawn
— and says nothing about training variance, which needs more training runs.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data import CLASS_NAMES
from .metrics import softmax
from .provenance import (
    EvidenceError,
    load_logits,
    read_json,
    sha256_file,
    sha256_labels,
    utc_now,
    write_json,
)

INTERVALS_REPORT = Path("outputs/test-intervals-v2.json")
DEFAULT_RESAMPLES = 2_000
DEFAULT_SEED = 73
DEFAULT_ALPHA = 0.05
TOLERANCE = 1e-9
SOURCE_REPORT = "outputs/final-test-report.json"
SCOPE = (
    "sampling uncertainty of the single stored test run; training/seed "
    "variance is NOT measured here and needs additional training runs"
)


def _validate_class_vectors(
    labels: np.ndarray, *predictions: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Validate nonempty one-dimensional AG News class vectors."""
    arrays = tuple(np.asarray(values) for values in (labels, *predictions))
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("labels and predictions must be one-dimensional")
    if not arrays[0].size or any(values.size != arrays[0].size for values in arrays[1:]):
        raise ValueError("labels and prediction arrays must have the same nonzero length")
    for values in arrays:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("labels and predictions must contain integer class IDs")
        if np.any((values < 0) | (values >= len(CLASS_NAMES))):
            raise ValueError("labels and predictions must use AG News class IDs 0 through 3")
    return arrays


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    size = len(CLASS_NAMES)
    matrix = np.bincount(
        labels * size + predictions, minlength=size * size
    ).reshape(size, size)
    true_positive = np.diag(matrix).astype(float)
    denominator = 2 * true_positive + (matrix.sum(0) - true_positive) + (
        matrix.sum(1) - true_positive
    )
    per_class = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(per_class.mean())


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Macro-F1 by confusion counts; matches sklearn with zero_division=0."""
    checked_labels, checked_predictions = _validate_class_vectors(labels, predictions)
    return _macro_f1(checked_labels, checked_predictions)


def bootstrap_delta(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    tuned_predictions: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap over rows, resampling both systems together.

    Paired resampling matters: the two systems are scored on the same articles,
    so sampling them independently would overstate the uncertainty of the gap.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    labels, base_predictions, tuned_predictions = _validate_class_vectors(
        labels, base_predictions, tuned_predictions
    )

    generator = np.random.default_rng(seed)
    rows = len(labels)
    draws = {"base": [], "tuned": [], "delta": []}
    for _ in range(resamples):
        sample = generator.integers(0, rows, rows)
        sampled_labels = labels[sample]
        base = _macro_f1(sampled_labels, base_predictions[sample])
        tuned = _macro_f1(sampled_labels, tuned_predictions[sample])
        draws["base"].append(base)
        draws["tuned"].append(tuned)
        draws["delta"].append(tuned - base)

    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    point = {
        "base": _macro_f1(labels, base_predictions),
        "tuned": _macro_f1(labels, tuned_predictions),
    }
    point["delta"] = point["tuned"] - point["base"]
    summary = {}
    for name, values in draws.items():
        low, high = (float(x) for x in np.percentile(values, percentiles))
        summary[name] = {
            "macro_f1" if name != "delta" else "delta": point[name],
            "ci_lower": low,
            "ci_upper": high,
            "ci_width": high - low,
        }
    summary["resamples_without_improvement"] = int(
        sum(1 for value in draws["delta"] if value <= 0)
    )
    summary["settings"] = {
        "resamples": resamples,
        "seed": seed,
        "alpha": alpha,
        "method": "paired percentile bootstrap over test rows",
    }
    return summary


def paired_disagreement(
    labels: np.ndarray, base_predictions: np.ndarray, tuned_predictions: np.ndarray
) -> dict[str, Any]:
    """Row-level comparison: what the adapter fixed versus what it broke."""
    labels, base_predictions, tuned_predictions = _validate_class_vectors(
        labels, base_predictions, tuned_predictions
    )
    base_correct = base_predictions == labels
    tuned_correct = tuned_predictions == labels
    fixed = int((~base_correct & tuned_correct).sum())
    broken = int((base_correct & ~tuned_correct).sum())
    return {
        "rows": int(len(labels)),
        "disagreements": int((base_predictions != tuned_predictions).sum()),
        "tuned_fixed_base_error": fixed,
        "tuned_broke_base_success": broken,
        "both_correct": int((base_correct & tuned_correct).sum()),
        "both_wrong": int((~base_correct & ~tuned_correct).sum()),
        "mcnemar": mcnemar(fixed, broken),
    }


def mcnemar(fixed: int, broken: int) -> dict[str, Any]:
    """Exact two-sided McNemar test on the discordant pairs.

    Under the null the adapter is equally likely to fix or break a row, so the
    discordant pairs are Binomial(n, 0.5). The exact tail is computed in log
    space because it underflows float64 long before it stops being meaningful.
    """
    if (
        isinstance(fixed, bool)
        or isinstance(broken, bool)
        or not isinstance(fixed, (int, np.integer))
        or not isinstance(broken, (int, np.integer))
    ):
        raise ValueError("counts must be integers")
    fixed, broken = int(fixed), int(broken)
    if fixed < 0 or broken < 0:
        raise ValueError("counts cannot be negative")
    discordant = fixed + broken
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "chi_square": None,
            "log10_p_value": None,
            "p_value": None,
            "p_value_scientific": None,
            "note": "the systems never disagreed on correctness",
        }
    chi_square = (abs(fixed - broken) - 1) ** 2 / discordant if discordant else 0.0
    log10_p = _log10_two_sided_binomial_tail(discordant, min(fixed, broken))
    p_value = float(10.0**log10_p)
    return {
        "discordant_pairs": discordant,
        "chi_square": float(chi_square),
        "log10_p_value": log10_p,
        "p_value": p_value if p_value > 0 else None,
        "p_value_scientific": _scientific_probability(log10_p),
    }


def _scientific_probability(log10_p: float) -> str:
    """Represent arbitrarily small probabilities without float underflow."""
    exponent = math.floor(log10_p)
    mantissa = 10.0 ** (log10_p - exponent)
    # Keep enough precision to be useful while avoiding platform-level lgamma
    # noise in the final few digits of this human-readable representation.
    return f"{mantissa:.12g}e{exponent}"


def _log10_two_sided_binomial_tail(trials: int, successes: int) -> float:
    """log10 of the two-sided Binomial(n, 0.5) tail, capped at log10(1)=0."""
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        - trials * math.log(2.0)
        for k in range(successes + 1)
    ]
    largest = max(log_terms)
    log_tail = largest + math.log(sum(math.exp(term - largest) for term in log_terms))
    two_sided = log_tail + math.log(2.0)
    return float(min(0.0, two_sided / math.log(10.0)))


def _predictions(report: dict[str, Any], name: str, root: Path) -> np.ndarray:
    system = report["systems"][name]
    logits = load_logits(system["logits"], root=root)
    predictions = softmax(logits).argmax(1)
    if sha256_labels(predictions.tolist()) != system["prediction_sha256"]:
        raise EvidenceError(f"{name} predictions do not match their recorded hash")
    return predictions


def build_intervals(
    *,
    root: Path = Path("."),
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Write schema-v2 intervals from the existing frozen final test report."""
    root = Path(root)
    report = read_json(root / "outputs/final-test-report.json")
    if resamples != DEFAULT_RESAMPLES or seed != DEFAULT_SEED:
        raise EvidenceError(
            "published intervals use the frozen 2,000-resample, seed-73 protocol"
        )
    if report.get("test_evaluated") is not True or report.get("test_evaluations_run") != 1:
        raise EvidenceError("intervals require the single frozen final-test evaluation")
    if labels is None:
        from .config import DataConfig
        from .data import load_dataset

        bundle = load_dataset(allow_test=True, config=DataConfig(**report["config"]["data"]))
        labels = bundle.require_test().labels
    if sha256_labels(labels) != report["test_label_sha256"]:
        raise EvidenceError("test labels no longer match the final report")

    label_array = np.asarray(labels)
    base = _predictions(report, "base", root)
    tuned = _predictions(report, "tuned", root)
    intervals = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "source_report": SOURCE_REPORT,
        "source_report_sha256": sha256_file(root / SOURCE_REPORT),
        "test_label_sha256": report["test_label_sha256"],
        "new_test_evaluations": 0,
        "scope": SCOPE,
        "bootstrap": bootstrap_delta(
            label_array, base, tuned, resamples=resamples, seed=seed
        ),
        "paired": paired_disagreement(label_array, base, tuned),
    }
    write_json(intervals, root / INTERVALS_REPORT)
    return intervals


def verify_intervals(*, root: Path = Path("."), labels: list[int] | None = None) -> dict[str, Any]:
    """Recompute the stored intervals and reject an edited report."""
    root = Path(root)
    stored = read_json(root / INTERVALS_REPORT)
    if stored.get("source_report") != SOURCE_REPORT:
        raise EvidenceError("intervals source_report must be the frozen final-test report")
    source_path = root / SOURCE_REPORT
    source_hash = sha256_file(source_path)
    if stored.get("source_report_sha256") != source_hash:
        raise EvidenceError("intervals report is not bound to the current final-test report")
    report = read_json(source_path)
    if report.get("test_evaluated") is not True or report.get("test_evaluations_run") != 1:
        raise EvidenceError("intervals require the single frozen final-test evaluation")
    if labels is None:
        from .config import DataConfig
        from .data import load_dataset

        bundle = load_dataset(allow_test=True, config=DataConfig(**report["config"]["data"]))
        labels = bundle.require_test().labels
    label_hash = sha256_labels(labels)
    if label_hash != report.get("test_label_sha256"):
        raise EvidenceError("test labels no longer match the final report")
    label_array = np.asarray(labels)
    base = _predictions(report, "base", root)
    tuned = _predictions(report, "tuned", root)

    recomputed_bootstrap = bootstrap_delta(
        label_array,
        base,
        tuned,
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_SEED,
        alpha=DEFAULT_ALPHA,
    )
    expected = {
        "schema_version": 2,
        "source_report": SOURCE_REPORT,
        "source_report_sha256": source_hash,
        "test_label_sha256": label_hash,
        "new_test_evaluations": 0,
        "scope": SCOPE,
        "bootstrap": recomputed_bootstrap,
        "paired": paired_disagreement(label_array, base, tuned),
    }
    actual = {key: value for key, value in stored.items() if key != "created_at_utc"}
    created_at = stored.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise EvidenceError("intervals created_at_utc must be a timestamp string")
    try:
        datetime.fromisoformat(created_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise EvidenceError("intervals created_at_utc is not a valid UTC timestamp") from error
    _assert_tree("intervals", actual, expected)
    return {
        "verified": True,
        "delta": stored["bootstrap"]["delta"]["delta"],
        "ci": [stored["bootstrap"]["delta"]["ci_lower"], stored["bootstrap"]["delta"]["ci_upper"]],
    }


def _assert_tree(name: str, actual: Any, expected: Any) -> None:
    """Compare a complete deterministic JSON tree and reject omitted fields."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            actual_keys = set(actual) if isinstance(actual, dict) else set()
            raise EvidenceError(
                f"{name} fields differ: missing={sorted(set(expected) - actual_keys)}, "
                f"unexpected={sorted(actual_keys - set(expected))}"
            )
        for key, value in expected.items():
            _assert_tree(f"{name}.{key}", actual[key], value)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise EvidenceError(f"{name} length differs from recomputed evidence")
        for index, value in enumerate(expected):
            _assert_tree(f"{name}[{index}]", actual[index], value)
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise EvidenceError(f"{name} must be a finite number")
        try:
            number = float(actual)
        except (TypeError, ValueError) as error:
            raise EvidenceError(f"{name} must be a finite number") from error
        if not math.isfinite(number) or not math.isclose(
            number, expected, rel_tol=TOLERANCE, abs_tol=0.0
        ):
            raise EvidenceError(f"{name} recorded as {actual} but recomputes to {expected}")
        return
    if isinstance(expected, int) and type(actual) is not int:
        raise EvidenceError(f"{name} must be an integer")
    if actual != expected:
        raise EvidenceError(f"{name} recorded as {actual!r} but recomputes to {expected!r}")
