"""Uncertainty for the single test run, recomputed from its stored logits.

Nothing here touches a GPU or spends test budget: it resamples and re-analyzes
the one evaluation that already happened. That bounds *sampling* uncertainty —
how the measured gap would move if a different sample of articles had been drawn
— and says nothing about training variance, which needs more training runs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .data import CLASS_NAMES
from .metrics import softmax
from .provenance import (
    EvidenceError,
    load_logits,
    sha256_file,
    read_json,
    sha256_labels,
    utc_now,
    write_json,
)

INTERVALS_REPORT = Path("outputs/test-intervals.json")
DEFAULT_RESAMPLES = 2_000
DEFAULT_SEED = 73
TOLERANCE = 1e-9


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Macro-F1 by confusion counts; matches sklearn with zero_division=0."""
    size = len(CLASS_NAMES)
    matrix = np.bincount(
        np.asarray(labels) * size + np.asarray(predictions), minlength=size * size
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


def bootstrap_delta(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    tuned_predictions: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap over rows, resampling both systems together.

    Paired resampling matters: the two systems are scored on the same articles,
    so sampling them independently would overstate the uncertainty of the gap.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    labels = np.asarray(labels)
    if not (len(labels) == len(base_predictions) == len(tuned_predictions)):
        raise ValueError("labels and both prediction arrays must be the same length")

    generator = np.random.default_rng(seed)
    rows = len(labels)
    draws = {"base": [], "tuned": [], "delta": []}
    for _ in range(resamples):
        sample = generator.integers(0, rows, rows)
        sampled_labels = labels[sample]
        base = macro_f1(sampled_labels, base_predictions[sample])
        tuned = macro_f1(sampled_labels, tuned_predictions[sample])
        draws["base"].append(base)
        draws["tuned"].append(tuned)
        draws["delta"].append(tuned - base)

    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    point = {
        "base": macro_f1(labels, base_predictions),
        "tuned": macro_f1(labels, tuned_predictions),
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
    labels = np.asarray(labels)
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
    if fixed < 0 or broken < 0:
        raise ValueError("counts cannot be negative")
    discordant = fixed + broken
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "chi_square": None,
            "log10_p_value": None,
            "note": "the systems never disagreed on correctness",
        }
    chi_square = (abs(fixed - broken) - 1) ** 2 / discordant if discordant else 0.0
    log10_p = _log10_two_sided_binomial_tail(discordant, min(fixed, broken))
    representable = log10_p > -300
    return {
        "discordant_pairs": discordant,
        "chi_square": float(chi_square),
        "log10_p_value": log10_p,
        # No p-value is zero. When the exact tail underflows float64, report the
        # bound instead of a number that would be false if quoted.
        "p_value": float(10**log10_p) if representable else None,
        "p_value_upper_bound": None if representable else 1e-300,
    }


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
    """Write ``outputs/test-intervals.json`` from the existing final test report."""
    root = Path(root)
    report = read_json(root / "outputs/final-test-report.json")
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
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source_report": "outputs/final-test-report.json",
        "source_report_sha256": sha256_file(root / "outputs/final-test-report.json"),
        "test_label_sha256": report["test_label_sha256"],
        "new_test_evaluations": 0,
        "scope": (
            "sampling uncertainty of the single stored test run; training/seed "
            "variance is NOT measured here and needs additional training runs"
        ),
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
    settings = stored["bootstrap"]["settings"]
    report = read_json(root / "outputs/final-test-report.json")
    if labels is None:
        from .config import DataConfig
        from .data import load_dataset

        bundle = load_dataset(allow_test=True, config=DataConfig(**report["config"]["data"]))
        labels = bundle.require_test().labels
    label_array = np.asarray(labels)
    base = _predictions(report, "base", root)
    tuned = _predictions(report, "tuned", root)

    recomputed = bootstrap_delta(
        label_array,
        base,
        tuned,
        resamples=settings["resamples"],
        seed=settings["seed"],
        alpha=settings["alpha"],
    )
    for name in ("base", "tuned", "delta"):
        for key, value in stored["bootstrap"][name].items():
            if abs(float(value) - float(recomputed[name][key])) > TOLERANCE:
                raise EvidenceError(
                    f"bootstrap {name}.{key} recorded as {value} but recomputes to "
                    f"{recomputed[name][key]}"
                )
    # Everything else the file asserts is a claim too. Checking only the metric
    # blocks once let a forged "0 of 2,000 resamples" and a forged
    # new_test_evaluations pass verification untouched.
    for field in ("resamples_without_improvement", "settings"):
        if stored["bootstrap"][field] != recomputed[field]:
            raise EvidenceError(
                f"bootstrap {field} recorded as {stored['bootstrap'][field]} but "
                f"recomputes to {recomputed[field]}"
            )
    if stored["paired"] != paired_disagreement(label_array, base, tuned):
        raise EvidenceError("paired disagreement counts do not match their own logits")
    if stored.get("new_test_evaluations") != 0:
        raise EvidenceError(
            "this analysis resamples a stored run and cannot have spent test budget; "
            f"new_test_evaluations reads {stored.get('new_test_evaluations')!r}"
        )
    recorded_source = stored.get("source_report_sha256")
    if recorded_source is not None:
        actual_source = sha256_file(root / stored["source_report"])
        if actual_source != recorded_source:
            raise EvidenceError(
                f"{stored['source_report']} changed since these intervals were computed"
            )
    return {
        "verified": True,
        "delta": stored["bootstrap"]["delta"]["delta"],
        "ci": [stored["bootstrap"]["delta"]["ci_lower"], stored["bootstrap"]["delta"]["ci_upper"]],
    }
