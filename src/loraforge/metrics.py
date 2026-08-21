"""Classification and calibration metrics with fixed four-class semantics."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import CLASS_NAMES


def _class_ids(values: Any, name: str, *, allow_invalid: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer class IDs")
    if not allow_invalid and np.any((array < 0) | (array >= len(CLASS_NAMES))):
        raise ValueError(f"{name} must use AG News class IDs 0 through 3")
    return array.astype(np.int64, copy=False)


def _logits_and_labels(
    class_logits: np.ndarray, labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    # Preserve the recorded dtype: dividing float32 logits before softmax has
    # slightly different rounding from first widening them to float64, and the
    # strict evidence verifier intentionally detects that numerical drift.
    logits = np.asarray(class_logits)
    label_array = _class_ids(labels, "labels")
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError("class logits must have shape [rows, 4]")
    if len(logits) != len(label_array):
        raise ValueError("class logits and labels must have the same nonzero length")
    if (
        not np.issubdtype(logits.dtype, np.number)
        or np.issubdtype(logits.dtype, np.bool_)
        or np.issubdtype(logits.dtype, np.complexfloating)
    ):
        raise ValueError("class logits must contain real numeric values")
    if not np.isfinite(logits).all():
        raise ValueError("class logits must contain only finite values")
    return logits, label_array


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_predictions(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    label_array = _class_ids(labels, "labels")
    prediction_array = _class_ids(predictions, "predictions", allow_invalid=True)
    if len(label_array) != len(prediction_array):
        raise ValueError("labels and predictions must have the same nonzero length")
    indices = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        label_array, prediction_array, labels=indices, zero_division=0
    )
    return {
        "rows": len(label_array),
        "accuracy": float(accuracy_score(label_array, prediction_array)),
        "macro_f1": float(
            f1_score(
                label_array,
                prediction_array,
                labels=indices,
                average="macro",
                zero_division=0,
            )
        ),
        "invalid_prediction_rate": float(
            np.count_nonzero(
                (prediction_array < 0) | (prediction_array >= len(CLASS_NAMES))
            )
            / len(prediction_array)
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            label_array, prediction_array, labels=indices
        ).tolist(),
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: list[int], *, n_bins: int = 15
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    label_array = _class_ids(labels, "labels")
    if type(n_bins) is not int or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape [rows, 4]")
    if len(probabilities) != len(label_array):
        raise ValueError("probabilities and labels must have the same nonzero length")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == label_array
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (confidence >= low) & (confidence <= high)
            if index == 0
            else (confidence > low) & (confidence <= high)
        )
        count = int(mask.sum())
        if count:
            average_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = average_confidence - accuracy
            ece += count / len(label_array) * abs(gap)
        else:
            average_confidence = accuracy = gap = None
        bins.append(
            {
                "lower": float(low),
                "upper": float(high),
                "count": count,
                "average_confidence": average_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {"ece": float(ece), "bins": bins, "n_bins": n_bins}


def negative_log_likelihood(
    class_logits: np.ndarray, labels: list[int], temperature: float = 1.0
) -> float:
    logits, label_array = _logits_and_labels(class_logits, labels)
    fitted_temperature = _positive_finite(temperature, "temperature")
    probabilities = softmax(logits / fitted_temperature)
    true_probability = probabilities[np.arange(len(label_array)), label_array]
    return float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean())


def fit_temperature(
    class_logits: np.ndarray,
    labels: list[int],
    *,
    low: float = 0.05,
    high: float = 10.0,
    tolerance: float = 1e-4,
) -> float:
    """Golden-section search for the NLL-minimizing temperature.

    Raises if the minimum lies outside the search range. Returning the boundary
    silently would report a wall as a fitted value: a model whose true optimum is
    far above `high` would be handed a temperature that leaves it badly
    miscalibrated, and every ECE derived from it would be wrong without any signal
    that the fit failed.
    """
    low = _positive_finite(low, "temperature search lower bound")
    high = _positive_finite(high, "temperature search upper bound")
    tolerance = _positive_finite(tolerance, "temperature search tolerance")
    if low >= high:
        raise ValueError("temperature search lower bound must be below its upper bound")
    if tolerance >= high - low:
        raise ValueError("temperature search tolerance must be smaller than its range")
    class_logits, labels = _logits_and_labels(class_logits, labels)
    search_low, search_high = low, high
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = high - inverse_phi * (high - low)
    d = low + inverse_phi * (high - low)
    while high - low > tolerance:
        if negative_log_likelihood(class_logits, labels, c) < negative_log_likelihood(
            class_logits, labels, d
        ):
            high = d
        else:
            low = c
        c = high - inverse_phi * (high - low)
        d = low + inverse_phi * (high - low)
    fitted = float((low + high) / 2.0)

    margin = 10 * tolerance
    if fitted <= search_low + margin or fitted >= search_high - margin:
        raise ValueError(
            f"temperature search converged to its boundary ({fitted:.4f} in "
            f"[{search_low}, {search_high}]); the optimum lies outside the range, so "
            "this is not a fitted temperature. Widen the range deliberately and "
            "record that the range changed."
        )
    return fitted


def evaluation_block(
    class_logits: np.ndarray, labels: list[int], *, temperature: float = 1.0
) -> dict[str, Any]:
    class_logits, label_array = _logits_and_labels(class_logits, labels)
    temperature = _positive_finite(temperature, "temperature")
    probabilities = softmax(class_logits / temperature)
    predictions = probabilities.argmax(axis=1).tolist()
    return {
        **evaluate_predictions(label_array.tolist(), predictions),
        "nll": negative_log_likelihood(class_logits, label_array.tolist(), temperature),
        "calibration": expected_calibration_error(probabilities, label_array.tolist()),
        "temperature": temperature,
    }
