"""Classification and calibration metrics with fixed four-class semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import CLASS_NAMES


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_predictions(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must have the same nonzero length")
    indices = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=indices, average="macro", zero_division=0)
        ),
        "invalid_prediction_rate": float(
            sum(prediction not in indices for prediction in predictions) / len(predictions)
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
        "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: list[int], *, n_bins: int = 15
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape [rows, 4]")
    if len(probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("probabilities and labels must have the same nonzero length")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(labels)
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
            ece += count / len(labels) * abs(gap)
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
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = softmax(np.asarray(class_logits) / temperature)
    true_probability = probabilities[np.arange(len(labels)), np.asarray(labels)]
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
    probabilities = softmax(np.asarray(class_logits) / temperature)
    predictions = probabilities.argmax(axis=1).tolist()
    return {
        **evaluate_predictions(labels, predictions),
        "nll": negative_log_likelihood(class_logits, labels, temperature),
        "calibration": expected_calibration_error(probabilities, labels),
        "temperature": temperature,
    }
