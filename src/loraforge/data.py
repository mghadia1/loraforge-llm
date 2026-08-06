"""Pinned AG News loading with deterministic development subsets and a test lock."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import DataConfig


CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
CLASS_CODES = ("A", "B", "C", "D")
CODE_TO_LABEL = dict(zip(CLASS_CODES, range(len(CLASS_NAMES))))
LABEL_TO_CODE = dict(enumerate(CLASS_CODES))


class LockedTestSplitError(PermissionError):
    """Raised when code touches the publisher test split before the final run."""


@dataclass(frozen=True)
class Example:
    row_id: str
    text: str
    label: int
    source_index: int


@dataclass(frozen=True)
class Split:
    name: str
    examples: tuple[Example, ...]

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.examples]

    @property
    def labels(self) -> list[int]:
        return [item.label for item in self.examples]

    def class_counts(self) -> dict[str, int]:
        counts = Counter(self.labels)
        return {name: counts.get(index, 0) for index, name in enumerate(CLASS_NAMES)}

    def id_sha256(self) -> str:
        payload = "\n".join(item.row_id for item in self.examples).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    train: Split
    validation: Split
    test: Split | None = None

    def require_test(self) -> Split:
        if self.test is None:
            raise LockedTestSplitError(
                "publisher test is locked; select the adapter and temperature on validation, "
                "then explicitly load allow_test=True for the one final evaluation"
            )
        return self.test


def _row_id(source_split: str, index: int, text: str, label: int) -> str:
    raw = f"{source_split}\0{index}\0{label}\0{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _to_examples(source_split: str, rows: Iterable[dict[str, Any]]) -> list[Example]:
    return [
        Example(
            row_id=_row_id(source_split, index, str(row["text"]), int(row["label"])),
            text=str(row["text"]),
            label=int(row["label"]),
            source_index=index,
        )
        for index, row in enumerate(rows)
    ]


def deterministic_development_split(
    rows: Iterable[dict[str, Any]], config: DataConfig
) -> tuple[Split, Split]:
    """Select balanced train/validation subsets without depending on RNG libraries."""
    grouped: dict[int, list[Example]] = {index: [] for index in range(len(CLASS_NAMES))}
    for item in _to_examples("train", rows):
        if item.label not in grouped:
            raise ValueError(f"unexpected class label {item.label}")
        grouped[item.label].append(item)

    train: list[Example] = []
    validation: list[Example] = []
    required = config.train_per_class + config.validation_per_class
    for label, items in grouped.items():
        if len(items) < required:
            raise ValueError(f"class {label} has {len(items)} rows; {required} required")
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{config.seed}\0{item.row_id}".encode("utf-8")
            ).hexdigest(),
        )
        train.extend(ordered[: config.train_per_class])
        validation.extend(ordered[config.train_per_class : required])

    train.sort(key=lambda item: item.row_id)
    validation.sort(key=lambda item: item.row_id)
    if set(item.row_id for item in train) & set(item.row_id for item in validation):
        raise AssertionError("development split overlap")
    return Split("train", tuple(train)), Split("validation", tuple(validation))


def load_dataset(*, allow_test: bool = False, config: DataConfig | None = None) -> DatasetBundle:
    config = config or DataConfig()
    from datasets import load_dataset as hf_load_dataset

    raw = hf_load_dataset(config.dataset_name, revision=config.dataset_revision)
    if len(raw["train"]) != 120_000 or len(raw["test"]) != config.publisher_test_rows:
        raise ValueError("upstream AG News split sizes changed")
    names = tuple(raw["train"].features["label"].names)
    if names != CLASS_NAMES:
        raise ValueError(f"upstream class names changed: {names}")
    train, validation = deterministic_development_split(raw["train"], config)
    test = None
    if allow_test:
        test = Split("test", tuple(_to_examples("test", raw["test"])))
    return DatasetBundle(train=train, validation=validation, test=test)


def describe(bundle: DatasetBundle, config: DataConfig) -> dict[str, Any]:
    splits = [bundle.train, bundle.validation]
    if bundle.test is not None:
        splits.append(bundle.test)
    return {
        "schema_version": 1,
        "dataset": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "publisher_train_rows": 120_000,
        "publisher_test_rows": config.publisher_test_rows,
        "selection": {
            "seed": config.seed,
            "algorithm": "per-class SHA-256 ordering; fixed counts; row-ID sort",
            "unused_publisher_train_rows": 120_000 - len(bundle.train) - len(bundle.validation),
        },
        "classes": list(CLASS_NAMES),
        "splits": {
            split.name: {
                "rows": len(split),
                "class_counts": split.class_counts(),
                "row_ids_sha256": split.id_sha256(),
            }
            for split in splits
        },
        "test_loaded": bundle.test is not None,
    }


def write_stats(bundle: DatasetBundle, config: DataConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(describe(bundle, config), indent=2) + "\n", encoding="utf-8")
