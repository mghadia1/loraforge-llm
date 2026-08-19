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


class SplitLeakError(ValueError):
    """Raised when a row used for training also appears in a split used to judge it.

    Caught by hand once: growing the training set toward the publisher's full
    120,000 rows would have swallowed the 2,000 validation rows, and every
    selection decision made afterwards would have been scored on trained-on data.
    A bare `assert` was not enough, since `python -O` removes it.
    """


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
    # Size of the publisher split the development rows were drawn from. Recorded
    # rather than assumed: describe() used to subtract from a hardcoded 120,000,
    # which fabricates an "unused rows" figure for any other corpus and writes it
    # into an evidence artifact.
    publisher_train_rows: int | None = None

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
    validation_start = (
        config.train_per_class
        if config.validation_start_per_class is None
        else config.validation_start_per_class
    )
    validation_stop = validation_start + config.validation_per_class
    required = config.train_per_class + config.validation_per_class
    for label, items in grouped.items():
        minimum_rows = max(required, validation_stop)
        if len(items) < minimum_rows:
            raise ValueError(
                f"class {label} has {len(items)} rows; {minimum_rows} required"
            )
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{config.seed}\0{item.row_id}".encode("utf-8")
            ).hexdigest(),
        )
        held_out = ordered[validation_start:validation_stop]
        training_candidates = ordered[:validation_start] + ordered[validation_stop:]
        train.extend(training_candidates[: config.train_per_class])
        validation.extend(held_out)

    train.sort(key=lambda item: item.row_id)
    validation.sort(key=lambda item: item.row_id)
    train_split = Split("train", tuple(train))
    validation_split = Split("validation", tuple(validation))
    assert_disjoint(train_split, validation_split)
    return train_split, validation_split


def content_id(example: Example) -> str:
    """Identity of a row by what it contains, independent of which split it came from.

    `row_id` mixes the source split name into its digest, so the same article in
    the publisher's train and test splits gets two different row_ids. Comparing
    row_ids across splits therefore compares two namespaces that cannot intersect,
    and the check silently passes no matter what the data holds. Content identity
    is what a cross-split leak actually means.
    """
    payload = f"{example.label}\0{' '.join(example.text.split())}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_disjoint(trained_on: Split, judged_on: Split) -> None:
    """Refuse any overlap between rows trained on and rows used to judge training."""
    shared_rows = {item.row_id for item in trained_on.examples} & {
        item.row_id for item in judged_on.examples
    }
    shared_content = {content_id(item) for item in trained_on.examples} & {
        content_id(item) for item in judged_on.examples
    }
    shared = shared_rows or shared_content
    if shared:
        kind = (
            "also appear in" if shared_rows else "share identical article text and label with"
        )
        raise SplitLeakError(
            f"{len(shared)} of {len(judged_on)} {judged_on.name} rows {kind} "
            f"{trained_on.name}; selection would be scored on trained-on data"
        )


def load_dataset(*, allow_test: bool = False, config: DataConfig | None = None) -> DatasetBundle:
    config = config or DataConfig()
    from datasets import load_dataset as hf_load_dataset

    publisher_train = hf_load_dataset(
        config.dataset_name,
        revision=config.dataset_revision,
        split="train",
    )
    if len(publisher_train) != 120_000:
        raise ValueError("upstream AG News train split size changed")
    names = tuple(publisher_train.features["label"].names)
    if names != CLASS_NAMES:
        raise ValueError(f"upstream class names changed: {names}")
    train, validation = deterministic_development_split(publisher_train, config)
    test = None
    if allow_test:
        publisher_test = hf_load_dataset(
            config.dataset_name,
            revision=config.dataset_revision,
            split="test",
        )
        if len(publisher_test) != config.publisher_test_rows:
            raise ValueError("upstream AG News test split size changed")
        test_names = tuple(publisher_test.features["label"].names)
        if test_names != CLASS_NAMES:
            raise ValueError(f"upstream test class names changed: {test_names}")
        test = Split("test", tuple(_to_examples("test", publisher_test)))
    bundle = DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        publisher_train_rows=len(publisher_train),
    )
    assert_no_leaks(bundle)
    return bundle


def assert_no_leaks(bundle: DatasetBundle) -> None:
    """Check every split pair that could contaminate a decision, on every load."""
    assert_disjoint(bundle.train, bundle.validation)
    if bundle.test is not None:
        assert_disjoint(bundle.train, bundle.test)
        assert_disjoint(bundle.validation, bundle.test)


def describe(bundle: DatasetBundle, config: DataConfig) -> dict[str, Any]:
    splits = [bundle.train, bundle.validation]
    if bundle.test is not None:
        splits.append(bundle.test)
    selection = {
        "seed": config.seed,
        "algorithm": "per-class SHA-256 ordering; fixed counts; row-ID sort",
        "unused_publisher_train_rows": (
            None
            if bundle.publisher_train_rows is None
            else bundle.publisher_train_rows - len(bundle.train) - len(bundle.validation)
        ),
    }
    if config.validation_start_per_class is not None:
        selection.update(
            {
                "algorithm": (
                    "per-class SHA-256 ordering; fixed validation window; "
                    "training excludes validation; row-ID sort"
                ),
                "validation_start_per_class": config.validation_start_per_class,
            }
        )
    return {
        "schema_version": 1,
        "dataset": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "publisher_train_rows": bundle.publisher_train_rows,
        "publisher_test_rows": config.publisher_test_rows,
        "selection": selection,
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
