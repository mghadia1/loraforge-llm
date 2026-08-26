"""Hashes, package versions, and the failure used when evidence does not verify."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


AUDITED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "numpy",
    "peft",
    "scikit-learn",
    "torch",
    "transformers",
    "trl",
)


class EvidenceError(RuntimeError):
    """Raised when a stored artifact disagrees with what its own data recomputes."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> dict[str, Any]:
    """Hash every file in a directory by sorted relative path, plus a combined digest."""
    root = Path(path)
    if not root.is_dir():
        raise EvidenceError(f"{root} is not a directory")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise EvidenceError(f"{root} contains no files to hash")
    per_file = {str(item.relative_to(root)): sha256_file(item) for item in files}
    combined = hashlib.sha256()
    for name, digest in per_file.items():
        combined.update(f"{name}\0{digest}\n".encode("utf-8"))
    return {
        "files": per_file,
        "combined_sha256": combined.hexdigest(),
        "total_bytes": int(sum(item.stat().st_size for item in files)),
    }


def verify_directory_snapshot(
    path: Path,
    recorded: dict[str, Any],
    *,
    mutable_files: frozenset[str] = frozenset(),
) -> None:
    """Verify every immutable file in a recorded directory snapshot.

    PEFT writes ``README.md`` beside adapter weights.  That model card is
    distribution metadata, not executable adapter payload, and may be improved
    after training.  Callers can therefore mark it mutable while still requiring
    the recorded config and weight files to exist, match their hashes, and be the
    only immutable files present.
    """
    root = Path(path)
    if not root.is_dir():
        raise EvidenceError(f"{root} is not a directory")
    recorded_files = recorded.get("files")
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise EvidenceError("recorded directory snapshot has no file manifest")

    actual_paths = {
        str(item.relative_to(root)): item
        for item in root.rglob("*")
        if item.is_file()
    }
    expected_payload = set(recorded_files) - mutable_files
    actual_payload = set(actual_paths) - mutable_files
    if expected_payload != actual_payload:
        raise EvidenceError(
            "adapter payload files differ from the recorded snapshot: "
            f"missing={sorted(expected_payload - actual_payload)}, "
            f"unexpected={sorted(actual_payload - expected_payload)}"
        )
    for name in sorted(expected_payload):
        actual = sha256_file(actual_paths[name])
        if actual != recorded_files[name]:
            raise EvidenceError(
                f"adapter payload file {name} hash {actual} does not match "
                f"the recorded {recorded_files[name]}"
            )

    if not mutable_files:
        actual = sha256_directory(root)
        if actual != recorded:
            raise EvidenceError("directory snapshot does not match its recorded manifest")


def sha256_array(array: np.ndarray) -> str:
    """Hash a numeric array so a report cannot be edited away from its own logits."""
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(f"{values.dtype.str}\0{values.shape}\0".encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def sha256_labels(labels: list[int]) -> str:
    """Digest a label or prediction sequence, refusing an empty one.

    Hashing nothing yields the well-known digest of the empty string, which would
    compare equal to any other empty sequence and let a vacuous check pass.
    """
    if len(labels) == 0:
        raise EvidenceError("refusing to hash an empty label sequence")
    payload = ",".join(str(int(value)) for value in labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in AUDITED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu_name": None,
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        return record
    record["cuda_available"] = bool(torch.cuda.is_available())
    if record["cuda_available"]:
        record["gpu_name"] = torch.cuda.get_device_name(0)
        record["cuda_version"] = torch.version.cuda
    return record


def _resolve_logit_path(root: Path, relative_path: str) -> Path:
    """Resolve a canonical ``.npy`` path without allowing evidence to escape ``root``."""
    if not isinstance(relative_path, str) or not relative_path:
        raise EvidenceError("logits path must be a nonempty repo-relative string")
    candidate = Path(relative_path)
    if not candidate.parts or candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceError(
            f"logits path must stay repo-relative under its root: {relative_path!r}"
        )
    if candidate.suffix != ".npy":
        raise EvidenceError(f"logits path must end in .npy: {relative_path!r}")

    resolved_root = Path(root).resolve()
    target = (resolved_root / candidate).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as error:
        raise EvidenceError(
            f"logits path escapes its evidence root: {relative_path!r}"
        ) from error
    return target


def resolve_adapter_directory(root: Path, relative_path: str) -> Path:
    """Resolve an adapter directory without allowing evidence to escape ``root``."""
    if not isinstance(relative_path, str) or not relative_path:
        raise EvidenceError(
            "adapter directory must be a nonempty repo-relative string"
        )
    candidate = Path(relative_path)
    if not candidate.parts or candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceError(
            "adapter directory must stay repo-relative under its evidence root: "
            f"{relative_path!r}"
        )

    resolved_root = Path(root).resolve()
    target = (resolved_root / candidate).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as error:
        raise EvidenceError(
            f"adapter directory escapes its evidence root: {relative_path!r}"
        ) from error
    return target


def save_logits(array: np.ndarray, root: Path, relative_path: str) -> dict[str, Any]:
    """Write logits under ``root`` but record the repo-relative path, so evidence relocates."""
    target = _resolve_logit_path(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    np.save(target, values)
    return {
        "path": relative_path,
        "sha256": sha256_array(values),
        "shape": list(values.shape),
    }


def load_logits(reference: dict[str, Any], *, root: Path = Path(".")) -> np.ndarray:
    """Load logits only when their path, shape, and hash match the report."""
    if not isinstance(reference, dict):
        raise EvidenceError("logits reference must be an object")
    path = _resolve_logit_path(root, reference.get("path"))
    if not path.exists():
        raise EvidenceError(f"missing logits file {path}")
    values = np.load(path, allow_pickle=False).astype(np.float32)
    actual_shape = list(values.shape)
    if reference.get("shape") != actual_shape:
        raise EvidenceError(
            f"{path} shape {actual_shape} does not match the recorded {reference.get('shape')!r}"
        )
    actual = sha256_array(values)
    if actual != reference.get("sha256"):
        raise EvidenceError(
            f"{path} hash {actual} does not match the recorded {reference.get('sha256')!r}"
        )
    return values


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise EvidenceError(f"required artifact {path} does not exist")
    return json.loads(Path(path).read_text(encoding="utf-8"))
