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


def save_logits(array: np.ndarray, root: Path, relative_path: str) -> dict[str, Any]:
    """Write logits under ``root`` but record the repo-relative path, so evidence relocates."""
    target = Path(root) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    np.save(target, values)
    return {
        "path": relative_path,
        "sha256": sha256_array(values),
        "shape": list(values.shape),
    }


def load_logits(reference: dict[str, Any], *, root: Path = Path(".")) -> np.ndarray:
    """Load a saved logit file and refuse it if its hash no longer matches the report."""
    path = root / reference["path"]
    if not path.exists():
        raise EvidenceError(f"missing logits file {path}")
    values = np.load(path).astype(np.float32)
    actual = sha256_array(values)
    if actual != reference["sha256"]:
        raise EvidenceError(
            f"{path} hash {actual} does not match the recorded {reference['sha256']}"
        )
    return values


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise EvidenceError(f"required artifact {path} does not exist")
    return json.loads(Path(path).read_text(encoding="utf-8"))
