from __future__ import annotations

import json
from pathlib import Path


FRESH_T4_NOTEBOOK = Path("notebooks/loraforge_t4_phase2_fresh.ipynb")


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )


def test_fresh_t4_training_uses_the_validated_canonical_config() -> None:
    source = _code_source(FRESH_T4_NOTEBOOK)

    assert "RUN_ROOT = Path('runs/phase2-fresh')" in source
    assert "if RUN_ROOT.exists():" in source
    assert (
        "loraforge.cli train --config configs/experiment.json --root {RUN_ROOT}"
        in source
    )
    assert "loraforge.cli freeze-selection --root {RUN_ROOT}" in source
    assert "loraforge.cli verify --root {RUN_ROOT} --training-only" in source
    assert "default_config" not in source
    assert "train_qlora" not in source
