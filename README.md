# LoRAForge

LoRAForge is an evidence-first QLoRA study: adapt the public Apache-2.0
`mistralai/Mistral-7B-Instruct-v0.3` model to four-class AG News topic
classification on a single free T4, then compare the untuned base and selected
adapter under one frozen held-out evaluation.

**Current status: the full pipeline is implemented and tested GPU-free; no GPU
run has happened yet, so there are no results. Resume eligible: no.** Do not
claim a baseline score, trainable-parameter percentage, memory use, adapter
improvement, or calibration result until the notebooks produce the corresponding
evidence and `loraforge verify` reproduces it. Mayank must also pass the
[explanation gate](docs/LEARNING_GUIDE.md).

## Frozen design

- Dataset: `fancyzhx/ag_news` at commit
  `eb185aade064a813bc0b7f42de02595523103ca4`.
- Model: `mistralai/Mistral-7B-Instruct-v0.3` at commit
  `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- Development data: balanced deterministic subset, 8,000 train and 2,000
  validation rows. The other 110,000 publisher-train rows are unused.
- Held-out data: all 7,600 publisher-test rows, locked until one final
  base-versus-selected-adapter evaluation.
- Output contract: `A=World`, `B=Sports`, `C=Business`, `D=Sci/Tech`.
- Class scores: the next-token logits of four contextual single-token codes,
  normalized together. This makes full macro-F1 and ECE feasible.
- QLoRA: 4-bit NF4, double quantization, FP16 compute, rank 16, alpha 32,
  dropout 0.05, attention and MLP projection targets.
- Selection: validation macro-F1. Calibration temperatures are also fit on
  validation; test labels never select a checkpoint or temperature.

## What exists

- Deterministic data loader, split digests, test lock, and three real examples.
- Prompt and assistant-token-only label masking, plus the padding collator that
  keeps prompt and padding tokens out of the loss.
- Four-class macro-F1, per-class metrics, NLL, ECE, and temperature scaling.
- 4-bit base-model loader, efficient class-logit scorer, LoRA attachment, and
  trainable-parameter audit.
- Two-epoch training driver with per-epoch checkpointing, validation scoring,
  and a one-line selection rule (macro-F1; ties to the earlier epoch).
- A pre-test freeze step and a guarded single publisher-test evaluation.
- Hash-backed evidence: every reported number is recomputable from stored logits
  via `loraforge verify`, and hand-edited reports fail verification.
- 38 GPU-free tests and a GitHub Actions workflow.
- Two T4 notebooks: [phase 1](notebooks/loraforge_t4.ipynb) (baseline and QLoRA
  setup) and [phase 2](notebooks/loraforge_t4_phase2.ipynb) (train, freeze, one
  final test).

## What does not exist yet

No number. There is no base macro-F1, no trainable-parameter percentage, no T4
memory figure, no training time, no trained adapter, no test delta, and no ECE
result, because no GPU run has been executed. The `outputs/` directory is empty.

## Order of operations

```bash
# 1. on a T4: notebooks/loraforge_t4.ipynb  -> outputs/base-validation.json, outputs/qlora-setup.json
# 2. on a T4: notebooks/loraforge_t4_phase2.ipynb, which runs:
loraforge train --root .              # two frozen epochs, per-epoch validation, selection
loraforge freeze-selection --root .   # GPU-free: verify evidence, fit both temperatures
loraforge final-test --root . --confirm i-am-running-the-single-final-test
loraforge verify --root .             # recompute every number from the stored logits
```

## Local checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
loraforge write-config
loraforge data-stats
loraforge examples
```

If the `loraforge` console script reports `No module named 'loraforge'` on this
Mac, macOS has set the hidden flag on the editable install's `.pth` file and
Python 3.14 skips hidden `.pth` files. Use `PYTHONPATH=src python -m
loraforge.cli <command>` instead; pytest is unaffected.

`write-config`, `data-stats`, `examples`, `freeze-selection`, and `verify` never
load model weights; only `train` and `final-test` need a GPU, and only
`final-test` reads the publisher test split. Read
[`docs/how-it-works.md`](docs/how-it-works.md) before changing the protocol and
[`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) before claiming any of it.
