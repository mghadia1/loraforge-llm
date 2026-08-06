# LoRAForge

LoRAForge is an evidence-first QLoRA study: adapt the public Apache-2.0
`mistralai/Mistral-7B-Instruct-v0.3` model to four-class AG News topic
classification on a single free T4, then compare the untuned base and selected
adapter under one frozen held-out evaluation.

**Current status: first half implemented, no GPU results yet, resume eligible:
no.** Do not claim a baseline score, trainable-parameter percentage, memory use,
adapter improvement, or calibration result until the notebook produces the
corresponding evidence. Mayank must also pass the final explanation gate.

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
- Prompt and assistant-token-only label masking.
- Four-class macro-F1, per-class metrics, NLL, ECE, and temperature scaling.
- 4-bit base-model loader, efficient class-logit scorer, LoRA attachment, and
  trainable-parameter audit.
- T4 notebook through untuned validation baseline and QLoRA setup.
- 15 GPU-free tests and GitHub Actions workflow.
- Exact [Claude handoff](docs/CLAUDE_HANDOFF.md) for training and final evaluation.

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

Local commands do not load model weights or publisher test data. Run
[`notebooks/loraforge_t4.ipynb`](notebooks/loraforge_t4.ipynb) on a T4 for the
first GPU evidence, then follow the handoff. Read
[`docs/how-it-works.md`](docs/how-it-works.md) before changing the protocol.
