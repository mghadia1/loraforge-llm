# LoRAForge

LoRAForge is an evidence-first QLoRA study that adapts the public Apache-2.0
`mistralai/Mistral-7B-Instruct-v0.3` model to four-class AG News topic
classification on a single free Tesla T4. The untuned base and selected adapter
were compared once on the full publisher test split after checkpoint and
temperature selection were frozen on validation.

**Experiment status: complete and independently verified. Resume eligible:
no, pending Mayank's oral [explanation gate](docs/LEARNING_GUIDE.md).**

## Verified result

| Split/system | Accuracy | Macro-F1 | ECE before calibration | ECE after calibration |
|---|---:|---:|---:|---:|
| Validation, base | 0.7475 | 0.7299 | 0.2165 | 0.0299 |
| Validation, selected QLoRA | 0.9310 | 0.9310 | 0.0300 | 0.0166 |
| Publisher test, base | 0.7428 | 0.7262 | 0.2241 | 0.0263 |
| Publisher test, selected QLoRA | **0.9333** | **0.9333** | 0.0290 | **0.0078** |

The selected epoch-2 adapter improved publisher-test macro-F1 by **0.2071**
and accuracy by **0.1905**. The final report records exactly one test
evaluation over all 7,600 rows. `loraforge verify` recomputes both systems'
metrics from their stored logits and verifies their SHA-256 hashes.

See [the measured results](docs/results.md) for resource use, per-class results,
calibration, limitations, and the evidence trail.

## Frozen design

- Dataset: `fancyzhx/ag_news` at commit
  `eb185aade064a813bc0b7f42de02595523103ca4`.
- Model: `mistralai/Mistral-7B-Instruct-v0.3` at commit
  `c170c708c41dac9275d15a8fff4eca08d52bab71`.
- Development data: balanced deterministic subset, 8,000 train and 2,000
  validation rows. The other 110,000 publisher-train rows are unused.
- Held-out data: all 7,600 publisher-test rows, accessed for one final
  base-versus-selected-adapter evaluation.
- Output contract: `A=World`, `B=Sports`, `C=Business`, `D=Sci/Tech`.
- Class scores: the next-token logits of four contextual single-token codes,
  normalized together.
- QLoRA: 4-bit NF4, double quantization, FP16 compute, rank 16, alpha 32,
  dropout 0.05, attention and MLP projection targets.
- Training: two frozen epochs, effective batch 16, learning rate `2e-4`,
  assistant-answer-only loss.
- Selection: validation macro-F1, with exact ties resolved to the earlier
  epoch. Separate temperatures are fit on validation for base and tuned models.

## Evidence and reproducibility

The repository includes:

- phase-one baseline/setup evidence;
- training report, frozen selection, and final test report;
- raw validation and test logits;
- deterministic split and prompt audits;
- GPU-free tests for leakage, masking, selection, hashes, metric
  recomputation, and evidence tampering.

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
PYTHONPATH=src python -m loraforge.cli verify --root .
```

Verification requires the pinned AG News data revision but does not run the
model. It selects epoch 2 again and recomputes the final macro-F1 values and
logit hashes.

## Selected adapter

The 167,838,575-byte selected adapter is distributed as the private GitHub
release asset
`loraforge-selected-adapter-epoch2.tar.gz` under
[v1.0-evidence](https://github.com/mghadia1/loraforge-llm/releases/tag/v1.0-evidence).
The same release includes `loraforge-training-adapters-evidence.tar.gz`, which
restores both epoch checkpoints and the selected copy required by the strict
training verifier. Epoch 2 and `selected` are hard-linked in that archive to
avoid storing identical weights twice.
Its archive and inner-file hashes are pinned in
[`docs/evidence/selected-adapter-release.json`](docs/evidence/selected-adapter-release.json).
The large weights are intentionally excluded from normal Git history.

## Repository map

- [`docs/how-it-works.md`](docs/how-it-works.md): protocol and implementation.
- [`docs/results.md`](docs/results.md): measured results and limitations.
- [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md): explanation gate.
- [`outputs/final-test-report.json`](outputs/final-test-report.json): final
  metrics and provenance.
- [`outputs/frozen-selection.json`](outputs/frozen-selection.json): pre-test
  checkpoint and calibration decision.
- [`notebooks/loraforge_t4_phase2_fresh.ipynb`](notebooks/loraforge_t4_phase2_fresh.ipynb):
  safe training/freeze notebook.

## Limitations

- This is one seed and one T4 run, not a variance study.
- The training subset is balanced and uses 8,000 of 120,000 publisher-train
  rows; the result does not measure full-data training.
- Decoding is constrained to four class-code logits, so it measures
  classification rather than open-ended instruction following.
- There is no full-fine-tuning or other-architecture baseline.
- Temperature scaling improves calibration only; it cannot change predictions
  or macro-F1.
- The test result is final by design and was not used for tuning.
