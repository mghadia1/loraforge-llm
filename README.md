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
PYTHONPATH=src python -m loraforge.cli verify --root . --reports-only
```

Reports-only verification requires the pinned AG News data revision but does
not run the model or require adapter files. It selects epoch 2 again,
recomputes every stored metric and logit hash, and explicitly reports
`adapter_files_verified: false`. Strict verification remains the default and
also checks both checkpoint directories plus the selected adapter.

## Expanded-data follow-up (validation only)

[`configs/experiment-expanded-data.json`](configs/experiment-expanded-data.json)
defines a separate follow-up that trains on 16,000 unique publisher-train rows
while preserving the completed experiment's exact 2,000 validation rows. It
uses one epoch, so its 1,000 optimizer steps match the original run's two
epochs over 8,000 rows. This isolates broader data coverage from additional
training compute.

The follow-up sets `test_evaluations_allowed` to `0`; `loraforge final-test`
refuses that config. It has no result yet and must not change or replace the
verified experiment above. Prepare it in a separate ignored run directory:

```bash
mkdir -p runs/expanded-data/outputs
cp outputs/base-validation.json runs/expanded-data/outputs/
loraforge data-stats --config configs/experiment-expanded-data.json \
  --output docs/evidence/expanded-data-stats.json
loraforge train --config configs/experiment-expanded-data.json \
  --root runs/expanded-data
loraforge freeze-selection --root runs/expanded-data
```

The existing base-validation artifact is reusable because the validation row
IDs are deliberately unchanged. Training still requires a suitable CUDA GPU;
this repository does not claim the follow-up has run.

## Selected adapter

The 167,838,575-byte selected adapter is distributed as ordered release parts
under
[v1.0-evidence](https://github.com/mghadia1/loraforge-llm/releases/tag/v1.0-evidence).
Reassemble them in filename order:

```bash
cat loraforge-selected-adapter-epoch2.tar.gz.part-* > loraforge-selected-adapter-epoch2.tar.gz
shasum -a 256 loraforge-selected-adapter-epoch2.tar.gz
tar -xzf loraforge-selected-adapter-epoch2.tar.gz
```

The expected archive digest is
`206aa291ff32904eff9569c40a8265182a227a9ec9e46e64e9a38f1e4f800603`.
All part, archive, and inner-file hashes are pinned in
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
