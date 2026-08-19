# Measured results

All numbers on this page come from the hash-backed artifacts under `outputs/`.
The command `PYTHONPATH=src python -m loraforge.cli verify --root .`
independently selected epoch 2 and recomputed the final metrics from stored
logits on August 8, 2026.

## Validation and selection

| System/checkpoint | Accuracy | Macro-F1 |
|---|---:|---:|
| Untuned base | 0.7475 | 0.7299 |
| QLoRA epoch 1 | 0.9250 | 0.9248 |
| QLoRA epoch 2 | **0.9310** | **0.9310** |

Epoch 2 was selected by the frozen rule: maximum validation macro-F1, with an
exact tie resolved to the earlier epoch. The phase-two base re-score differed
from the phase-one baseline by only 0.000034 macro-F1, inside the predeclared
0.005 compatibility tolerance.

Separate validation-only temperatures were fitted after selection:

- base: 4.5263;
- tuned: 1.2925.

Temperature scaling left every argmax prediction unchanged.

## Single publisher-test evaluation

The complete 7,600-row publisher test split was evaluated once after the
selection was frozen.

| System | Accuracy | Macro-F1 | NLL before/after | ECE before/after |
|---|---:|---:|---:|---:|
| Untuned base | 0.7428 | 0.7262 | 1.9637 / 0.7128 | 0.2241 / 0.0263 |
| Selected QLoRA | **0.9333** | **0.9333** | **0.2093 / 0.1965** | **0.0290 / 0.0078** |

The selected adapter improved macro-F1 by **0.2071** and accuracy by
**0.1905**. This is the final held-out result; it was not used to choose an
epoch, prompt, hyperparameter, or calibration temperature.

### Per-class test F1

| Class | Base | Selected QLoRA |
|---|---:|---:|
| World | 0.7944 | **0.9399** |
| Sports | 0.8964 | **0.9872** |
| Business | 0.7096 | **0.9008** |
| Sci/Tech | 0.5043 | **0.9052** |

The largest change was Sci/Tech recall: 0.3574 for the base system versus
0.8916 after adaptation. The base model often mapped Sci/Tech articles to
Business; QLoRA substantially reduced that error mode.

## Runtime and artifact measurements

- Hardware: Tesla T4, CUDA 12.8.
- Training: 13,083.95 seconds (about 3 hours 38 minutes) for two epochs.
- Training report peak CUDA allocation: 5.77 GiB.
- Phase-one model/setup peak CUDA allocation: 12.34 GiB.
- Base test scoring: 1,248.92 seconds (20 minutes 49 seconds).
- Tuned test scoring: 1,477.03 seconds (24 minutes 37 seconds).
- Selected adapter: 167,838,575 bytes.
- Trainable LoRA weights: 41,943,040.

The first phase-one setup artifact reports a misleading 1.1037% trainable
percentage because bitsandbytes' packed `Params4bit.numel()` undercounts the
base denominator; the code and tests now reject that packed count. The original
artifact remains unchanged as part of the audit trail.

**The measured quantity is the numerator: 41,943,040 trainable adapter
parameters — roughly 0.6% of a 7B model.** No artifact in this repository
records the unpacked denominator, so a precise percentage would rest on
Mistral's published parameter count rather than on something measured here.
`parameter_report` now runs inside training and will record the audited count
directly on the next run.

## Uncertainty of the measured gap

`loraforge intervals` resamples the rows of the single stored test run. It spends
no test budget and needs no GPU: it re-analyzes the evaluation that already
happened, and `loraforge verify` recomputes it.

| System | Macro-F1 | 95% CI |
|---|---:|---:|
| Untuned base | 0.7262 | [0.7168, 0.7357] |
| Selected QLoRA | 0.9333 | [0.9276, 0.9386] |
| **Improvement** | **0.2071** | **[0.1970, 0.2166]** |

In 0 of 2,000 paired resamples did the adapter fail to beat the base model.

Row-level, the two systems disagree on 1,767 of 7,600 test rows: the adapter
**fixed 1,577 base errors and broke 129** (McNemar chi-square 1,227.3 on 1,706
discordant pairs, exact two-sided log10 p = -316.1).

This bounds **sampling** uncertainty only — how the gap would move if a different
sample of articles had been drawn. It does **not** measure training variance,
which would require additional training runs with different seeds. That
limitation stands.

## Capacity ablation: rank 4 versus rank 16

A controlled second run (`docs/ABLATION_RANK4.md`) cut LoRA rank from 16 to 4, scaling
alpha with it so the `alpha/rank` update scaling stayed fixed.

| | rank 16 | rank 4 |
|---|---:|---:|
| validation macro-F1 | 0.9310 | 0.9360 |
| trainable parameters | 41,943,040 (0.578%) | 10,485,760 (0.1445%) |
| adapter bytes | 167,838,575 | 42,008,469 |

The difference is **+0.0050 with a 95% CI of [-0.0023, +0.0122]** and McNemar p = 0.220,
so the two are statistically indistinguishable: **equal quality at a quarter of the
trainable parameters**, not a rank-4 win. Validation only; the test split was not used.

## Evidence trail

- `outputs/training-report.json`: loss curve, environment, validation logits,
  adapter hashes, and selected epoch.
- `outputs/frozen-selection.json`: selected checkpoint and validation-fitted
  temperatures, written before the test evaluation.
- `outputs/final-test-report.json`: exactly one base-versus-tuned test result.
- `outputs/logits/*.npy`: raw logits from which reported metrics are
  recomputed.
- `outputs/test-intervals.json`: bootstrap confidence intervals and paired
  error analysis, recomputed from the same stored logits without a new test run.
- `docs/evidence/selected-adapter-release.json`: release archive and inner
  adapter hashes.

The release provides the selected inference adapter as ordered, hash-pinned
parts. `loraforge verify --reports-only` reproduces numerical claims from the
tracked logits; strict verification additionally requires the owner-maintained
epoch checkpoint archive.

The final verifier output was:

```text
training_report: verified, selected_epoch: 2
final_test_report: verified
base macro_f1: 0.7261902147623102
tuned macro_f1: 0.9332520991438105
macro_f1_delta: 0.2070618843815003
```

## Limitations

- One seed and one hardware run do not establish **training** variance. Test-set
  sampling uncertainty is now quantified (see the confidence intervals above),
  but a different seed could still produce a different adapter, and that has not
  been measured.
- Only 8,000 of 120,000 publisher-train examples were used.
- The comparison is against the same untuned base model, not full fine-tuning
  or another architecture.
- Constrained A-D class logits guarantee a valid output and therefore do not
  measure free-form generation robustness.
- AG News is a clean four-class benchmark; performance should not be treated as
  evidence for medical, safety-critical, or open-domain deployment.
- The final test is intentionally not repeat-tuned. Any future protocol change
  requires a separately versioned experiment.
