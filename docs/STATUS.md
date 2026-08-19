# Status — August 8, 2026

## Completion

- Implementation: complete.
- Phase-one T4 baseline and QLoRA setup: complete.
- Two-epoch QLoRA training: complete.
- Validation-only checkpoint selection: complete; epoch 2 selected.
- Frozen validation-only calibration: complete.
- Single publisher-test evaluation: complete.
- Local evidence verification: complete.
- Selected-adapter release package: prepared.
- Résumé eligible: **no** until Mayank passes the oral explanation gate.

The repository contains a single completed experiment. No test result was used
to alter its prompt, data, model, hyperparameters, checkpoint, or calibration
temperature.

## Verified headline result

| System | Publisher-test accuracy | Publisher-test macro-F1 |
|---|---:|---:|
| Untuned Mistral 7B | 0.7428 | 0.7262 |
| Selected epoch-2 QLoRA | **0.9333** | **0.9333** |

Macro-F1 delta: **+0.2071**. Final evaluation count: **1**.

See [results.md](results.md) for per-class metrics, calibration, runtime, and
limitations.

## Verified development protocol

- Train: 8,000 rows, exactly 2,000 per class; digest
  `0ec701367f1111d94a659335a9c3e811683a407e32350a4865e53f43bdfeaa5d`.
- Validation: 2,000 rows, exactly 500 per class; digest
  `bd9922811b0418edba481a1f73fede5a202f934133ebac6a0cf866bdb2143c7c`.
- Publisher test: 7,600 rows, accessed after selection was frozen.
- Base validation compatibility check: 0.000034 macro-F1 difference between
  phase-one and phase-two scoring, under the frozen 0.005 tolerance.
- Epoch 1 validation macro-F1: 0.9248.
- Epoch 2 validation macro-F1: 0.9310; selected.

## Audit corrections preserved

### Tokenizer compatibility

The first T4 attempt exposed a Transformers return-shape change:
`apply_chat_template(tokenize=True)` returned a `BatchEncoding`. The prompt
normalizer and manual left-padding path now accept the supported return shapes,
and regression tests cover the original failure.

### Packed 4-bit parameter denominator

`outputs/qlora-setup.json` remains unedited and reports 1.1037% trainable
parameters. That percentage is not quoted as a result: packed 4-bit
`Params4bit.numel()` undercounted the full denominator. The valid, measured
quantity is the numerator — 41,943,040 trainable adapter parameters, roughly
0.6% of a 7B model. The code and tests now reject the packed denominator.

No completed run recorded an audited unpacked denominator, because
`parameter_report` was not called during training. It is now invoked before the
first optimizer step, so the next run stores the audited count in
`outputs/training-report.json`. Until then, quote the numerator and an
approximate fraction, not a precise percentage.

### Efficient final-position scoring

The scorer applies the LM head only to the final sequence position when the
installed Transformers version supports it. It falls back safely on older
versions. The phase-two base compatibility gate demonstrated that this compute
optimization did not change the experimental contract.

## Evidence inventory

| Artifact | Status |
|---|---|
| `docs/evidence/data-stats.json` | tracked |
| `outputs/base-validation.json` | tracked |
| `outputs/qlora-setup.json` | tracked, with documented percentage correction |
| `outputs/training-report.json` | verified |
| `outputs/frozen-selection.json` | verified, test-evaluated marker true |
| `outputs/final-test-report.json` | verified |
| `outputs/logits/*.npy` | verified against recorded array hashes |
| selected adapter | release asset; inner and archive hashes pinned |

The local verifier independently returned:

```text
training_report: verified, selected_epoch: 2
final_test_report: verified
base macro_f1: 0.7261902147623102
tuned macro_f1: 0.9332520991438105
macro_f1_delta: 0.2070618843815003
```

## Remaining human gate

The experiment is technically complete. Publication to the résumé or portfolio
remains blocked until Mayank answers all twelve questions in
[`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) unaided. The candidate wording is in
[`RESUME_CANDIDATE.md`](RESUME_CANDIDATE.md), but it must not be copied into an
application before that gate is passed.
