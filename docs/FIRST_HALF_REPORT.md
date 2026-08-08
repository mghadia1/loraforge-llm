# First-half completion report

> Historical pre-GPU report. The experiment has since completed; see
> `docs/results.md` and `docs/STATUS.md` for verified measurements.

Completed by Codex on August 6, 2026.

## Delivered

1. Pinned public dataset/model revisions and frozen JSON configuration.
2. Deterministic balanced AG News development subset with exact digests.
3. Locked publisher test split; no development evidence command loads it.
4. One shared instruction prompt for base and tuned models.
5. Efficient four-logit classification/calibration design using contextual
   single-token `A–D` codes.
6. Assistant-token-only supervised labels.
7. 4-bit NF4 base loader, LoRA attachment, and trainable-parameter audit.
8. Macro-F1/per-class/NLL/ECE/temperature metric implementation.
9. T4 notebook through validation baseline and audited QLoRA setup.
10. Fifteen passing GPU-free tests and a CI workflow.

## Real local evidence

- Pinned dataset publisher sizes: 120,000 train / 7,600 test.
- Selected development rows: 8,000 train / 2,000 validation, balanced.
- Test loaded by development commands: false.
- Train digest: `0ec701367f1111d94a659335a9c3e811683a407e32350a4865e53f43bdfeaa5d`.
- Validation digest: `bd9922811b0418edba481a1f73fede5a202f934133ebac6a0cf866bdb2143c7c`.
- Token audit: median 127, p95 160, max 432; zero over frozen 512.
- Tests: 15 passed.

## Deliberately not claimed

No base macro-F1, trainable percentage, T4 memory, training time, adapter size,
tuned macro-F1, test delta, or ECE result exists yet. The T4 notebook must
produce the two prerequisite artifacts before Claude completes measured work.
