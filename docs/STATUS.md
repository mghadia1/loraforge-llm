# Status — August 6, 2026

- Build steps 1–3 implementation (Codex): complete.
- Build steps 4–6 implementation (Claude): complete, GPU-free and unexecuted.
- Real dataset development evidence: complete; test not loaded.
- GPU execution of untuned validation baseline: **pending T4**.
- GPU execution of QLoRA parameter/memory audit: **pending T4**.
- GPU execution of training and the single final test: **pending T4**.
- Tests: 38 passing locally, all GPU-free.
- Resume eligible: no — no measured result exists, and the explanation gate in
  `docs/LEARNING_GUIDE.md` has not been attempted.

Verified development split:

- train: 8,000, exactly 2,000 per class, digest
  `0ec701367f1111d94a659335a9c3e811683a407e32350a4865e53f43bdfeaa5d`;
- validation: 2,000, exactly 500 per class, digest
  `bd9922811b0418edba481a1f73fede5a202f934133ebac6a0cf866bdb2143c7c`;
- publisher test: 7,600 declared by the pinned dataset, not loaded by the
  development evidence command.

## Correction to the frozen first half — August 7, 2026

The first T4 attempt crashed in `score_class_codes`. Current transformers return
a `BatchEncoding` from `apply_chat_template(tokenize=True)`, not a flat list of
token IDs. Three consequences, all now fixed by `prompts.prompt_token_ids`,
which normalizes list, dict, and batch-of-one returns:

1. baseline scoring raised `ValueError` in `tokenizer.pad`;
2. `encode_supervised_example` would have built training sequences out of dict
   *keys* rather than tokens;
3. the "prompt exceeds `max_sequence_length`" guard compared the length of a
   two-key mapping against 512, so it could never fire.

This changes no frozen fact — same prompt wording, same token IDs, same 512
limit. It only makes the code produce what the protocol already specified. Tests
now cover all three tokenizer return shapes and the oversized-prompt rejection.

Open question for the rerun: `docs/evidence/token-length-audit.json` (median
127, p95 160, max 432) predates this fix. The live guard in
`encode_supervised_example` now re-checks every row against 512 during training,
so a stale audit cannot let a truncated row through silently.

## Second correction — the trainable percentage, August 7, 2026

`outputs/qlora-setup.json` from the first T4 run records
`trainable_percent: 1.1036754331979965`. **Do not quote that number.** It counts
bitsandbytes `Params4bit` tensors by `numel()`, which returns stored elements —
two 4-bit weights live in one byte — so the denominator is roughly half the real
parameter count. That is why the artifact's `total_parameters` reads
3,800,305,664 for a model with about 7.25 billion parameters.

The numerator is sound: 41,943,040 trainable LoRA parameters, held in fp16 and
not packed. Against Mistral 7B's real 7,248,023,552 parameters that is
**0.5787%**. `parameter_report` now unpacks 4-bit tensors, reports the raw
`numel()` sum separately as `stored_tensor_elements`, and a test fails if the
packed count is ever used as the denominator again.

The recorded artifact is left exactly as the GPU produced it. The corrected
count will appear in `outputs/training-report.json` on the next run.

## Scoring optimization — August 7, 2026

`score_class_codes` now asks the model to apply the LM head to the final
position only (`logits_to_keep=1`, or `num_logits_to_keep` on older
transformers), instead of computing a `[batch, 512, 32768]` logit tensor and
discarding all but one row. The kwarg is discovered by walking the PEFT wrapper
chain; if no version supports it the call falls back to full logits, and the
returned last-position logits are identical either way.

This is a compute path change made *after* `outputs/base-validation.json` was
recorded, so the phase-2 re-score of the base model may differ from macro-F1
0.7299 in the last few decimals. `check_baseline_agreement` allows 0.005, which
absorbs kernel-level differences while still catching a changed prompt,
tokenizer, or model revision. If the re-score disagrees by more than that, the
run aborts before training rather than reporting an unexplained shift.

## What runs next, in order

1. `notebooks/loraforge_t4.ipynb` on a Colab/Kaggle T4 → `outputs/base-validation.json`
   and `outputs/qlora-setup.json`, restored into the repository.
2. `notebooks/loraforge_t4_phase2.ipynb` → training, `outputs/training-report.json`,
   `adapters/selected/`, `outputs/frozen-selection.json`, then one
   `outputs/final-test-report.json`.
3. `loraforge verify` locally, reproducing every number from the stored logits.
4. Only then: fill results into the README, and only after the oral gate does
   anything reach the résumé or the site.

## Artifacts and who writes them

| Artifact | Written by | Exists |
|---|---|---|
| `docs/evidence/*.json` | phase-1 local commands | yes |
| `outputs/base-validation.json` | phase-1 notebook (T4) | no |
| `outputs/qlora-setup.json` | phase-1 notebook (T4) | no |
| `outputs/training-report.json`, `adapters/` | `loraforge train` (T4) | no |
| `outputs/frozen-selection.json` | `loraforge freeze-selection` | no |
| `outputs/final-test-report.json` | `loraforge final-test` (T4, once) | no |
