# Claude Code handoff — LoRAForge second half

You own Build Steps 4–7 only. Do not redo or silently change the frozen first
half. Start only after Mayank runs `notebooks/loraforge_t4.ipynb` on a T4 and
places these files in the repository:

- `outputs/base-validation.json`
- `outputs/qlora-setup.json`

If either is absent, stop after GPU-free implementation and tell Mayank exactly
which notebook cells remain. Never invent their numbers.

## Read before editing

Read, in full:

1. `PROJECT_SPEC.md`
2. `README.md`
3. `docs/how-it-works.md`
4. `docs/evidence/data-stats.json`
5. `docs/evidence/formatted-examples.json`
6. `docs/evidence/token-length-audit.json`
7. `configs/experiment.json`
8. every module in `src/loraforge/`
9. every test
10. the T4 notebook and both returned output files

Run `python -m pytest -q` before changing anything.

## Frozen facts you may not tune

- Dataset/model names and exact revisions.
- Seed 73; 2,000 train + 500 validation examples per class.
- Publisher test contains 7,600 rows and remains locked.
- Prompt wording and `A–D` mapping.
- `max_sequence_length=512`.
- 4-bit NF4/double-quant/FP16 configuration.
- LoRA rank 16, alpha 32, dropout 0.05, and target modules.
- Two epochs, learning rate `2e-4`, effective batch 16, validation macro-F1
  checkpoint selection.
- Exactly one final publisher-test evaluation.

If a frozen configuration cannot run on T4, preserve the failure artifact and
propose a new protocol version. Do not edit the original evidence until it
appears successful.

## Step 4 — training

Implement a tokenized dataset using `encode_supervised_example`. The loss must
see only the assistant code token and EOS; prompt labels stay `-100`. Add a
padding collator that pads labels with `-100` and inputs with the tokenizer pad
ID. Do not use an SFT path that computes loss over the user prompt.

Train the two frozen epochs. Save epoch checkpoints, optimizer/training state,
loss curve, wall time, GPU name, allocated/peak memory, package versions, and
adapter hashes. After each epoch, run the existing efficient class scorer over
validation and record macro-F1. Select the higher validation macro-F1; define a
deterministic earlier-epoch tie break. Write `outputs/training-report.json` and
copy/freeze the winner under `adapters/selected/`.

## Step 5 — one final evaluation

Before accessing test, write `outputs/frozen-selection.json` containing selected
checkpoint, hashes, validation metrics, base/tuned validation-logit hashes, and
separate validation-fitted temperatures. Refuse to proceed if this file or the
selected adapter is missing.

Implement a final command requiring an explicit confirmation flag. It must:

1. refuse to overwrite an existing final report;
2. load publisher test once;
3. score the untuned base and selected adapter with identical prompts;
4. report macro-F1, accuracy, per-class precision/recall/F1, confusion matrix,
   NLL, ECE before/after each system's validation-fitted temperature, latency,
   and invalid-output rate;
5. assert temperature scaling leaves argmax predictions unchanged;
6. record raw-logit/prediction hashes and all provenance;
7. honestly report the delta even if QLoRA loses.

Do not choose anything from test. Preserve failed attempts separately.

## Step 6 — harden

Add GPU-free tests for the collator, masked loss, checkpoint selection/tie break,
frozen-selection verifier, single-test guard, metric recomputation, and
hand-edited evidence rejection. Keep CI GPU-free. Update README/results/status
only from real artifacts. Add a learning guide and explanation gate; keep
`resume_eligible: no` until Mayank passes it.

## Step 7 — optional

Only after Steps 4–6 pass, consider merging/serving. Do not let optional FastAPI
or Docker work delay the measured report. Never describe a merged model as
4-bit unless the deployed artifact is actually quantized and verified.

## Definition of done

- Real adapter and curves saved from T4.
- One frozen base-vs-tuned test report with recomputable metrics.
- Validation-only temperature fitting and test ECE before/after.
- Measured trainable percentage, peak GPU memory, time, and adapter size/hash.
- GPU-free tests and GitHub Actions green.
- README states limitations and any negative result.
- No résumé/site/application update before Mayank's oral gate.
