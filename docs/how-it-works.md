# How LoRAForge works

## Why four class codes

AG News has four topics. The prompt lists a fixed mapping from topics to `A–D`
and asks for one code only. Under Mistral's real chat template, each code is one
contextual token. A single model forward pass therefore yields four comparable
next-token logits. Softmax over those logits creates a conditional distribution
over the allowed classes, supporting macro-F1, NLL, confidence, and ECE without
77 separate candidate sequences or fuzzy output parsing.

The contextual detail matters. Mistral has multiple vocabulary IDs that decode
to the same letter. LoRAForge derives IDs after the actual `[/INST]` boundary
and fails if appending a code is not exactly one token. It also tokenizes the
chat template directly, avoiding a duplicated beginning-of-sequence token.

## Data protocol

The publisher provides 120,000 train rows and 7,600 test rows. LoRAForge pins a
specific dataset commit. Within each class, publisher-train rows receive stable
row IDs and are ordered by a seed-bound SHA-256. The first 2,000 per class form
training; the next 500 form validation. This yields balanced, disjoint, exactly
reproducible 8,000/2,000 subsets without depending on a library RNG version.

The default loader asks Hugging Face for `split="train"` only; it does not even
request the publisher test split. A caller must explicitly set `allow_test=True`,
which makes a separate `split="test"` request after checkpoint and temperature
selection are frozen.

## Prompt loss

The supervised sequence contains the system instruction, article, assistant
boundary, one answer-code token, and EOS. Labels are `-100` for every prompt
token, so the causal-LM loss trains only the answer code and EOS. Training on
the prompt itself would waste capacity and blur what the classification loss
means.

A tokenizer-only audit found four development prompts above 384 tokens and a
maximum of 432. The frozen maximum is therefore 512; none of the 10,000
development articles are silently truncated.

## QLoRA setup

The base model is loaded in 4-bit NF4 with double quantization and FP16 compute.
Its weights stay frozen. PEFT adds rank-16 `A` and `B` matrices to attention and
MLP projection layers. The audit recomputes trainable and total parameters and
raises if any trainable parameter name is not a LoRA adapter.

The completed run used a Tesla T4. The phase-one setup reached 12.34 GiB peak
CUDA allocation; the training report records 5.77 GiB peak after its own memory
reset. The selected adapter contains 41,943,040 trainable LoRA weights and is
167,838,575 bytes on disk. The original setup artifact's packed-4-bit
denominator error is preserved and explained in `results.md`.

## Evaluation boundary

The untuned and tuned systems use identical prompts and A–D scoring. Validation
selects the adapter and fits a temperature for each system. The one final test
run reports macro-F1, per-class metrics, NLL, ECE before/after temperature, and
invalid-output rate. Temperature scaling cannot change argmax predictions, so
it cannot improve macro-F1.

## Training the two frozen epochs

`build_supervised_features` encodes each training row and asserts that exactly
two tokens — the answer code and EOS — escape the `-100` mask. `pad_batch`
right-pads a batch: inputs with the pad ID, attention with `0`, labels with
`-100`, so padding contributes nothing to the loss either. This is deliberately
not an off-the-shelf SFT text path, which would compute loss over the prompt.

Before the first optimizer step, the trainer re-scores validation with the
adapter switched off and refuses to continue unless it reproduces the phase-one
baseline macro-F1. That single check catches a changed prompt, tokenizer, or
model revision between the two GPU sessions — the failure that would otherwise
turn into a fake improvement.

After each epoch the adapter is saved, hashed, and scored on validation. The
selection rule is one line: highest validation macro-F1, and an exact tie goes
to the earlier epoch. Both checkpoints, their validation logits, the loss curve,
wall time, peak CUDA memory, package versions, and adapter hashes land in
`outputs/training-report.json`.

## Why the evidence is hashed

Every reported number is recomputable from raw logits stored next to it, and
`loraforge verify` recomputes them. Metrics are re-derived from the `.npy`
logits, logits are checked against their SHA-256, adapters are checked against
their directory hash, and the selected epoch is re-derived from the rule rather
than trusted. Editing a macro-F1 by hand in a report makes verification fail —
which is the point. The GPU-free tests include exactly those tamper cases.

## The one test evaluation

`frozen-selection.json` is written before the test split is ever loaded. It
pins the selected adapter, its hashes, the validation metrics, and the two
validation-fitted temperatures. The final command refuses to run unless that
file exists, refuses if the adapter's hash has changed since, refuses without an
explicit confirmation string, and refuses to overwrite an existing final report.

Both systems are scored from the same loaded model — adapter disabled for the
base, enabled for the tuned — so the prompt, tokenizer, and quantization are
identical by construction. Because decoding is constrained to the four class
logits, the invalid-output rate is 0 by construction; that is a property of the
scoring design, not a result. A failure during test scoring is written to
`outputs/failed-attempts/` rather than retried silently, and a negative delta is
reported exactly like a positive one.

## Measured outcome

Epoch 2 won on validation with 0.9310 macro-F1. In the single 7,600-row
publisher-test evaluation, the untuned base scored 0.7262 macro-F1 and the
selected adapter scored 0.9333, a +0.2071 difference. The final verifier
recomputed both results from stored logits and matched their recorded hashes.
Calibration reduced tuned ECE from 0.0290 to 0.0078 without changing a single
prediction. Full results and limitations are in `docs/results.md`.
