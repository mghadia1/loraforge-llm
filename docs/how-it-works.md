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
Row IDs retain their publisher split and source index for provenance, while
leak detection separately hashes the normalized text actually shown to the
model. The latter hash has no split namespace, so duplicate content is rejected
even across publisher train and test splits or when whitespace differs.

The default loader asks Hugging Face for `split="train"` only; it does not even
request the publisher test split. A caller must explicitly set `allow_test=True`,
which makes a separate `split="test"` request after checkpoint and temperature
selection are frozen.

The validation window can also be pinned independently of the training count.
The validation-only expanded-data preset keeps the original 500 rows per class
at offsets 2,000–2,499, excludes them from training, and grows training to
4,000 rows per class using the next deterministic candidates. One epoch over
16,000 rows has the same 1,000 optimizer-step budget as two epochs over the
original 8,000 rows. Its config disables publisher-test evaluation, so any
comparison must remain on the unchanged validation split.

## Prompt loss

The supervised sequence contains the system instruction, article, assistant
boundary, one answer-code token, and EOS. Labels are `-100` for every prompt
token, so the causal-LM loss trains only the answer code and EOS. Training on
the prompt itself would waste capacity and blur what the classification loss
means.

A tokenizer-only audit found four original-development prompts above 384 tokens
and a maximum of 432. The frozen maximum is therefore 512; none of the original
10,000 development articles are silently truncated. The expanded-data preset
has its own reproducible train-only audit over all 18,000 development rows. The
artifact binds its summaries to the config, split row IDs, tokenizer revision,
and ordered per-row token lengths, while recording that publisher test was not
loaded.

## QLoRA setup

The base model is loaded in 4-bit NF4 with double quantization and FP16 compute.
Its weights stay frozen. PEFT adds rank-16 `A` and `B` matrices to attention and
MLP projection layers. The audit recomputes trainable and total parameters and
raises if any trainable parameter name is not a LoRA adapter.

Experiment JSON is parsed as an exact typed schema before any model or dataset
work starts. Missing and unexpected fields are rejected instead of inheriting
silent dataclass defaults, JSON booleans cannot stand in for integer counts,
and every nested object and target-module array keeps its declared shape.

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
it cannot improve macro-F1. Metric computation rejects non-finite or wrongly
shaped logits, invalid labels/probabilities, non-positive calibration-bin counts,
and invalid temperature-search bounds instead of emitting plausible numbers.
NLL uses max-shifted log-sum-exp arithmetic, so an extremely unlikely true
class retains its full finite loss instead of being capped by a clipped,
underflowed softmax probability.

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
logits down to per-class precision/recall/F1 and every calibration bin, logits
are confined to repo-relative paths under the evidence root and checked against
both their recorded array shape and SHA-256. Adapter directory references are
also required to be repo-relative, and their canonical paths — including any
directory symlinks — must remain under the evidence root before reports-only
verification, strict hashing, or model loading. The selected epoch is re-derived
from the rule rather than trusted. The training report's embedded experiment
config is parsed through the complete typed schema before dataset loading, and
its model identity, development row counts, epoch sequence, test-lock state,
and selection rule must agree with that config. Epoch checkpoints are checked
against their exact directory manifests; payload files and subdirectories must
be real entries rather than symlinks, so no executable weight can escape the
evidence root while still passing a same-content hash check. For a distributed selected adapter, every recorded
configuration and weight file remains hash-checked even if its Hugging Face
`README.md` changes as distribution metadata. Locally generated model cards are
written under `outputs/`, outside the adapter payload.
The CLI loads each required pinned publisher split once and shares its ordered
validation and test labels across every verifier stage, so a full reports-only
check does not repeatedly reopen the same held-out data.
An explicit `verify --training-only` scope ignores existing final-test and
interval artifacts, loads publisher train only, and verifies the validation
metrics, checkpoint selection, and adapter manifests without requesting the
publisher test split.
The verifier also re-fits both temperatures from the hash-checked validation
logits, checks the frozen calibration metrics, and proves that the final report
used those validation-fitted values. It also requires the final report's model
revision, selected epoch, complete adapter manifest, and experiment config to
match the validated training and frozen-selection evidence. The publisher-test
row count is always checked; when the verifier loads the pinned dataset, it also
checks the ordered row-ID digest. The one-run counter must be an exact JSON
integer, the frozen gate must record the same test-consumption timestamp as the
final report, and every reported accuracy, macro-F1, and calibrated-ECE delta is
re-derived from the two verified system blocks. Editing any stored metric,
provenance field, delta, or calibration temperature by hand makes verification
fail — which is the point. The GPU-free tests include exactly those tamper cases.

Generated model cards quote measured run values only from the run reports. If
there is no artifact for a classical baseline or annotated error analysis, the
card states that limitation instead of inserting a remembered estimate or an
unsupported explanation. Before rendering, the generator runs reports-only
training verification against publisher-train validation labels. Held-out
numbers are quoted only when the final report's exact SHA-256 is bound by the
tracked intervals evidence, so model-card generation never needs to reopen the
publisher test split.

The schema-v2 intervals report is also bound to the exact final-report SHA-256.
Verification recomputes its entire deterministic tree, including the scope,
zero-new-evaluations claim, bootstrap settings and no-improvement count, paired
counts, and McNemar statistics. The continuity correction clamps equal fixed
and broken counts to a zero chi-square rather than squaring a negative adjusted
difference. Standalone interval commands also require an exact integer one-run
count and a fully typed embedded config before any dataset load. Extremely small p-values carry a scientific
string so float underflow can never turn them into a false zero.

## The one test evaluation

`frozen-selection.json` is written before the test split is ever loaded. It
pins the selected adapter, its hashes, the validation metrics, and the two
validation-fitted temperatures. The final command refuses to run unless that
file exists, refuses if the adapter's hash has changed since, refuses without an
explicit confirmation string, and refuses to overwrite an existing final report.
It also revalidates the experiment config at the entry point: the test budget
must be the JSON integer `1` (not boolean `true`), and `resume_eligible` must
remain `false` until the separate explanation gate passes.

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
