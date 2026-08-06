# How the LoRAForge first half works

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

The publisher test split is absent unless a caller explicitly requests it. It
must be loaded only after checkpoint and temperature selection are frozen.

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

No local claim says this fits a T4. That becomes evidence only after the T4
notebook records the GPU model, allocated/peak memory, and trainable percentage.

## Evaluation boundary

The untuned and tuned systems use identical prompts and A–D scoring. Validation
selects the adapter and fits a temperature for each system. The one final test
run reports macro-F1, per-class metrics, NLL, ECE before/after temperature, and
invalid-output rate. Temperature scaling cannot change argmax predictions, so
it cannot improve macro-F1.
