# Learning guide and explanation gate

Nothing from LoRAForge goes on the résumé, the portfolio site, or an application
until Mayank can answer every question in the gate below **unaided** — no notes,
no reading from this file. Passing tests is not the gate. Measured numbers are
not the gate. The gate is being able to defend the work in an interview.

## Numbers you must be able to explain

- Validation selected epoch 2: 0.9310 macro-F1.
- Publisher test: base 0.7262, tuned 0.9333, delta +0.2071.
- Test accuracy: 0.7428 to 0.9333.
- Tuned test ECE: 0.0290 before validation-fitted temperature scaling and
  0.0078 after.
- Two-epoch T4 training time: about 3 hours 38 minutes.
- Selected adapter: 167,838,575 bytes; 41,943,040 trainable LoRA weights.

Memorizing these values is insufficient. You must connect each number to the
artifact that produced it and explain why the protocol makes the comparison
credible.

## Read in this order

1. `docs/how-it-works.md` — the whole protocol in one pass.
2. `src/loraforge/prompts.py` — why a class is one token and why the prompt is
   masked out of the loss.
3. `src/loraforge/qlora.py` — what PEFT actually attaches and what the audit
   proves.
4. `src/loraforge/collate.py` — where `-100` comes from and what it does.
5. `src/loraforge/selection.py` and `src/loraforge/final_test.py` — why the test
   split is untouchable until the selection is frozen.

## The gate — answer out loud, in your own words

**LoRA and QLoRA**

1. What does LoRA change about a weight matrix `W`, and what stays frozen?
2. Rank 16 on a 4096-wide projection: roughly how many parameters is that per
   matrix, and why is it so much smaller than the matrix it adapts?
3. What does NF4 4-bit quantization save, and what is kept in higher precision?
4. Why can a low-rank update be enough for this task when full fine-tuning
   moves every weight?
5. Where does the memory actually go on a 16 GB T4 during training — name the
   three biggest consumers.

**This experiment specifically**

6. Why is the class predicted from four next-token logits instead of generating
   text and parsing it? Name one thing this buys and one thing it hides.
7. Why do the base and tuned systems have to use the identical prompt?
8. What is masked with `-100`, and what would go wrong if the prompt tokens were
   supervised too?
9. Why is the checkpoint chosen on validation and not on test?
10. Why is a temperature fit separately for the base and the tuned system, and
    why can temperature scaling never improve macro-F1?

**Design choice — pick one and defend it**

11. Name a decision that could reasonably have gone the other way (rank 16 vs 8,
    2,000 rows per class vs all 30,000, two epochs vs early stopping, constrained
    class logits vs free generation). Say what you chose, what you traded away,
    and what evidence would change your mind.

**Failure mode — name at least one real one**

12. A concrete way this pipeline could produce a good-looking number that is not
    real, and what in the repository is there to catch it. (Examples that count:
    picking the checkpoint on test; the tokenizer mapping a code to a different
    contextual token so the logits are not the class scores; the base model being
    silently prompted differently from the tuned one; reporting ECE improvement
    as an accuracy improvement.)

## Rules while learning

- Do not add a number to any document that you cannot point to in
  `outputs/*.json` and reproduce with `loraforge verify`.
- If a run fails, keep the failed artifact. A preserved failure is evidence; a
  quietly repeated run is not.
- `resume_eligible` stays `false` in `configs/experiment.json` until the gate is
  passed. Flipping it is a deliberate act, not a side effect.
