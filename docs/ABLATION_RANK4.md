# Rank-4 ablation — plan

**Question.** The frozen run used rank 16. Does this adaptation actually need 16
directions, or is rank 16 over-provisioned?

**Prediction, stated before running.** AG News adaptation is behavioral, not
knowledge-bound: the base model already reads the articles, and training taught a
labelling convention plus an output format. If that change is genuinely
low-dimensional, rank 4 should land within noise of rank 16. The frozen run's
validation macro-F1 was 0.9310; the tuned test CI width was 0.011, so "within
noise" means roughly ±0.01.

Recording the prediction first is the point. If rank 4 matches, the finding is
that the frozen run was over-provisioned by 4x. If it drops clearly, the finding
is that the adaptation needed more directions than expected. Either is a result;
neither is a failure.

## What varies, and what must not

`configs/experiment-rank4.json` differs from `configs/experiment.json` in exactly
three fields, enforced by `tests/test_ablation_config.py`:

| field | frozen | ablation | why |
|---|---:|---:|---|
| `lora.rank` | 16 | 4 | the variable under test |
| `lora.alpha` | 32 | 8 | **must** move with rank |
| `test_evaluations_allowed` | 1 | 0 | the single test run is spent |

**Why alpha moves.** LoRA scales its update by `alpha / rank`. Holding alpha at 32
while dropping rank to 4 would raise that scale from 2 to 8 — a four-fold larger
update. A weaker result would then be unattributable: too few directions, or too
large a step? Scaling alpha to 8 holds the ratio at 2 so rank is the only change.

Everything else is identical, including seed 73, the same 8,000 training and
2,000 validation rows, two epochs, learning rate 2e-4, effective batch 16, and
`max_sequence_length` 512.

## Running it

```bash
mkdir -p runs/rank4/outputs
cp outputs/base-validation.json runs/rank4/outputs/      # same base model, same prompt
loraforge train --config configs/experiment-rank4.json --root runs/rank4
```

The separate `--root` keeps the frozen artifacts untouchable. Training re-scores
the untuned base with the adapter disabled and aborts unless it reproduces the
phase-one baseline, which is why that file must be copied across.

The run still requires a human-started T4. This protocol does not assume that a
lower LoRA rank produces a proportional runtime or memory reduction.

## Reading the result

Compare **validation** macro-F1 against 0.9310. The test split is not available to
this arm and must not be used.

- **Within about ±0.01** — rank 16 was over-provisioned. The claim becomes
  "matched the validation result with one quarter of the LoRA rank."
- **Clearly lower** — the adaptation needed more than four directions. Report the
  gap; that bounds the intrinsic dimensionality of this task from below.
- **Clearly higher** — treat with suspicion and check for a confound before
  believing it; less capacity should not help.

`resume_eligible` stays false. This arm changes nothing about the explanation gate.

---

# Rank-4 ablation — result

The reports record an August 18, 2026 Colab Tesla T4 run. `loraforge
compare-runs --strict` confirms the hash-checked validation evidence, exact
ordered validation rows, and checkpoint selection. It schema-validates that the
two reports record the intended config difference, the same GPU, and the same
installed-library versions. Those control fields are self-reported and are not
independently corroborated. Strict mode therefore calls them matching recorded
controls, not proof that the experiment was controlled in every unobserved way.

| | rank 16 | rank 4 |
|---|---:|---:|
| validation macro-F1 (epoch 2) | 0.9310 | **0.9360** |
| validation macro-F1 (epoch 1) | 0.9248 | 0.9182 |

Parameter counts, adapter bytes, wall time, and peak memory are intentionally
omitted. Strict comparison verifies the validation/logit evidence, but these
resource values exist only in mutable run reports for this pair and are not
independently bound.

## The difference is not distinguishable from noise

Paired bootstrap over the 2,000 validation rows, 2,000 resamples, seed 73:

- difference (rank 4 − rank 16): **+0.0050**
- 95% confidence interval: **[−0.0023, +0.0122]** — spans zero
- rank 4 failed to beat rank 16 in 199 of 2,000 resamples
- McNemar on the paired predictions: **p = 0.220** over 54 discordant pairs

The two adapters disagree on **56 of 2,000 rows**: rank 4 is right on 32 of them,
rank 16 on 22. They have learned very nearly the same function.

**So the finding is equivalence, not superiority.** Reporting "rank 4 beat rank 16"
would be claiming a ten-row difference on a 2,000-row split as a result.

## What this means

The result is consistent with the prediction recorded before the run: rank 4
produced statistically indistinguishable validation quality with one quarter of
the LoRA rank setting. It does not independently establish that rank was the only
causal difference, because the historical config, GPU, and package metadata have
no second evidence source.

Two things this ablation does *not* show:

- **No sole-cause claim is supported.** The configs pass the exact typed schema,
  but the matching config and environment fields remain self-reported provenance.
- **No runtime or memory conclusion is supported.** Those measurements are not
  independently bound for both runs, so the comparison omits them.
- **This does not generalize to other tasks.** AG News adaptation is behavioural —
  a labelling convention and an output format on top of comprehension the base model
  already had. A task requiring knowledge the base model lacks should need more
  directions, and this result says nothing about where that boundary sits.

`resume_eligible` remains false.
