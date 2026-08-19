# Rank-4 ablation — plan

**Question.** The frozen run used rank 16 (41,943,040 trainable parameters). Does
this adaptation actually need 16 directions, or is rank 16 over-provisioned?

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

Expected cost: about 3.6 hours on a T4, the same as the frozen run — rank barely
affects step time, because the frozen base model dominates the compute.

## Reading the result

Compare **validation** macro-F1 against 0.9310. The test split is not available to
this arm and must not be used.

- **Within about ±0.01** — rank 16 was over-provisioned. The adapter drops from
  167 MB to roughly 42 MB at equal quality, and the claim becomes "matched the
  result with a quarter of the parameters."
- **Clearly lower** — the adaptation needed more than four directions. Report the
  gap; that bounds the intrinsic dimensionality of this task from below.
- **Clearly higher** — treat with suspicion and check for a confound before
  believing it; less capacity should not help.

`resume_eligible` stays false. This arm changes nothing about the explanation gate.

---

# Rank-4 ablation — result

Run August 18, 2026 on a Colab Tesla T4. `loraforge compare-runs --strict` confirms
the comparison is controlled: identical library stack, same GPU model, and the only
config differences are the two under test plus the removed test budget.

| | rank 16 | rank 4 |
|---|---:|---:|
| validation macro-F1 (epoch 2) | 0.9310 | **0.9360** |
| validation macro-F1 (epoch 1) | 0.9248 | 0.9182 |
| trainable parameters | 41,943,040 (0.578%) | **10,485,760 (0.1445%)** |
| adapter bytes | 167,838,575 | **42,008,469** |
| peak CUDA | 5.77 GiB | 5.48 GiB |
| wall time | 13,083.9 s | 14,487.9 s |

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

The prediction recorded before the run held: this adaptation is low-dimensional
enough that four directions carry it. **Rank 16 was over-provisioned by four times** —
the same task quality for a quarter of the trainable parameters and a quarter of the
adapter on disk.

Two things this ablation does *not* show:

- **Rank 4 is not faster to train.** It took 11% longer on the same GPU model. The
  frozen 7B base dominates step time, so trainable-parameter count barely moves the
  clock; that gap is shared-infrastructure variance, not a property of rank.
- **This does not generalize to other tasks.** AG News adaptation is behavioural —
  a labelling convention and an output format on top of comprehension the base model
  already had. A task requiring knowledge the base model lacks should need more
  directions, and this result says nothing about where that boundary sits.

`resume_eligible` remains false.
