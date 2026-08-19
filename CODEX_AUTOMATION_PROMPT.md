# LoRAForge — Codex automation prompt (v2)

Scope: work **only** in `projects/loraforge-llm`. Goal: make **one bounded, technically
meaningful improvement per run** to the public LoRAForge QLoRA/transformer project — or, if none
exists, make no change and say so. A no-op run with an honest report is a **successful** run.

---

## Hard rules (never violate)

1. **No fabrication.** Never invent results, metrics, GPU runs, training evidence, or résumé
   claims. Every number must trace to a hash-backed artifact.
2. **Test-set integrity.** Never rerun, tune against, or repeatedly inspect the official held-out
   AG News test split. Model selection and calibration happen on validation only, under the
   frozen protocol. "Better" may be claimed only after a validation improvement under that
   protocol.
3. **Git safety.** Never rewrite history, force-push, push to `main`, merge, change repo
   visibility, publish secrets, or alter release evidence. Preserve uncommitted user edits
   (e.g. the current `src/loraforge/selection.py` change) — do not clobber them.
4. **No activity-only work.** Never create empty/backdated commits, split one logical change into
   artificial commits, or lower quality to influence the contribution graph.
5. **resume_eligible stays false** until Mayank separately passes the explanation gate. A code
   change is not a résumé claim.
6. **GPU is human-started.** Never start expensive GPU training locally or drive Colab. Prepare
   notebooks/commands so Mayank can run a T4; do not claim a run happened without artifacts.

---

## Start of every run

1. Read: `README.md`, `docs/STATUS.md`, `docs/results.md`, `docs/how-it-works.md`,
   `configs/experiment.json`, `docs/AUTOMATION_LOG.md` (create it if missing — this is your
   memory), the tests, and `git status`.
2. `git fetch origin`; base new work on latest `origin/main`. List open `codex/*` branches and
   PRs so you don't duplicate in-flight work.
3. Decide the single best change for this run (next section). If none clears the value bar, stop
   and write the report — that is a valid outcome.

## Choosing the one change

Pick the **highest-value** improvement available, using this priority order:

1. Correctness / data leakage / provenance / licensing bugs.
2. Regression tests, deterministic evaluation, evidence verification.
3. Adapter loading, inference, prompt/tokenization validation, transformer-understanding depth.
4. Reproducible T4 Colab workflow, runtime/memory measurement, experiment config.
5. Validation-only calibration, error analysis, model-selection safeguards, measured limits.
6. Documentation — **only** when it describes already-implemented, verified behavior.

Use the weekly rotation (Mon dataset/leakage, Tue tests/determinism, Wed inference/tokenization,
Thu Colab/runtime, Fri calibration/error analysis, Sat review/CI feedback, Sun dataset research)
**only as a tie-breaker** when two candidates are equally valuable. Never do lower-value on-theme
work over a higher-value off-theme fix.

**Reject as non-improvements:** cosmetic edits, reformatting, comment churn, doc changes not tied
to implemented behavior, or anything you cannot independently explain.

## Dataset discovery (Sunday, or when a change genuinely needs data)

Do **not** search for datasets on non-research runs. On Sunday (or when a concrete, tested change
requires new data):

- Prefer a controlled use of the **110,000 unused AG News publisher-train rows** over external
  data unless a candidate adds meaningful coverage.
- For each candidate verify: license/redistribution, source + version/revision, language, label
  compatibility with the A–D AG News contract, size/splits, quality, access stability, and
  duplicate/leakage risk against AG News validation and publisher-test content.
- Reject unclear licensing, unverifiable provenance, incompatible labels without a defensible
  mapping, or suspected contamination. Treat a different label ontology/domain as a **separate**
  experiment — never mix it into the completed experiment or overwrite its evidence.
- Record every candidate reviewed (selected or rejected, with reason) in `docs/AUTOMATION_LOG.md`
  so future runs don't re-review it. Discovery alone is **not** a reason to commit — only a
  bounded, tested change is (a download/audit script, schema/prompt validation, dedup/leakage
  checks, a versioned validation-only config, or a safe T4 notebook).

## Implement → verify → isolate

1. Create a fresh branch `codex/loraforge-<short-topic>` off `origin/main`. Reuse an existing
   `codex/*` branch/PR **only** if this is a direct follow-up to that exact change (review
   feedback or a failing check). Never add unrelated work to a broad automation branch.
2. Implement the change. Stage only files belonging to it.
3. Run the relevant tests **and** the evidence verifier. Inspect the diff.
4. Commit only if tests pass and the value is clear. 1–3 commits are fine only when naturally
   atomic (e.g. implementation / a distinct regression-test fix / docs of verified behavior).
5. Push the branch; open or update a **draft** PR describing exactly what was verified. Do not
   merge.

## End-of-run report (always)

Report concisely:
- Dataset candidates reviewed + selection/rejection reasons (or "none — not a research run").
- Problem addressed; branch name; files changed.
- Tests/verifier results.
- Commit + PR links.
- Any measured limitation.
- Whether a T4 Colab run is ready or still required.
- The single best next improvement.
- Exactly one status:
  - **READY TO MERGE** — scoped, locally verified, pushed, CI green, PR accurately describes the
    evidence.
  - **BLOCKED** — name the concrete blocker (missing review, failing test/CI, conflict,
    unavailable evidence, permission).
  - **NO CHANGE** — no improvement cleared the value bar this run; nothing committed.

Append a one-line summary of the run to `docs/AUTOMATION_LOG.md`.
Mayank reviews READY TO MERGE PRs and decides whether to merge.
