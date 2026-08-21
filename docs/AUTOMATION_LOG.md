# LoRAForge automation log

- 2026-08-18 09:05 MST — Reverified `codex/loraforge-lock-test-loading` at `7410122` (56 tests, reports-only verifier, diff check, and GitHub `unit` check passed); draft PR creation remains blocked by GitHub app 403 and invalid `gh` authentication, so no new code or evidence was committed.
- 2026-08-18 12:58 MST — Retried draft PR creation for `codex/loraforge-lock-test-loading`; app 403, invalid `gh` authentication, and a signed-out browser still block submission, so the compare page was left open for sign-in and no repository state changed.
- 2026-08-19 09:10 MST — Added and verified a pinned train-only 18,000-row token-length audit for the expanded-data preset (124 tests passed; zero rows over 512; audit `test_loaded` is false); no GPU training or test evaluation ran.
- 2026-08-20 09:00 MST — Updated reports-only verification to load each required pinned publisher split once and share validation/test labels across all evidence stages while preserving test-row provenance checks; 126 tests and the reports-only evidence verifier passed.
- 2026-08-21 10:48 MST — Hardened config and final-test gates so JSON booleans cannot masquerade as test budgets or schema versions, and `resume_eligible` cannot be enabled before the explanation gate.
