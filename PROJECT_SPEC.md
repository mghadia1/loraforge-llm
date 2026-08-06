# LoRAForge — Parameter-Efficient Fine-Tuning of a Real LLM (QLoRA)

> **Build spec, August 6, 2026.** Your third and strongest ML project: efficiently fine-tune
> an actual billions-parameter open LLM on a single free GPU, with the same honest evaluation
> discipline as TextForge and PaperTrail. Targets the exact 2026 skill employers name — "LLM
> specialists who optimize transformer models and manage parameter counts." Name is a
> placeholder (LoRAForge, TuneLab, ...).

**Honesty gate (same as every project here):** nothing on the resume/site/application until
Mayank can explain, unaided, what LoRA/QLoRA does, one design choice, and one failure mode.
Real public data + real open model only. No claimed numbers until measured. `resume_eligible:
no` until the gate closes.

---

## 1. Why this is a tier above TextForge

| | TextForge | LoRAForge |
|---|---|---|
| Model size | DistilBERT, 66M params | Llama 3 8B / Mistral 7B — **~8B params** |
| Method | full fine-tune (all weights) | **LoRA/QLoRA** — train ~0.1–1% of params |
| Compute | fits easily | 4-bit quantization to fit **8B on one free GPU** |
| Skill signalled | "I can fine-tune a model" | "I can adapt a real LLM efficiently under memory limits" |

The second row is the whole point. Full fine-tuning an 8B model needs ~60+ GB of GPU memory.
**QLoRA** loads the base model in 4-bit and trains small low-rank adapter matrices, cutting
that to what a single free Colab/Kaggle T4 (16 GB) can hold. Being able to explain *why that
works* is a genuinely senior signal.

## 2. What it does (one sentence)

Take a pretrained open LLM, fine-tune it with QLoRA on a real instruction/classification
dataset so it beats the base model on a held-out task, and evaluate honestly — including
calibration and a base-vs-tuned comparison.

## 3. The real ML you must be able to explain

- **LoRA:** instead of updating a weight matrix `W`, you freeze it and learn a low-rank update
  `W + BA` where `B` and `A` are tiny (rank `r`, e.g. 8–16). You train `A` and `B` only — a
  fraction of the parameters — then optionally merge them back.
- **QLoRA:** load the frozen base model in **4-bit** (NF4 quantization) to save memory; keep
  the LoRA adapters in higher precision. This is what makes 8B fit on 16 GB.
- **Why it works:** fine-tuning mostly needs *small* corrections to a strong pretrained model;
  a low-rank update captures those, so you don't need to move all 8B weights.
- **Evaluation:** a base-model baseline (same prompts, no fine-tuning) vs the tuned adapter,
  measured on a held-out set — so the LoRA gain is proven, not assumed. Reuse TextForge's
  metric + calibration rigor.

## 4. Task options (pick one — keep the eval clean)

1. **Instruction/classification** (easiest to evaluate honestly): fine-tune for a structured
   task with a clear metric — e.g. multi-class intent/topic classification phrased as
   instructions. Metric: macro-F1 vs the base model, plus calibration.
2. **Domain style/format adherence:** teach the model a specific output format (e.g. always
   return valid JSON for a schema); metric: % valid outputs, base vs tuned.

Recommendation: **option 1** — it gives a crisp, defensible number and reuses your existing
evaluation code almost directly.

## 5. Architecture / flow

```
   Real public dataset (Hugging Face Datasets), fixed split + seed
                     │
                     ▼
   Base model in 4-bit (bitsandbytes NF4)  ── Llama 3 8B or Mistral 7B
                     │  attach LoRA adapters (peft): rank r, target attn/MLP proj
                     ▼
   Train adapters only (Hugging Face Trainer / TRL SFTTrainer) on free GPU
                     │  early-stop / select on validation metric
                     ▼
   Frozen adapter  ── evaluate ONCE on held-out test:
        base-model baseline vs tuned:  macro-F1, per-class, calibration (ECE)
                     │
                     ▼
   Save adapter (tiny, ~10–100 MB) + report + reliability table
        (optional) merge adapter → serve via /generate
```

## 6. Tech stack (all free, all standard)

- **Core:** `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit), `trl` (SFTTrainer),
  `datasets`, `accelerate`, PyTorch
- **Model:** Llama 3 8B Instruct or Mistral 7B (gated on HF — accept the license; both free)
- **Compute:** free Kaggle T4 (16 GB, ~30 hr/week) or Google Colab — QLoRA fits
- **Eval:** reuse TextForge's `metrics.py` + `calibration.py` patterns (macro-F1, ECE)
- **Ops:** pytest (adapter config, prompt formatting, metric math — no GPU needed), CI,
  `README`, `docs/how-it-works.md`, and the "read before you claim it" doc

New skills here: LoRA/PEFT, 4-bit quantization, instruction fine-tuning, working within GPU
memory limits — the headline LLM-engineering skills of 2026.

> **Compute reality:** you cannot train this in the sandbox or on a Mac CPU. Use a free
> Kaggle/Colab GPU notebook for the training step; keep the repo (code, configs, eval, saved
> adapter, reports) as the deliverable. The adapter is small enough to commit or release.

## 7. Build order

1. **Data + prompt format.** Load a real dataset, pin split/seed, define the exact
   instruction template. *Deliverable:* dataset stats + 3 formatted examples.
2. **Base-model baseline.** Run the *un-tuned* model on the val/test prompts, measure macro-F1.
   *Deliverable:* the number QLoRA must beat.
3. **QLoRA setup.** Load base in 4-bit, attach LoRA adapters (`r`, `alpha`, target modules,
   dropout), confirm only adapters are trainable (print trainable-param %). *Deliverable:* a
   config that fits in 16 GB.
4. **Train.** SFTSFT/Trainer on the free GPU, select the adapter on validation. *Deliverable:*
   saved adapter + train/val curves.
5. **Evaluate once.** Base vs tuned on held-out test: macro-F1, per-class, **ECE/calibration**.
   *Deliverable:* results table + honest analysis (where it helped, where it didn't).
6. **Harden.** Tests (GPU-free: config, prompt formatting, metric/calibration math), CI,
   README, `how-it-works.md`. *Deliverable:* repo matching your others.
7. **Optional.** Merge adapter and serve `/generate` via FastAPI + Docker (your MLOps step).

## 8. Honest metrics to claim (fill after measuring)

- Base model + dataset + split sizes; instruction template.
- **Trainable parameters as a % of total** (the LoRA headline, e.g. "0.2% of 8B").
- Peak GPU memory used (proves it fit on one consumer/free GPU).
- Base-model macro-F1 vs QLoRA-tuned macro-F1 on held-out test; the delta.
- Calibration (ECE) before/after — reuse your TextForge module.
- Training time + cost (keep it $0 on free GPU).

## 9. Resume bullets — TEMPLATE (fill after measuring)

> Built **LoRAForge**: fine-tuned **[Llama 3 8B / Mistral 7B]** with **QLoRA** (4-bit NF4
> base + rank-[r] LoRA adapters, **[X]% of params trainable**) on a single **free [T4] GPU**,
> lifting held-out macro-F1 from **[base]** to **[tuned]** over the un-tuned base model.

> Kept the evaluation honest: base-vs-adapter comparison on a frozen test split, plus
> temperature-scaled calibration (ECE **[before]→[after]**); adapter is **[size] MB**, tests
> green in GitHub Actions.

## 10. Guardrails

- Real public dataset + real open model only; pin split, seed, and adapter config.
- Always report the **base-model baseline** — if QLoRA barely beats it, that's a finding.
- Evaluate on test **once**, after selecting the adapter on validation.
- Watch for **eval leakage**: make sure your instruction template doesn't leak the label, and
  that the base-model baseline uses the identical prompt.
- `resume_eligible: no` until the explanation gate closes.

## 11. First session checklist

1. On Kaggle/Colab: `pip install transformers peft bitsandbytes trl datasets accelerate`.
2. Load the base model in 4-bit and print trainable-param % after attaching LoRA — prove the
   memory path fits before any training.
3. Then follow build order 1→6. Ask me to scaffold the data/prompt + baseline first (GPU-free
   parts I can help write and test here).
