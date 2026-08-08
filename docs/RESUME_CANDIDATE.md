# Résumé candidate — blocked pending explanation gate

Do not copy this into a résumé, portfolio, or application until Mayank answers
all twelve questions in `LEARNING_GUIDE.md` unaided. The wording below is
truthful and backed by verified artifacts; the remaining gate is ownership and
understanding.

## Project entry

**LoRAForge — Evidence-First QLoRA Adaptation**
Python, PyTorch, Transformers, PEFT, bitsandbytes, scikit-learn

- Fine-tuned Mistral-7B-Instruct with rank-16 QLoRA on 8,000 balanced AG News
  examples using a single Tesla T4; selected the checkpoint on a separate
  2,000-example validation set and improved held-out macro-F1 from 0.726 to
  0.933 on the 7,600-row publisher test split.
- Built a constrained four-token classification and calibration pipeline with
  assistant-only masked loss, validation-fitted temperature scaling, and
  hash-backed reports that recompute metrics from stored logits and reject
  edited evidence.

## Short application answer

I built LoRAForge to learn parameter-efficient fine-tuning without treating a
single benchmark score as proof. I adapted Mistral 7B with QLoRA on a free T4,
froze checkpoint and calibration choices on validation, and evaluated the
publisher test once. The selected adapter improved macro-F1 from 0.726 to
0.933. I also stored raw logits and adapter hashes so the repository can
recompute the result and reject altered reports. The most useful lesson was
that evaluation design—prompt parity, split discipline, and evidence checks—is
as important as getting the training loop to run.

## Claims intentionally excluded

- “Production-ready” or “state of the art.”
- General LLM reasoning or generation improvement.
- Multi-seed statistical significance.
- Full fine-tuning or CUDA-kernel implementation experience.
- Medical, robotics, or safety-critical performance.
