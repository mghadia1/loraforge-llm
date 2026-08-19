---
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
license: apache-2.0
language:
- en
datasets:
- fancyzhx/ag_news
tags:
- lora
- qlora
- text-classification
- peft
---

# QLoRA adapter for fancyzhx/ag_news topic classification

A rank-16 QLoRA adapter for [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), trained to classify news articles into World, Sports, Business, or Sci/Tech. The base model is frozen in 4-bit NF4; only the adapter was trained.

Source, evidence, and verification tooling: https://github.com/mghadia1/loraforge-llm

## Results

Selected on **validation macro-F1** at epoch 2 of 2. The untuned base model was scored with the identical prompt, so the comparison isolates the adapter.

| System | Split | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Untuned base | validation | 0.7475 | 0.7299 |
| **This adapter** | validation | **0.9310** | **0.9310** |
| Untuned base | held-out test (7,600 rows) | 0.7428 | 0.7262 |
| **This adapter** | held-out test (7,600 rows) | 0.9333 | 0.9333 |

The held-out split was evaluated **once**, after the checkpoint and calibration temperature were frozen on validation.

## Training

- Base model: `mistralai/Mistral-7B-Instruct-v0.3` at revision `c170c708c41dac9275d15a8fff4eca08d52bab71`
- Data: `fancyzhx/ag_news` at revision `eb185aade064a813bc0b7f42de02595523103ca4`, 8,000 training and 2,000 validation rows, balanced across four classes, selected deterministically with seed 73
- Quantization: 4-bit NF4, double quantization, float16 compute
- LoRA: rank 16, alpha 32, dropout 0.05, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Trainable parameters: **not recorded by this run (it predates the parameter audit)**
- Optimization: 2 epochs, learning rate 0.0002, effective batch 16, paged_adamw_8bit, 3% warmup
- Loss is computed **only** on the answer token and EOS; every prompt token is masked with `-100`
- Hardware: Tesla T4, 3.63 h, peak 5.77 GiB CUDA
- Adapter size: 167,838,575 bytes

## How the class is read

The prompt asks for one code — `A`=World, `B`=Sports, `C`=Business, `D`=Sci/Tech — and the prediction is the argmax over those four next-token logits in a single forward pass. Because decoding is constrained to the four codes, an unparseable output is impossible by construction, so a 0% invalid-output rate is a property of the scoring design and not a result.

## Limitations

- Trained and evaluated only on fancyzhx/ag_news: short English news headlines with a single topic label. Nothing here supports use on other domains, longer documents, or safety-critical decisions.
- One seed, one hardware run. Test-set sampling uncertainty has been quantified for the reference run; training variance across seeds has not been measured.
- Only 8,000 of the publisher's 120,000 training rows were used.
- The comparison is against the same untuned base model, not against full fine-tuning or a different architecture. A TF-IDF and logistic-regression baseline reaches 0.887 validation macro-F1 in about a second of CPU time, so the adapter's advantage should be weighed against its cost.
- Labels in this dataset are genuinely ambiguous between Business and Sci/Tech, which accounts for roughly half the remaining errors.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
base = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.3', quantization_config=quantization, device_map='auto'
)
model = PeftModel.from_pretrained(base, '<this-adapter>')
```

## Verification

Every number above is read from hashed artifacts rather than typed by hand. `loraforge verify` recomputes the metrics from the stored raw logits and fails if a report has been edited.
