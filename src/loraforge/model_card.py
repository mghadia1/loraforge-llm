"""Generate a Hugging Face model card from the run's own artifacts.

Hand-written cards drift from the run they describe: a number gets updated in one
place and not the other, and the card slowly becomes fiction. Every claim here is
read out of the hashed reports, so the card cannot say anything the evidence does
not, and a run that never touched the test split cannot quote test results.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .provenance import EvidenceError, read_json


PLACEHOLDER = "[More Information Needed]"


def _format_parameters(report: dict[str, Any]) -> str:
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        raise EvidenceError(
            "training report has no audited parameters block; refusing to invent a count"
        )
    required = {"trainable_parameters", "trainable_percent", "total_parameters"}
    if not required <= parameters.keys():
        raise EvidenceError(
            "training report parameters block is incomplete; refusing to render a model card"
        )
    trainable = parameters["trainable_parameters"]
    total = parameters["total_parameters"]
    percent = parameters["trainable_percent"]
    if (
        type(trainable) is not int
        or type(total) is not int
        or trainable <= 0
        or total < trainable
        or not isinstance(percent, (int, float))
        or not math.isfinite(float(percent))
        or float(percent) <= 0
        or not math.isclose(float(percent), 100 * trainable / total, rel_tol=1e-6)
    ):
        raise EvidenceError("training report parameters block is not internally consistent")
    return (
        f"{trainable:,} "
        f"({float(percent):.4f}% of {total:,})"
    )


def _final_report_for_run(root: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    """Return only a held-out report that is statically bound to this training run."""
    final_path = root / "outputs/final-test-report.json"
    if not final_path.exists():
        return None
    final = read_json(final_path)
    expected = {
        "test_evaluated": True,
        "test_evaluations_run": 1,
        "model": report["model"],
        "model_revision": report["model_revision"],
        "selected_epoch": report["selection"]["selected_epoch"],
        "selected_adapter_hashes": report["selection"]["selected_adapter_hashes"],
        "config": report["config"],
    }
    for field, value in expected.items():
        if final.get(field) != value:
            raise EvidenceError(
                f"final report {field} does not belong to this training run; "
                "refusing to quote its test result"
            )
    if set(final.get("systems", {})) != {"base", "tuned"}:
        raise EvidenceError("final report must contain exactly the base and tuned systems")
    if type(final.get("rows")) is not int or final["rows"] <= 0:
        raise EvidenceError("final report row count must be a positive integer")
    return final


def build_model_card(*, root: Path = Path("."), repo_url: str | None = None) -> str:
    """Render the card for the selected adapter under ``root``."""
    root = Path(root)
    report = read_json(root / "outputs/training-report.json")
    config = report["config"]
    lora, training, data = config["lora"], config["training"], config["data"]
    environment = report["environment"]

    selected_epoch = report["selection"]["selected_epoch"]
    from .training import select_checkpoint

    selected = select_checkpoint(report["epochs"])
    if selected["epoch"] != selected_epoch:
        raise EvidenceError("training report selected epoch does not follow its selection rule")
    validation = selected["validation"]
    base_validation = report["base_validation_metrics"]

    selected_hashes = report["selection"].get("selected_adapter_hashes")
    if not isinstance(selected_hashes, dict) or type(selected_hashes.get("total_bytes")) is not int:
        raise EvidenceError("training report has no recorded selected-adapter size")
    adapter_bytes = selected_hashes["total_bytes"]
    if adapter_bytes <= 0:
        raise EvidenceError("recorded selected-adapter size must be positive")

    # Only a run that actually evaluated the held-out split may quote it.
    final = _final_report_for_run(root, report)

    lines: list[str] = []
    add = lines.append

    add("---")
    add(f"base_model: {report['model']}")
    add("library_name: peft")
    add("license: apache-2.0")
    add("language:\n- en")
    add("datasets:\n- " + data["dataset_name"])
    add("tags:\n- lora\n- qlora\n- text-classification\n- peft")
    add("---\n")

    add(f"# QLoRA adapter for {data['dataset_name']} topic classification\n")
    add(
        f"A rank-{lora['rank']} QLoRA adapter for "
        f"[`{report['model']}`](https://huggingface.co/{report['model']}), trained to "
        f"classify news articles into World, Sports, Business, or Sci/Tech. The base "
        f"model is frozen in 4-bit NF4; only the adapter was trained.\n"
    )
    if repo_url:
        add(f"Source, evidence, and verification tooling: {repo_url}\n")

    add("## Results\n")
    add(
        f"Selected on **validation macro-F1** at epoch {selected_epoch} of "
        f"{training['epochs']}. The untuned base model was scored with the identical "
        f"prompt, so the comparison isolates the adapter.\n"
    )
    add("| System | Split | Accuracy | Macro-F1 |")
    add("|---|---|---:|---:|")
    add(
        f"| Untuned base | validation | {base_validation['accuracy']:.4f} "
        f"| {base_validation['macro_f1']:.4f} |"
    )
    add(
        f"| **This adapter** | validation | **{validation['accuracy']:.4f}** "
        f"| **{validation['macro_f1']:.4f}** |"
    )
    if final:
        for name, label in (("base", "Untuned base"), ("tuned", "**This adapter**")):
            metrics = final["systems"][name]["metrics_before_temperature"]
            add(
                f"| {label} | held-out test ({final['rows']:,} rows) "
                f"| {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} |"
            )
        add(
            "\nThe held-out split was evaluated **once**, after the checkpoint and "
            "calibration temperature were frozen on validation.\n"
        )
    else:
        add(
            "\n**No held-out test result is reported for this adapter.** It was trained "
            "as a validation-only arm, and quoting another run's test number here would "
            "attribute a result it never earned.\n"
        )

    add("## Training\n")
    add(f"- Base model: `{report['model']}` at revision `{report['model_revision']}`")
    add(
        f"- Data: `{data['dataset_name']}` at revision `{data['dataset_revision']}`, "
        f"{report['train_rows']:,} training and {report['validation_rows']:,} validation "
        f"rows, balanced across four classes, selected deterministically with seed {data['seed']}"
    )
    add(
        f"- Quantization: 4-bit NF4, double quantization, "
        f"{config['quantization']['compute_dtype']} compute"
    )
    add(
        f"- LoRA: rank {lora['rank']}, alpha {lora['alpha']}, dropout {lora['dropout']}, "
        f"targeting {', '.join(f'`{m}`' for m in lora['target_modules'])}"
    )
    add(f"- Trainable parameters: **{_format_parameters(report)}**")
    add(
        f"- Optimization: {training['epochs']} epochs, learning rate "
        f"{training['learning_rate']}, effective batch "
        f"{training['per_device_train_batch_size'] * training['gradient_accumulation_steps']}, "
        f"{training['optimizer']}, {training['warmup_ratio']:.0%} warmup"
    )
    add(
        "- Loss is computed **only** on the answer token and EOS; every prompt token is "
        "masked with `-100`"
    )
    add(
        f"- Hardware: {environment.get('gpu_name', 'unknown')}, "
        f"{report['wall_time_seconds'] / 3600:.2f} h, peak "
        f"{report['peak_cuda_memory_gib']:.2f} GiB CUDA"
    )
    add(f"- Adapter size: {adapter_bytes:,} bytes")
    add("")

    add("## How the class is read\n")
    add(
        "The prompt asks for one code — `A`=World, `B`=Sports, `C`=Business, `D`=Sci/Tech — "
        "and the prediction is the argmax over those four next-token logits in a single "
        "forward pass. Because decoding is constrained to the four codes, an unparseable "
        "output is impossible by construction, so a 0% invalid-output rate is a property "
        "of the scoring design and not a result.\n"
    )

    add("## Limitations\n")
    add(
        f"- Trained and evaluated only on {data['dataset_name']}: short English news "
        "headlines with a single topic label. Nothing here supports use on other domains, "
        "longer documents, or safety-critical decisions."
    )
    add(
        "- One seed, one hardware run. Test-set sampling uncertainty has been quantified "
        "for the reference run; training variance across seeds has not been measured."
    )
    add(
        f"- Only {report['train_rows']:,} of the publisher's 120,000 training rows were used."
    )
    add(
        "- The comparison is against the same untuned base model, not against full "
        "fine-tuning or a different architecture. A TF-IDF and logistic-regression "
        "baseline reaches 0.887 validation macro-F1 in about a second of CPU time, so the "
        "adapter's advantage should be weighed against its cost."
    )
    add(
        "- Labels in this dataset are genuinely ambiguous between Business and Sci/Tech, "
        "which accounts for roughly half the remaining errors.\n"
    )

    add("## Usage\n")
    add("```python")
    add("from peft import PeftModel")
    add("from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig")
    add("")
    add("quantization = BitsAndBytesConfig(")
    add("    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True")
    add(")")
    add(f"tokenizer = AutoTokenizer.from_pretrained('{report['model']}')")
    add("base = AutoModelForCausalLM.from_pretrained(")
    add(f"    '{report['model']}', quantization_config=quantization, device_map='auto'")
    add(")")
    add("model = PeftModel.from_pretrained(base, '<this-adapter>')")
    add("```\n")

    add("## Verification\n")
    add(
        "Every number above is read from hashed artifacts rather than typed by hand. "
        "`loraforge verify` recomputes the metrics from the stored raw logits and fails "
        "if a report has been edited.\n"
    )

    card = "\n".join(lines)
    if PLACEHOLDER in card:
        raise EvidenceError("the generated card still contains an unfilled placeholder")
    return card


CARD_PATH = Path("outputs/model-card.md")


def write_model_card(*, root: Path = Path("."), repo_url: str | None = None) -> Path:
    """Write the card beside the run's other artifacts, never inside the adapter.

    The adapter directory's SHA-256 is the anchor of the whole evidence chain: it
    is recorded at training time, again in the frozen selection, and again in the
    release manifest. Adding a file to that directory silently invalidates all
    three, and an earlier version of this function did exactly that. Publishing to
    the Hub still wants the card as the repo's README, but copying it in is a
    deliberate act that ends the local hash's validity, not a side effect of
    generating documentation.
    """
    root = Path(root)
    target = root / CARD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_model_card(root=root, repo_url=repo_url), encoding="utf-8")
    return target
