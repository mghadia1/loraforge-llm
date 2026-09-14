"""Generate a Hugging Face model card from the run's own artifacts.

Hand-written cards drift from the run they describe: a number gets updated in one
place and not the other, and the card slowly becomes fiction. Every claim here is
read out of the hashed reports, so the card cannot say anything the evidence does
not, and a run that never touched the test split cannot quote test results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .provenance import EvidenceError, read_json, sha256_file


PLACEHOLDER = "[More Information Needed]"
QLORA_SETUP = Path("outputs/qlora-setup.json")
RELEASE_EVIDENCE = Path("docs/evidence/selected-adapter-release.json")


def _verified_setup_evidence(
    root: Path, report: dict[str, Any]
) -> dict[str, Any] | None:
    """Cross-check resource claims against the independent pre-training setup."""
    path = root / QLORA_SETUP
    if not path.exists():
        return None
    setup = read_json(path)
    config = report["config"]
    expected = {
        "schema_version": 1,
        "model": report["model"],
        "model_revision": report["model_revision"],
        "quantization": config["quantization"],
        "lora": config["lora"],
        "test_loaded": False,
        "adapter_trained": False,
    }
    for field, value in expected.items():
        if type(setup.get(field)) is not type(value) or setup.get(field) != value:
            raise EvidenceError(
                f"QLoRA setup {field} does not match the verified training run"
            )

    parameters = setup.get("parameters")
    if not isinstance(parameters, dict):
        raise EvidenceError("QLoRA setup has no parameter evidence")
    trainable = parameters.get("trainable_parameters")
    if type(trainable) is not int or trainable <= 0:
        raise EvidenceError("QLoRA setup trainable parameter count must be a positive integer")
    if parameters.get("only_lora_parameters_trainable") is not True:
        raise EvidenceError("QLoRA setup does not prove that only LoRA parameters are trainable")

    training_parameters = report.get("parameters")
    if training_parameters is not None:
        if not isinstance(training_parameters, dict):
            raise EvidenceError("training report parameters must be an object")
        recorded = training_parameters.get("trainable_parameters")
        if type(recorded) is not int or recorded != trainable:
            raise EvidenceError(
                "training report trainable parameter count does not match QLoRA setup evidence"
            )

    environment = report.get("environment")
    gpu_name = setup.get("gpu_name")
    if (
        not isinstance(environment, dict)
        or not isinstance(gpu_name, str)
        or not gpu_name.strip()
        or environment.get("gpu_name") != gpu_name
    ):
        raise EvidenceError(
            "training report GPU does not match the independent QLoRA setup evidence"
        )
    return setup


def _verified_release_adapter_size(
    root: Path, report: dict[str, Any]
) -> int | None:
    """Return adapter bytes only when the release manifest binds this exact run."""
    path = root / RELEASE_EVIDENCE
    if not path.exists():
        return None
    release = read_json(path)
    selection = report["selection"]
    expected = {
        "schema_version": 1,
        "selected_epoch": selection["selected_epoch"],
        "adapter_directory": selection["selected_adapter_dir"],
        "adapter_directory_hash": selection["selected_adapter_hashes"],
        "base_model": report["model"],
        "base_model_revision": report["model_revision"],
    }
    for field, value in expected.items():
        if type(release.get(field)) is not type(value) or release.get(field) != value:
            raise EvidenceError(
                f"selected-adapter release {field} does not match the verified training run"
            )
    adapter_bytes = release["adapter_directory_hash"].get("total_bytes")
    if type(adapter_bytes) is not int or adapter_bytes <= 0:
        raise EvidenceError("release adapter size must be a positive integer")
    return adapter_bytes


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
        actual = final.get(field)
        if type(actual) is not type(value) or actual != value:
            raise EvidenceError(
                f"final report {field} does not belong to this training run; "
                "refusing to quote its test result"
            )
    if set(final.get("systems", {})) != {"base", "tuned"}:
        raise EvidenceError("final report must contain exactly the base and tuned systems")
    if type(final.get("rows")) is not int or final["rows"] <= 0:
        raise EvidenceError("final report row count must be a positive integer")
    return final


def _verified_reports_for_card(root: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    """Verify validation evidence and require held-out claims to be hash-bound."""
    from .intervals import INTERVALS_REPORT, SOURCE_REPORT
    from .selection import verify_training_report

    # Model-card generation is CPU-only and keeps publisher test locked. The
    # training verifier reloads publisher train to recompute validation metrics.
    verify_training_report(report, root=root, verify_adapters=False)
    final = _final_report_for_run(root, report)
    if final is None:
        return None

    intervals_path = root / INTERVALS_REPORT
    if not intervals_path.exists():
        raise EvidenceError(
            "held-out model-card claims require the hash-bound intervals report"
        )
    intervals = read_json(intervals_path)
    final_path = root / SOURCE_REPORT
    if (
        intervals.get("source_report") != SOURCE_REPORT
        or intervals.get("source_report_sha256") != sha256_file(final_path)
    ):
        raise EvidenceError(
            "final report is not hash-bound by the intervals evidence; "
            "refusing to quote its held-out results"
        )
    return final


def build_model_card(*, root: Path = Path("."), repo_url: str | None = None) -> str:
    """Render the card for the selected adapter under ``root``."""
    root = Path(root)
    report = read_json(root / "outputs/training-report.json")
    final = _verified_reports_for_card(root, report)
    config = report["config"]
    lora, training, data = config["lora"], config["training"], config["data"]
    setup = _verified_setup_evidence(root, report)

    selected_epoch = report["selection"]["selected_epoch"]
    from .training import select_checkpoint

    selected = select_checkpoint(report["epochs"])
    if selected["epoch"] != selected_epoch:
        raise EvidenceError("training report selected epoch does not follow its selection rule")
    validation = selected["validation"]
    base_validation = report["base_validation_metrics"]

    adapter_bytes = _verified_release_adapter_size(root, report)

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
    if setup is not None:
        trainable = setup["parameters"]["trainable_parameters"]
        add(
            f"- Trainable parameters: **{trainable:,}** (cross-checked against the "
            "pre-training QLoRA setup; no precise percentage is claimed)"
        )
    else:
        add(
            "- Trainable-parameter and hardware claims are omitted because this run has "
            "no matching pre-training setup evidence"
        )
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
    if setup is not None:
        add(f"- Hardware: {setup['gpu_name']} (recorded by setup and training artifacts)")
    if adapter_bytes is not None:
        add(f"- Adapter size: {adapter_bytes:,} bytes (verified by the release manifest)")
    else:
        add("- Adapter-size claim omitted because no matching release manifest is present")
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
        "fine-tuning, a different architecture, or a separately measured classical "
        "baseline."
    )
    add(
        "- The run records aggregate and per-class metrics, but no annotated error-analysis "
        "artifact; it therefore does not support causal claims about the remaining errors.\n"
    )

    add("## Usage\n")
    add("```python")
    add("from peft import PeftModel")
    add("from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig")
    add("")
    add("quantization = BitsAndBytesConfig(")
    add("    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True")
    add(")")
    add("revision = " + repr(report["model_revision"]))
    add(
        f"tokenizer = AutoTokenizer.from_pretrained('{report['model']}', revision=revision)"
    )
    add("base = AutoModelForCausalLM.from_pretrained(")
    add(
        f"    '{report['model']}', revision=revision, "
        "quantization_config=quantization, device_map='auto'"
    )
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
