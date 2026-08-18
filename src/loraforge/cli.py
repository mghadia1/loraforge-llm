"""LoRAForge commands: GPU-free preparation and verification, plus the guarded GPU runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import default_config, load_config
from .data import describe, load_dataset, write_stats
from .prompts import training_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loraforge")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-config", help="write the frozen experiment config")
    stats = commands.add_parser("data-stats", help="load real data without test and write stats")
    stats.add_argument("--output", type=Path, default=Path("docs/evidence/data-stats.json"))
    stats.add_argument("--config", type=Path, help="validated experiment JSON")
    examples = commands.add_parser("examples", help="write three deterministic formatted examples")
    examples.add_argument(
        "--output", type=Path, default=Path("docs/evidence/formatted-examples.json")
    )
    examples.add_argument("--config", type=Path, help="validated experiment JSON")

    train = commands.add_parser("train", help="run configured QLoRA training (needs a GPU)")
    train.add_argument("--root", type=Path, default=Path("."))
    train.add_argument("--config", type=Path, help="validated experiment JSON")

    freeze = commands.add_parser(
        "freeze-selection", help="verify training evidence and freeze the adapter plus temperatures"
    )
    freeze.add_argument("--root", type=Path, default=Path("."))

    final = commands.add_parser(
        "final-test", help="the single publisher-test evaluation (needs a GPU and confirmation)"
    )
    final.add_argument("--root", type=Path, default=Path("."))
    final.add_argument("--config", type=Path, help="validated experiment JSON")
    final.add_argument(
        "--confirm",
        default="",
        help="must be exactly 'i-am-running-the-single-final-test'",
    )

    intervals = commands.add_parser(
        "intervals",
        help="bootstrap confidence intervals and paired analysis from the stored test run",
    )
    intervals.add_argument("--root", type=Path, default=Path("."))
    intervals.add_argument("--resamples", type=int, default=2_000)
    intervals.add_argument("--seed", type=int, default=73)

    verify = commands.add_parser(
        "verify", help="recompute stored metrics from stored logits and reject edited evidence"
    )
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument(
        "--reports-only",
        action="store_true",
        help="verify reports/logits without requiring local adapter checkpoint files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = getattr(args, "config", None)
    config = load_config(config_path) if config_path else default_config()

    if args.command == "write-config":
        output = Path("configs/experiment.json")
        config.write(output)
        print(json.dumps({"output": str(output), "test_evaluations_allowed": 1}))
        return 0

    if args.command == "train":
        from .modeling import load_quantized_base
        from .qlora import attach_lora
        from .training import train_qlora

        bundle = load_dataset(allow_test=False, config=config.data)
        model, tokenizer = load_quantized_base(config)
        model = attach_lora(model, config)
        report = train_qlora(model, tokenizer, bundle, config, root=args.root)
        print(json.dumps({
            "selected_epoch": report["selection"]["selected_epoch"],
            "wall_time_seconds": report["wall_time_seconds"],
            "peak_cuda_memory_gib": report["peak_cuda_memory_gib"],
        }))
        return 0

    if args.command == "freeze-selection":
        from .selection import build_frozen_selection

        frozen = build_frozen_selection(root=args.root)
        print(json.dumps({
            "selected_epoch": frozen["selected_epoch"],
            "base_temperature": frozen["validation"]["base"]["temperature"],
            "tuned_temperature": frozen["validation"]["tuned"]["temperature"],
        }))
        return 0

    if args.command == "final-test":
        from .final_test import run_final_test

        report = run_final_test(config, confirmation=args.confirm, root=args.root)
        print(json.dumps({
            "base_macro_f1": report["systems"]["base"]["metrics_before_temperature"]["macro_f1"],
            "tuned_macro_f1": report["systems"]["tuned"]["metrics_before_temperature"]["macro_f1"],
            "macro_f1_delta": report["delta"]["macro_f1"],
        }))
        return 0

    if args.command == "intervals":
        from .intervals import build_intervals

        result = build_intervals(root=args.root, resamples=args.resamples, seed=args.seed)
        delta = result["bootstrap"]["delta"]
        print(json.dumps({
            "delta": delta["delta"],
            "ci": [delta["ci_lower"], delta["ci_upper"]],
            "resamples_without_improvement": result["bootstrap"]["resamples_without_improvement"],
            "fixed": result["paired"]["tuned_fixed_base_error"],
            "broke": result["paired"]["tuned_broke_base_success"],
            "new_test_evaluations": result["new_test_evaluations"],
        }, indent=2))
        return 0

    if args.command == "verify":
        from .final_test import verify_final_report
        from .intervals import INTERVALS_REPORT, verify_intervals
        from .provenance import read_json
        from .selection import TRAINING_REPORT, verify_training_report

        checked = {}
        if (Path(args.root) / TRAINING_REPORT).exists():
            selected = verify_training_report(
                read_json(Path(args.root) / TRAINING_REPORT),
                root=args.root,
                verify_adapters=not args.reports_only,
            )
            checked["training_report"] = {
                "verified": True,
                "selected_epoch": selected["epoch"],
                "adapter_files_verified": not args.reports_only,
            }
        if (Path(args.root) / "outputs/final-test-report.json").exists():
            checked["final_test_report"] = verify_final_report(root=args.root)
        if (Path(args.root) / INTERVALS_REPORT).exists():
            checked["test_intervals"] = verify_intervals(root=args.root)
        if not checked:
            raise SystemExit("nothing to verify: no training or final-test report exists yet")
        print(json.dumps(checked, indent=2))
        return 0

    bundle = load_dataset(allow_test=False, config=config.data)
    if args.command == "data-stats":
        write_stats(bundle, config.data, args.output)
        print(json.dumps(describe(bundle, config.data)))
        return 0

    examples = []
    selected_rows = [
        next(item for item in bundle.train.examples if item.label == label)
        for label in range(3)
    ]
    for item in selected_rows:
        examples.append({"row_id": item.row_id, "messages": training_messages(item.text, item.label)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
