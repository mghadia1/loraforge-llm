"""GPU-free LoRAForge preparation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import default_config
from .data import describe, load_dataset, write_stats
from .prompts import training_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loraforge")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-config", help="write the frozen experiment config")
    stats = commands.add_parser("data-stats", help="load real data without test and write stats")
    stats.add_argument("--output", type=Path, default=Path("docs/evidence/data-stats.json"))
    examples = commands.add_parser("examples", help="write three deterministic formatted examples")
    examples.add_argument(
        "--output", type=Path, default=Path("docs/evidence/formatted-examples.json")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = default_config()
    if args.command == "write-config":
        output = Path("configs/experiment.json")
        config.write(output)
        print(json.dumps({"output": str(output), "test_evaluations_allowed": 1}))
        return 0
    bundle = load_dataset(allow_test=False, config=config.data)
    if args.command == "data-stats":
        write_stats(bundle, config.data, args.output)
        print(json.dumps(describe(bundle, config.data)))
        return 0
    examples = []
    selected = [
        next(item for item in bundle.train.examples if item.label == label)
        for label in range(3)
    ]
    for item in selected:
        examples.append(
            {
                "row_id": item.row_id,
                "messages": training_messages(item.text, item.label),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
