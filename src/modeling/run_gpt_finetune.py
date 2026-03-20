"""Fine-tune a DAPT-adapted GPT classifier on slang-heavy sentiment data."""

from __future__ import annotations

import argparse
import json

from .train_classification import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune adapted GPT classifier.")
    parser.add_argument(
        "--config", required=True, help="Path to finetuning YAML config."
    )
    args = parser.parse_args()

    metrics = train_from_config(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
