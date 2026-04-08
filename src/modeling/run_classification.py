"""CLI entrypoint for baseline and adapted classification experiments."""

from __future__ import annotations

import argparse
import json

from .env_utils import load_project_env
from .train_classification import train_from_config


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(
        description="Run transformer sentiment classification training."
    )
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config."
    )
    args = parser.parse_args()

    results = train_from_config(args.config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
