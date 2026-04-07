"""Run classification experiments across multiple seeds and aggregate metrics."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .config import load_experiment_config
from .env_utils import load_project_env
from .logging_utils import ensure_dir, write_json
from .train_classification import train_from_experiment_config


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(
        description="Run transformer sentiment classification training across multiple seeds."
    )
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config."
    )
    parser.add_argument(
        "--seeds",
        required=True,
        nargs="+",
        type=int,
        help="One or more integer seeds.",
    )
    args = parser.parse_args()

    base_cfg = load_experiment_config(args.config)
    aggregate_output_dir = ensure_dir(Path(base_cfg.logging.output_dir) / "multiseed")

    per_seed_results: list[dict[str, float | int]] = []
    environment_provenance: dict[str, object] | None = None
    for seed in args.seeds:
        cfg = deepcopy(base_cfg)
        cfg.training.seed = seed
        cfg.logging.run_name = f"{base_cfg.logging.run_name}_seed{seed}"
        cfg.training.output_dir = str(
            Path(base_cfg.training.output_dir) / f"seed_{seed}"
        )
        cfg.logging.output_dir = str(Path(base_cfg.logging.output_dir) / f"seed_{seed}")

        metrics = train_from_experiment_config(cfg)
        if environment_provenance is None and isinstance(
            metrics.get("environment"), dict
        ):
            environment_provenance = metrics["environment"]
        per_seed_results.append(
            {
                "seed": seed,
                "test_accuracy": float(metrics["test_accuracy"]),
                "test_macro_f1": float(metrics["test_macro_f1"]),
                "num_test_samples": int(metrics["num_test_samples"]),
            }
        )

    summary = {
        "experiment_name": base_cfg.experiment_name,
        "config_path": args.config,
        "seeds": args.seeds,
        "environment": environment_provenance or {},
        "per_seed": per_seed_results,
        "aggregate": {
            "test_accuracy_mean": _mean(
                [row["test_accuracy"] for row in per_seed_results]
            ),
            "test_accuracy_std": _std(
                [row["test_accuracy"] for row in per_seed_results]
            ),
            "test_macro_f1_mean": _mean(
                [row["test_macro_f1"] for row in per_seed_results]
            ),
            "test_macro_f1_std": _std(
                [row["test_macro_f1"] for row in per_seed_results]
            ),
            "num_test_samples": (
                per_seed_results[0]["num_test_samples"] if per_seed_results else 0
            ),
        },
    }

    write_json(aggregate_output_dir / "multiseed_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
