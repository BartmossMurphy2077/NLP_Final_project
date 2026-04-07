"""Summarize multiseed aggregate outputs into a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _collect_multiseed_summaries(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for summary_path in root.rglob("multiseed_summary.json"):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        aggregate = payload.get("aggregate", {})
        environment = payload.get("environment", {})
        rows.append(
            {
                "experiment_name": payload.get("experiment_name", ""),
                "summary_path": str(summary_path),
                "config_path": payload.get("config_path", ""),
                "seeds": "|".join(str(seed) for seed in payload.get("seeds", [])),
                "test_accuracy_mean": aggregate.get("test_accuracy_mean", ""),
                "test_accuracy_std": aggregate.get("test_accuracy_std", ""),
                "test_macro_f1_mean": aggregate.get("test_macro_f1_mean", ""),
                "test_macro_f1_std": aggregate.get("test_macro_f1_std", ""),
                "num_test_samples": aggregate.get("num_test_samples", ""),
                "python_version": environment.get("python_version", ""),
                "torch_version": environment.get("torch_version", ""),
                "transformers_version": environment.get("transformers_version", ""),
                "device": environment.get("device", ""),
                "cuda_available": environment.get("cuda_available", ""),
                "cuda_device_name": environment.get("cuda_device_name", ""),
            }
        )

    return sorted(rows, key=lambda row: str(row["experiment_name"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multiseed aggregate outputs into a CSV table."
    )
    parser.add_argument(
        "--root",
        default="outputs/modeling",
        help="Root directory containing modeling outputs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/modeling/multiseed_summary_table.csv",
        help="CSV path to write the flat multiseed summary table.",
    )
    args = parser.parse_args()

    rows = _collect_multiseed_summaries(Path(args.root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(json.dumps({"rows_written": 0, "output": str(output_path)}, indent=2))
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows_written": len(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
