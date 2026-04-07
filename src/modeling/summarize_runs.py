"""Summarize modeling run outputs into a flat comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _collect_run_summaries(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in root.rglob("run_summary.json"):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        config = payload.get("config", {})
        data_cfg = config.get("data", {})
        training_cfg = config.get("training", {})
        rows.append(
            {
                "experiment_name": payload.get("experiment_name", ""),
                "summary_path": str(summary_path),
                "seed": training_cfg.get("seed", ""),
                "model_name": config.get("model", {}).get("name", ""),
                "allowed_variants": "|".join(data_cfg.get("allowed_variants", [])),
                "allowed_slang_labels": "|".join(data_cfg.get("allowed_slang_labels", [])),
                "test_accuracy": payload.get("test_accuracy", ""),
                "test_macro_f1": payload.get("test_macro_f1", ""),
                "num_test_samples": payload.get("num_test_samples", ""),
                "test_evaluated_from": payload.get("test_evaluated_from", ""),
            }
        )
    return sorted(rows, key=lambda row: (str(row["experiment_name"]), str(row["seed"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modeling run outputs.")
    parser.add_argument(
        "--root",
        default="outputs/modeling",
        help="Root directory containing modeling outputs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/modeling/run_summary_table.csv",
        help="CSV path to write the flat summary table.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = _collect_run_summaries(root)
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
