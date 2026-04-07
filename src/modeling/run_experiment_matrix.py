"""Run the core model-variant experiment matrix with multi-seed aggregation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from .env_utils import load_project_env


DEFAULT_CONFIGS = [
    "configs/modeling/bert_base_baseline.yaml",
    "configs/modeling/bert_base_slang_masked.yaml",
    "configs/modeling/bert_base_mixed.yaml",
    "configs/modeling/bertweet_baseline.yaml",
    "configs/modeling/bertweet_slang_masked.yaml",
    "configs/modeling/bertweet_mixed.yaml",
    "configs/modeling/gpt_classification_baseline.yaml",
    "configs/modeling/gpt_classification_slang_masked.yaml",
    "configs/modeling/gpt_classification_mixed.yaml",
]


def _run_one(config_path: str, seeds: list[int], dry_run: bool) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "src.modeling.run_multiseed",
        "--config",
        config_path,
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    if dry_run:
        return {
            "config": config_path,
            "status": "dry_run",
            "command": command,
            "exit_code": 0,
        }

    process = subprocess.run(command, check=False)
    return {
        "config": config_path,
        "status": "ok" if process.returncode == 0 else "failed",
        "command": command,
        "exit_code": process.returncode,
    }


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(
        description="Run the 3x3 core model/variant matrix with multi-seed aggregation."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="One or more integer seeds to run for each config.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to execute. Defaults to the 9 core model/variant configs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining configs even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without executing training.",
    )
    parser.add_argument(
        "--manifest-out",
        default="outputs/modeling/matrix_runs/matrix_manifest.json",
        help="Path to write run manifest JSON.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for config in args.configs:
        row = _run_one(config, args.seeds, dry_run=args.dry_run)
        rows.append(row)
        if row["status"] == "failed" and not args.continue_on_error:
            break

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "dry_run": args.dry_run,
        "num_requested": len(args.configs),
        "num_executed": len(rows),
        "num_failed": sum(1 for row in rows if row["status"] == "failed"),
        "rows": rows,
    }

    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
