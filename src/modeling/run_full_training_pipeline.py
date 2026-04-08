"""Run end-to-end training pipelines with resumable per-seed execution."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .config import ExperimentConfig, load_experiment_config
from .dapt import run_dapt
from .env_utils import load_project_env
from .logging_utils import ensure_dir, write_json
from .run_multiseed import _mean, _std
from .train_classification import train_from_experiment_config


CORE_MATRIX_CONFIGS = [
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

NON_GPT_CORE_CONFIGS = [
    "configs/modeling/bert_base_baseline.yaml",
    "configs/modeling/bert_base_slang_masked.yaml",
    "configs/modeling/bert_base_mixed.yaml",
    "configs/modeling/bertweet_baseline.yaml",
    "configs/modeling/bertweet_slang_masked.yaml",
    "configs/modeling/bertweet_mixed.yaml",
]

GPT_FINETUNE_CONFIGS = [
    "configs/modeling/gpt_finetune_slang.yaml",
    "configs/modeling/gpt_finetune_slang_mixed.yaml",
]

DAPT_CONFIG = "configs/modeling/gpt_dapt.yaml"


def _is_corrupt_checkpoint_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "error while deserializing header" in message
        or "incomplete metadata" in message
    )


def _clear_seed_checkpoints(seed_output_dir: Path) -> None:
    for checkpoint_dir in seed_output_dir.glob("checkpoint-step-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _seed_cfg(base_cfg: ExperimentConfig, seed: int) -> ExperimentConfig:
    cfg = deepcopy(base_cfg)
    cfg.training.seed = seed
    cfg.logging.run_name = f"{base_cfg.logging.run_name}_seed{seed}"
    cfg.training.output_dir = str(Path(base_cfg.training.output_dir) / f"seed_{seed}")
    cfg.logging.output_dir = str(Path(base_cfg.logging.output_dir) / f"seed_{seed}")
    return cfg


def _seed_summary_path(base_cfg: ExperimentConfig, seed: int) -> Path:
    return Path(base_cfg.logging.output_dir) / f"seed_{seed}" / "run_summary.json"


def _read_seed_metrics(summary_path: Path) -> dict[str, float | int]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "test_accuracy": float(payload["test_accuracy"]),
        "test_macro_f1": float(payload["test_macro_f1"]),
        "num_test_samples": int(payload["num_test_samples"]),
    }


def _write_multiseed_summary(
    base_cfg: ExperimentConfig,
    config_path: str,
    seeds: list[int],
    seed_metrics: dict[int, dict[str, float | int]],
    environment: dict[str, object],
) -> Path:
    aggregate_output_dir = ensure_dir(Path(base_cfg.logging.output_dir) / "multiseed")

    rows = [
        {
            "seed": seed,
            "test_accuracy": float(seed_metrics[seed]["test_accuracy"]),
            "test_macro_f1": float(seed_metrics[seed]["test_macro_f1"]),
            "num_test_samples": int(seed_metrics[seed]["num_test_samples"]),
        }
        for seed in seeds
    ]

    payload = {
        "experiment_name": base_cfg.experiment_name,
        "config_path": config_path,
        "seeds": seeds,
        "environment": environment,
        "per_seed": rows,
        "aggregate": {
            "test_accuracy_mean": _mean([row["test_accuracy"] for row in rows]),
            "test_accuracy_std": _std([row["test_accuracy"] for row in rows]),
            "test_macro_f1_mean": _mean([row["test_macro_f1"] for row in rows]),
            "test_macro_f1_std": _std([row["test_macro_f1"] for row in rows]),
            "num_test_samples": rows[0]["num_test_samples"] if rows else 0,
        },
    }

    out_path = aggregate_output_dir / "multiseed_summary.json"
    write_json(out_path, payload)
    return out_path


def _run_resumable_multiseed(
    config_path: str,
    seeds: list[int],
    skip_completed: bool,
    dry_run: bool,
) -> dict[str, object]:
    base_cfg = load_experiment_config(config_path)
    seed_metrics: dict[int, dict[str, float | int]] = {}
    environment: dict[str, object] = {}
    per_seed_status: list[dict[str, object]] = []

    for seed in seeds:
        summary_path = _seed_summary_path(base_cfg, seed)
        if skip_completed and summary_path.exists():
            metrics = _read_seed_metrics(summary_path)
            seed_metrics[seed] = metrics
            per_seed_status.append(
                {
                    "seed": seed,
                    "status": "skipped_completed",
                    "summary_path": str(summary_path),
                }
            )
            continue

        if dry_run:
            per_seed_status.append({"seed": seed, "status": "dry_run"})
            continue

        cfg = _seed_cfg(base_cfg, seed)
        try:
            metrics = train_from_experiment_config(cfg)
        except Exception as exc:  # noqa: BLE001
            if not _is_corrupt_checkpoint_error(exc):
                raise

            seed_output_dir = Path(cfg.training.output_dir)
            print(
                f"Corrupt checkpoint detected for {cfg.logging.run_name}; "
                "clearing checkpoint-step-* dirs and retrying seed once."
            )
            _clear_seed_checkpoints(seed_output_dir)

            cfg_retry = _seed_cfg(base_cfg, seed)
            cfg_retry.training.auto_resume_latest_checkpoint = False
            cfg_retry.training.resume_from_checkpoint = None
            metrics = train_from_experiment_config(cfg_retry)

        seed_metrics[seed] = {
            "test_accuracy": float(metrics["test_accuracy"]),
            "test_macro_f1": float(metrics["test_macro_f1"]),
            "num_test_samples": int(metrics["num_test_samples"]),
        }
        if not environment and isinstance(metrics.get("environment"), dict):
            environment = metrics["environment"]

        per_seed_status.append(
            {
                "seed": seed,
                "status": "completed",
                "summary_path": str(_seed_summary_path(base_cfg, seed)),
            }
        )

    if dry_run:
        return {
            "config": config_path,
            "experiment_name": base_cfg.experiment_name,
            "status": "dry_run",
            "per_seed": per_seed_status,
        }

    missing = [seed for seed in seeds if seed not in seed_metrics]
    if missing:
        raise RuntimeError(
            f"Missing metrics for seeds {missing} after run for config '{config_path}'."
        )

    multiseed_path = _write_multiseed_summary(
        base_cfg=base_cfg,
        config_path=config_path,
        seeds=seeds,
        seed_metrics=seed_metrics,
        environment=environment,
    )

    return {
        "config": config_path,
        "experiment_name": base_cfg.experiment_name,
        "status": "completed",
        "per_seed": per_seed_status,
        "multiseed_summary_path": str(multiseed_path),
    }


def _run_dapt_if_needed(skip_completed: bool, dry_run: bool) -> dict[str, object]:
    cfg = load_experiment_config(DAPT_CONFIG)
    marker = Path(cfg.training.output_dir) / "dapt_model"

    if skip_completed and marker.exists():
        return {
            "stage": "dapt",
            "config": DAPT_CONFIG,
            "status": "skipped_completed",
            "marker": str(marker),
        }

    if dry_run:
        return {"stage": "dapt", "config": DAPT_CONFIG, "status": "dry_run"}

    try:
        run_dapt(DAPT_CONFIG)
    except Exception:
        return {
            "stage": "dapt",
            "config": DAPT_CONFIG,
            "status": "failed",
            "exit_code": 1,
        }

    return {
        "stage": "dapt",
        "config": DAPT_CONFIG,
        "status": "completed",
        "marker": str(marker),
    }


def _dapt_model_ready() -> bool:
    cfg = load_experiment_config(DAPT_CONFIG)
    marker = Path(cfg.training.output_dir) / "dapt_model"
    return marker.exists()


def _run_summary_tables(dry_run: bool) -> list[dict[str, object]]:
    commands = [
        [
            sys.executable,
            "-m",
            "src.modeling.summarize_runs",
            "--root",
            "outputs/modeling",
            "--output",
            "outputs/modeling/run_summary_table.csv",
        ],
        [
            sys.executable,
            "-m",
            "src.modeling.summarize_multiseed",
            "--root",
            "outputs/modeling",
            "--output",
            "outputs/modeling/multiseed_summary_table.csv",
        ],
    ]

    rows: list[dict[str, object]] = []
    for command in commands:
        if dry_run:
            rows.append({"command": command, "status": "dry_run"})
            continue

        proc = subprocess.run(command, check=False)
        rows.append(
            {
                "command": command,
                "status": "completed" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
            }
        )
    return rows


def main() -> None:
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Run full training pipelines sequentially with resumable seed-level behavior."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds for classification pipelines.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip seeds/stages that already have completion artifacts.",
    )
    parser.add_argument(
        "--include-dapt",
        action="store_true",
        help="Also run GPT DAPT stage before GPT finetuning configs.",
    )
    parser.add_argument(
        "--include-gpt-finetune",
        action="store_true",
        help="Also run GPT slang finetuning configs as resumable multi-seed runs.",
    )
    parser.add_argument(
        "--non-gpt-only",
        action="store_true",
        help="Run only BERT-base and BERTweet configs (skip GPT classification/DAPT/finetune stages).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later configs/stages when one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and write manifest without executing training.",
    )
    parser.add_argument(
        "--manifest-out",
        default="outputs/modeling/pipeline_runs/full_pipeline_manifest.json",
        help="Path for pipeline manifest JSON.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    failed = False

    active_core_configs = (
        NON_GPT_CORE_CONFIGS if args.non_gpt_only else CORE_MATRIX_CONFIGS
    )

    include_dapt = bool(args.include_dapt and not args.non_gpt_only)
    include_gpt_finetune = bool(args.include_gpt_finetune and not args.non_gpt_only)

    if args.non_gpt_only and (args.include_dapt or args.include_gpt_finetune):
        rows.append(
            {
                "stage": "gpt_stages",
                "status": "skipped_non_gpt_only",
                "reason": "--non-gpt-only disables GPT classification, DAPT, and GPT finetuning stages.",
            }
        )

    for config in active_core_configs:
        try:
            row = _run_resumable_multiseed(
                config_path=config,
                seeds=args.seeds,
                skip_completed=args.skip_completed,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            row = {"config": config, "status": "failed", "error": str(exc)}
        rows.append(row)
        if row.get("status") == "failed":
            failed = True
            if not args.continue_on_error:
                break

    if include_dapt and (args.continue_on_error or not failed):
        dapt_row = _run_dapt_if_needed(
            skip_completed=args.skip_completed,
            dry_run=args.dry_run,
        )
        rows.append(dapt_row)
        if dapt_row.get("status") == "failed":
            failed = True

    if include_gpt_finetune and (args.continue_on_error or not failed):
        dapt_ready = True if args.dry_run else _dapt_model_ready()
        if not dapt_ready:
            for config in GPT_FINETUNE_CONFIGS:
                rows.append(
                    {
                        "config": config,
                        "status": "skipped_dependency",
                        "reason": "DAPT model missing at outputs/modeling/gpt_dapt/dapt_model",
                    }
                )
        else:
            for config in GPT_FINETUNE_CONFIGS:
                try:
                    row = _run_resumable_multiseed(
                        config_path=config,
                        seeds=args.seeds,
                        skip_completed=args.skip_completed,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {"config": config, "status": "failed", "error": str(exc)}
                rows.append(row)
                if row.get("status") == "failed":
                    failed = True
                    if not args.continue_on_error:
                        break

    if args.continue_on_error or not failed:
        rows.extend(_run_summary_tables(dry_run=args.dry_run))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "skip_completed": args.skip_completed,
        "include_dapt": args.include_dapt,
        "include_gpt_finetune": args.include_gpt_finetune,
        "non_gpt_only": args.non_gpt_only,
        "continue_on_error": args.continue_on_error,
        "dry_run": args.dry_run,
        "num_rows": len(rows),
        "num_failed": sum(1 for row in rows if row.get("status") == "failed"),
        "rows": rows,
    }

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
