from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_MODELING = ROOT / "outputs" / "modeling"
ANALYSIS_OUT = ROOT / "analysis" / "outputs"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if "bertweet" in name:
        return "bertweet"
    if "gpt" in name:
        return "gpt"
    if "bert" in name:
        return "bert"
    return "other"


def _variant_label(allowed_variants: str) -> str:
    v = (allowed_variants or "").strip().lower()
    if v in {"original", "slang_masked", "mixed"}:
        return v
    return v or "unknown"


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _load_seed_rows() -> list[dict[str, Any]]:
    raw = _read_csv(OUTPUTS_MODELING / "run_summary_table.csv")
    out: list[dict[str, Any]] = []
    for row in raw:
        summary_path = (row.get("summary_path") or "").replace("\\", "/")
        if "/seed_" not in summary_path:
            continue
        out.append(
            {
                "experiment_name": row.get("experiment_name", ""),
                "summary_path": summary_path,
                "seed": _to_int(row.get("seed", "0")),
                "model_name": row.get("model_name", ""),
                "model_family": _model_family(row.get("model_name", "")),
                "variant": _variant_label(row.get("allowed_variants", "")),
                "allowed_slang_labels": row.get("allowed_slang_labels", ""),
                "test_accuracy": _to_float(row.get("test_accuracy", "0")),
                "test_macro_f1": _to_float(row.get("test_macro_f1", "0")),
                "num_test_samples": _to_int(row.get("num_test_samples", "0")),
                "test_evaluated_from": row.get("test_evaluated_from", ""),
            }
        )
    return out


def _load_multiseed_rows() -> list[dict[str, Any]]:
    raw = _read_csv(OUTPUTS_MODELING / "multiseed_summary_table.csv")
    out: list[dict[str, Any]] = []
    for row in raw:
        out.append(
            {
                "experiment_name": row.get("experiment_name", ""),
                "summary_path": (row.get("summary_path") or "").replace("\\", "/"),
                "config_path": row.get("config_path", ""),
                "seeds": row.get("seeds", ""),
                "test_accuracy_mean": _to_float(row.get("test_accuracy_mean", "0")),
                "test_accuracy_std": _to_float(row.get("test_accuracy_std", "0")),
                "test_macro_f1_mean": _to_float(row.get("test_macro_f1_mean", "0")),
                "test_macro_f1_std": _to_float(row.get("test_macro_f1_std", "0")),
                "num_test_samples": _to_int(row.get("num_test_samples", "0")),
            }
        )
    return out


def _read_run_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_slang_heavy_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seed_rows:
        summary_rel = Path(row["summary_path"])
        summary_path = ROOT / summary_rel
        if not summary_path.exists():
            continue
        payload = _read_run_summary(summary_path)
        by_group = payload.get("test_metrics_by_group", {})
        slang_group = by_group.get("slang_label", {})
        heavy = slang_group.get("slang_heavy")
        if not heavy:
            continue
        rows.append(
            {
                "experiment_name": row["experiment_name"],
                "seed": row["seed"],
                "variant": row["variant"],
                "model_family": row["model_family"],
                "slang_heavy_accuracy": float(heavy.get("accuracy", 0.0)),
                "slang_heavy_macro_f1": float(heavy.get("macro_f1", 0.0)),
                "slang_heavy_support": int(heavy.get("support", 0)),
            }
        )
    return rows


def _aggregate_by_experiment(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["experiment_name"], []).append(float(row[metric_key]))
    out: dict[str, dict[str, Any]] = {}
    for exp, vals in grouped.items():
        out[exp] = {
            "mean": _safe_mean(vals),
            "std": _safe_std(vals),
            "n": len(vals),
        }
    return out


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

    seed_rows = _load_seed_rows()
    multiseed_rows = _load_multiseed_rows()

    _write_csv(
        ANALYSIS_OUT / "seed_metrics_clean.csv",
        seed_rows,
        [
            "experiment_name",
            "summary_path",
            "seed",
            "model_name",
            "model_family",
            "variant",
            "allowed_slang_labels",
            "test_accuracy",
            "test_macro_f1",
            "num_test_samples",
            "test_evaluated_from",
        ],
    )

    # Rankings by seed-level macro-f1
    ranking_seed = sorted(seed_rows, key=lambda r: r["test_macro_f1"], reverse=True)
    _write_csv(
        ANALYSIS_OUT / "ranking_seed_macro_f1.csv",
        ranking_seed,
        [
            "experiment_name",
            "seed",
            "model_family",
            "variant",
            "test_accuracy",
            "test_macro_f1",
            "num_test_samples",
            "summary_path",
        ],
    )

    ranking_multi = sorted(
        multiseed_rows, key=lambda r: r["test_macro_f1_mean"], reverse=True
    )
    _write_csv(
        ANALYSIS_OUT / "ranking_multiseed_macro_f1.csv",
        ranking_multi,
        [
            "experiment_name",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_macro_f1_mean",
            "test_macro_f1_std",
            "num_test_samples",
            "summary_path",
            "config_path",
            "seeds",
        ],
    )

    # Slang-heavy subgroup extraction from detailed summaries
    slang_rows = _extract_slang_heavy_rows(seed_rows)
    _write_csv(
        ANALYSIS_OUT / "slang_heavy_subgroup_metrics.csv",
        slang_rows,
        [
            "experiment_name",
            "seed",
            "variant",
            "model_family",
            "slang_heavy_accuracy",
            "slang_heavy_macro_f1",
            "slang_heavy_support",
        ],
    )

    slang_acc_agg = _aggregate_by_experiment(slang_rows, "slang_heavy_accuracy")
    slang_f1_agg = _aggregate_by_experiment(slang_rows, "slang_heavy_macro_f1")

    # Finetune deltas (apples-to-apples slang-heavy)
    def _agg(exp: str, metric: str) -> float:
        source = slang_acc_agg if metric == "acc" else slang_f1_agg
        return float(source.get(exp, {}).get("mean", 0.0))

    finetune_comparison = [
        {
            "comparison": "bert_base_baseline vs bert_finetune_slang_heavy",
            "baseline_slang_heavy_accuracy_mean": _agg("bert_base_baseline", "acc"),
            "finetune_accuracy_mean": float(
                next(
                    (
                        r["test_accuracy_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy"
                    ),
                    0.0,
                )
            ),
            "delta_accuracy_finetune_minus_baseline": float(
                next(
                    (
                        r["test_accuracy_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy"
                    ),
                    0.0,
                )
            )
            - _agg("bert_base_baseline", "acc"),
            "baseline_slang_heavy_macro_f1_mean": _agg("bert_base_baseline", "f1"),
            "finetune_macro_f1_mean": float(
                next(
                    (
                        r["test_macro_f1_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy"
                    ),
                    0.0,
                )
            ),
            "delta_macro_f1_finetune_minus_baseline": float(
                next(
                    (
                        r["test_macro_f1_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy"
                    ),
                    0.0,
                )
            )
            - _agg("bert_base_baseline", "f1"),
        },
        {
            "comparison": "bert_base_mixed vs bert_finetune_slang_heavy_mixed",
            "baseline_slang_heavy_accuracy_mean": _agg("bert_base_mixed", "acc"),
            "finetune_accuracy_mean": float(
                next(
                    (
                        r["test_accuracy_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy_mixed"
                    ),
                    0.0,
                )
            ),
            "delta_accuracy_finetune_minus_baseline": float(
                next(
                    (
                        r["test_accuracy_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy_mixed"
                    ),
                    0.0,
                )
            )
            - _agg("bert_base_mixed", "acc"),
            "baseline_slang_heavy_macro_f1_mean": _agg("bert_base_mixed", "f1"),
            "finetune_macro_f1_mean": float(
                next(
                    (
                        r["test_macro_f1_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy_mixed"
                    ),
                    0.0,
                )
            ),
            "delta_macro_f1_finetune_minus_baseline": float(
                next(
                    (
                        r["test_macro_f1_mean"]
                        for r in multiseed_rows
                        if r["experiment_name"] == "bert_finetune_slang_heavy_mixed"
                    ),
                    0.0,
                )
            )
            - _agg("bert_base_mixed", "f1"),
        },
    ]
    _write_csv(
        ANALYSIS_OUT / "finetune_vs_baseline_slang_heavy.csv",
        finetune_comparison,
        [
            "comparison",
            "baseline_slang_heavy_accuracy_mean",
            "finetune_accuracy_mean",
            "delta_accuracy_finetune_minus_baseline",
            "baseline_slang_heavy_macro_f1_mean",
            "finetune_macro_f1_mean",
            "delta_macro_f1_finetune_minus_baseline",
        ],
    )

    summary = {
        "num_seed_rows": len(seed_rows),
        "num_multiseed_rows": len(multiseed_rows),
        "top_seed_by_macro_f1": ranking_seed[0] if ranking_seed else {},
        "top_multiseed_by_macro_f1": ranking_multi[0] if ranking_multi else {},
        "bertweet_slang_masked_outlier": {
            "experiment": "bertweet_slang_masked",
            "stats": next(
                (
                    r
                    for r in multiseed_rows
                    if r["experiment_name"] == "bertweet_slang_masked"
                ),
                {},
            ),
        },
        "finetune_vs_baseline_slang_heavy": finetune_comparison,
        "generated_files": [
            "seed_metrics_clean.csv",
            "ranking_seed_macro_f1.csv",
            "ranking_multiseed_macro_f1.csv",
            "slang_heavy_subgroup_metrics.csv",
            "finetune_vs_baseline_slang_heavy.csv",
        ],
    }

    with (ANALYSIS_OUT / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
