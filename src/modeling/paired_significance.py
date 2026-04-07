"""Compute paired significance tests between two prediction files on matched test examples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random


def _load_predictions(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Prediction file has no rows: {path}")

    key_field = "base_id" if "base_id" in rows[0] else "id"
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(key_field, "")
        if not key:
            continue
        by_key[key] = row

    return key_field, by_key


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mcnemar_pvalue_chi_square(n01: int, n10: int) -> float:
    total = n01 + n10
    if total == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1.0) ** 2 / total
    # For df=1, chi-square survival function = erfc(sqrt(x/2)).
    return math.erfc(math.sqrt(stat / 2.0))


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n == 0:
        return 1.0

    k_small = min(k, n - k)
    p_low = sum(math.comb(n, i) for i in range(0, k_small + 1)) / (2**n)
    p_value = min(1.0, 2.0 * p_low)
    return p_value


def _bootstrap_accuracy_diff(
    correctness_a: list[int], correctness_b: list[int], n_bootstrap: int, seed: int
) -> dict[str, float]:
    n = len(correctness_a)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        acc_a = sum(correctness_a[i] for i in sample_idx) / n
        acc_b = sum(correctness_b[i] for i in sample_idx) / n
        diffs.append(acc_b - acc_a)

    diffs.sort()
    low_idx = int(0.025 * (n_bootstrap - 1))
    high_idx = int(0.975 * (n_bootstrap - 1))
    return {
        "mean": sum(diffs) / len(diffs),
        "ci95_low": diffs[low_idx],
        "ci95_high": diffs[high_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired significance tests over matched prediction rows."
    )
    parser.add_argument(
        "--a", required=True, help="Path to predictions CSV for model A."
    )
    parser.add_argument(
        "--b", required=True, help="Path to predictions CSV for model B."
    )
    parser.add_argument(
        "--output",
        default="outputs/modeling/paired_significance.json",
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for accuracy-difference CI.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    key_a, pred_a = _load_predictions(Path(args.a))
    key_b, pred_b = _load_predictions(Path(args.b))

    common_keys = sorted(set(pred_a).intersection(pred_b))
    if not common_keys:
        raise ValueError("No matched prediction keys between A and B.")

    n_gold_conflicts = 0
    correctness_a: list[int] = []
    correctness_b: list[int] = []

    n_a_correct_b_wrong = 0
    n_a_wrong_b_correct = 0

    for key in common_keys:
        row_a = pred_a[key]
        row_b = pred_b[key]

        gold_a = row_a["gold_label"]
        gold_b = row_b["gold_label"]
        if gold_a != gold_b:
            n_gold_conflicts += 1
            continue

        a_correct = int(row_a["pred_label"] == gold_a)
        b_correct = int(row_b["pred_label"] == gold_a)

        correctness_a.append(a_correct)
        correctness_b.append(b_correct)

        if a_correct == 1 and b_correct == 0:
            n_a_correct_b_wrong += 1
        elif a_correct == 0 and b_correct == 1:
            n_a_wrong_b_correct += 1

    matched_n = len(correctness_a)
    acc_a = _safe_div(sum(correctness_a), matched_n)
    acc_b = _safe_div(sum(correctness_b), matched_n)

    discordant = n_a_correct_b_wrong + n_a_wrong_b_correct
    sign_test_pvalue = _binom_two_sided_pvalue(n_a_wrong_b_correct, discordant)
    mcnemar_pvalue = _mcnemar_pvalue_chi_square(
        n_a_correct_b_wrong, n_a_wrong_b_correct
    )

    bootstrap = _bootstrap_accuracy_diff(
        correctness_a,
        correctness_b,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    result = {
        "a_path": args.a,
        "b_path": args.b,
        "key_field_a": key_a,
        "key_field_b": key_b,
        "matched_examples": matched_n,
        "excluded_gold_conflicts": n_gold_conflicts,
        "accuracy": {
            "a": acc_a,
            "b": acc_b,
            "difference_b_minus_a": acc_b - acc_a,
        },
        "discordant_pairs": {
            "a_correct_b_wrong": n_a_correct_b_wrong,
            "a_wrong_b_correct": n_a_wrong_b_correct,
            "total": discordant,
        },
        "tests": {
            "mcnemar_chi_square_pvalue": mcnemar_pvalue,
            "sign_test_two_sided_pvalue": sign_test_pvalue,
        },
        "bootstrap_accuracy_diff_b_minus_a": bootstrap,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
