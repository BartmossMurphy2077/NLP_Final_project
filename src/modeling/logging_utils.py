"""Prediction and error analysis logging utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_predictions(path: str | Path, rows: list[dict[str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_rows(
    ids: list[str],
    base_ids: list[str],
    texts: list[str],
    gold_labels: list[int],
    pred_labels: list[int],
    split_name: str,
    text_variants: list[str],
    slang_labels: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []

    for sample_id, base_id, text, gold, pred, text_variant, slang_label in zip(
        ids, base_ids, texts, gold_labels, pred_labels, text_variants, slang_labels
    ):
        row = {
            "id": sample_id,
            "base_id": base_id,
            "split": split_name,
            "text_variant": text_variant,
            "slang_label": slang_label,
            "text": text,
            "gold_label": ID_TO_LABEL[gold],
            "pred_label": ID_TO_LABEL[pred],
            "is_misclassified": str(int(gold != pred)),
        }
        all_rows.append(row)
        if gold != pred:
            bad_rows.append(row)

    return all_rows, bad_rows
