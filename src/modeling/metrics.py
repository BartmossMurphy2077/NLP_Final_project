"""Lightweight metrics to avoid strict sklearn dependency."""

from __future__ import annotations


def accuracy(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold)


def _precision_recall_f1(gold: list[int], pred: list[int], label: int) -> dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(sum(1 for g in gold if g == label)),
    }


def per_class_report(
    gold: list[int], pred: list[int], label_names: dict[int, str]
) -> dict[str, dict[str, float]]:
    if not gold:
        return {}
    return {
        label_names[label]: _precision_recall_f1(gold, pred, label)
        for label in sorted(label_names.keys())
    }


def macro_f1(gold: list[int], pred: list[int], num_labels: int) -> float:
    if not gold:
        return 0.0

    f1_scores = [_precision_recall_f1(gold, pred, label)["f1"] for label in range(num_labels)]
    return sum(f1_scores) / len(f1_scores)


def confusion_matrix(gold: list[int], pred: list[int], num_labels: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return matrix
