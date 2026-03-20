"""Dataset utilities shared across baseline and GPT experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .transforms import apply_ablation, apply_masking_mode


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


@dataclass
class Sample:
    sample_id: str
    text: str
    label: int
    split: str
    text_variant: str
    slang_label: str


class SentimentDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: torch.Tensor,
        ids: list[str],
        texts: list[str],
    ):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids
        self.texts = texts

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["sample_id"] = self.ids[idx]
        item["raw_text"] = self.texts[idx]
        return item


def load_samples(
    data_cfg: DataConfig,
    split_name: str,
    remove_profanity_flag: bool,
    remove_emojis_flag: bool,
    masking_mode: str,
    mask_token: str,
) -> list[Sample]:
    csv_path = Path(data_cfg.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing modeling input CSV: {csv_path}")

    expected_split = data_cfg.split_map[split_name]
    samples: list[Sample] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "id",
            data_cfg.split_column,
            data_cfg.label_column,
            data_cfg.text_column,
            data_cfg.variant_column,
        }
        if data_cfg.allowed_slang_labels:
            required.add(data_cfg.slang_column)
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Modeling input is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            if row[data_cfg.split_column] != expected_split:
                continue
            if row[data_cfg.variant_column] not in data_cfg.allowed_variants:
                continue
            if data_cfg.allowed_slang_labels:
                slang_label = (row.get(data_cfg.slang_column) or "").strip().lower()
                allowed_slang = {
                    value.lower() for value in data_cfg.allowed_slang_labels
                }
                if slang_label not in allowed_slang:
                    continue
            else:
                slang_label = (row.get(data_cfg.slang_column) or "").strip().lower()

            label_text = row[data_cfg.label_column].strip().lower()
            if label_text not in LABEL_TO_ID:
                continue

            text = row[data_cfg.text_column]
            text = apply_masking_mode(text, masking_mode, mask_token)
            text = apply_ablation(text, remove_profanity_flag, remove_emojis_flag)

            samples.append(
                Sample(
                    sample_id=row["id"],
                    text=text,
                    label=LABEL_TO_ID[label_text],
                    split=split_name,
                    text_variant=row[data_cfg.variant_column],
                    slang_label=slang_label,
                )
            )

    return samples


def tokenize_samples(
    samples: list[Sample], tokenizer: object, max_length: int
) -> SentimentDataset:
    texts = [sample.text for sample in samples]
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    ids = [sample.sample_id for sample in samples]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return SentimentDataset(encodings=encodings, labels=labels, ids=ids, texts=texts)
