"""Domain-adaptive pretraining for GPT-style models on social media corpora."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import load_experiment_config
from .env_utils import load_project_env
from .models import build_dapt_model, build_tokenizer
from .seed import set_seed


def _load_corpus(corpus_csv: Path, text_column: str) -> Dataset:
    if not corpus_csv.exists():
        raise FileNotFoundError(f"Missing DAPT corpus CSV: {corpus_csv}")

    texts: list[str] = []
    with corpus_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{text_column}' not found in {corpus_csv}")
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("DAPT corpus is empty after filtering blank rows.")

    return Dataset.from_dict({"text": texts})


def run_dapt(config_path: str) -> None:
    cfg = load_experiment_config(config_path)
    if not cfg.dapt.enabled:
        raise ValueError(
            "DAPT config is not enabled. Set dapt.enabled: true in config."
        )

    set_seed(cfg.training.seed)

    tokenizer = build_tokenizer(cfg.model)
    model = build_dapt_model(cfg.model)

    corpus = _load_corpus(Path(cfg.dapt.corpus_csv), cfg.dapt.text_column)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.dapt.block_size,
        )

    tokenized = corpus.map(_tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        weight_decay=cfg.training.weight_decay,
        save_steps=cfg.training.save_every_n_steps,
        save_total_limit=max(1, int(getattr(cfg.training, "save_total_limit", 2))),
        logging_steps=cfg.training.log_every_n_steps,
        seed=cfg.training.seed,
        fp16=cfg.training.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    out_dir = Path(cfg.training.output_dir) / "dapt_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(
        description="Run domain-adaptive pretraining for GPT."
    )
    parser.add_argument("--config", required=True, help="Path to DAPT YAML config.")
    args = parser.parse_args()

    run_dapt(args.config)


if __name__ == "__main__":
    main()
