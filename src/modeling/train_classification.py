"""Train and evaluate transformer classifiers with shared pipeline logic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig, load_experiment_config
from .dataset import load_samples, tokenize_samples
from .logging_utils import (
    build_prediction_rows,
    ensure_dir,
    write_json,
    write_predictions,
)
from .metrics import accuracy, macro_f1
from .models import build_classifier_model, build_tokenizer
from .seed import set_seed


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(
    model, loader, device
) -> tuple[list[int], list[int], list[str], list[str]]:
    model.eval()
    all_gold: list[int] = []
    all_pred: list[int] = []
    all_ids: list[str] = []
    all_texts: list[str] = []

    with torch.no_grad():
        for batch in loader:
            ids = batch.pop("sample_id")
            texts = batch.pop("raw_text")
            labels = batch["labels"]
            all_gold.extend(labels.tolist())
            all_ids.extend(ids)
            all_texts.extend(texts)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            all_pred.extend(preds)

    return all_gold, all_pred, all_ids, all_texts


def train_from_config(config_path: str) -> dict[str, float]:
    cfg: ExperimentConfig = load_experiment_config(config_path)
    set_seed(cfg.training.seed)

    out_dir = ensure_dir(cfg.training.output_dir)
    logs_dir = ensure_dir(cfg.logging.output_dir)
    device = _device()

    tokenizer = build_tokenizer(cfg.model)
    model = build_classifier_model(cfg.model).to(device)

    train_samples = load_samples(
        data_cfg=cfg.data,
        split_name="train",
        remove_profanity_flag=cfg.ablation.remove_profanity,
        remove_emojis_flag=cfg.ablation.remove_emojis,
        masking_mode=cfg.masking.mode,
        mask_token=cfg.masking.mask_token,
    )
    val_samples = load_samples(
        data_cfg=cfg.data,
        split_name="val",
        remove_profanity_flag=cfg.ablation.remove_profanity,
        remove_emojis_flag=cfg.ablation.remove_emojis,
        masking_mode=cfg.masking.mode,
        mask_token=cfg.masking.mask_token,
    )
    test_samples = load_samples(
        data_cfg=cfg.data,
        split_name="test",
        remove_profanity_flag=cfg.ablation.remove_profanity,
        remove_emojis_flag=cfg.ablation.remove_emojis,
        masking_mode=cfg.masking.mode,
        mask_token=cfg.masking.mask_token,
    )

    train_ds = tokenize_samples(train_samples, tokenizer, cfg.model.max_length)
    val_ds = tokenize_samples(val_samples, tokenizer, cfg.model.max_length)
    test_ds = tokenize_samples(test_samples, tokenizer, cfg.model.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.training.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.training.warmup_ratio),
        num_training_steps=total_steps,
    )

    global_step = 0
    best_val_f1 = -1.0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch={epoch + 1}", leave=False)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch.pop("sample_id", None)
            batch.pop("raw_text", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, cfg.training.gradient_accumulation_steps)
            loss.backward()

            if batch_idx % max(
                1, cfg.training.gradient_accumulation_steps
            ) == 0 or batch_idx == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * max(1, cfg.training.gradient_accumulation_steps)
            global_step += 1

            if global_step % cfg.training.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{(train_loss / max(1, global_step)):.4f}"})

            if global_step % cfg.training.save_every_n_steps == 0:
                ckpt_dir = out_dir / f"checkpoint-step-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        val_gold, val_pred, _, _ = _evaluate(model, val_loader, device)
        val_acc = accuracy(val_gold, val_pred)
        val_f1 = macro_f1(val_gold, val_pred, cfg.model.num_labels)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_dir = out_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        write_json(
            logs_dir / f"epoch_{epoch + 1}_metrics.json",
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, len(train_loader)),
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            },
        )

    test_gold, test_pred, test_ids, test_texts = _evaluate(model, test_loader, device)
    test_acc = accuracy(test_gold, test_pred)
    test_f1 = macro_f1(test_gold, test_pred, cfg.model.num_labels)

    prediction_rows, misclassified_rows = build_prediction_rows(
        ids=test_ids,
        texts=test_texts,
        gold_labels=test_gold,
        pred_labels=test_pred,
        split_name="test",
    )

    if cfg.logging.save_predictions:
        write_predictions(logs_dir / "predictions_test.csv", prediction_rows)
    if cfg.logging.save_misclassified:
        write_predictions(logs_dir / "misclassified_test.csv", misclassified_rows)

    summary = {
        "experiment_name": cfg.experiment_name,
        "config": asdict(cfg),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "num_test_samples": len(test_gold),
    }
    write_json(logs_dir / "run_summary.json", summary)

    final_dir = Path(cfg.training.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    return {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "num_test_samples": float(len(test_gold)),
    }
