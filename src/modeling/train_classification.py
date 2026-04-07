"""Train and evaluate transformer classifiers with shared pipeline logic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

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


def _latest_checkpoint_dir(out_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for entry in out_dir.glob("checkpoint-step-*"):
        if not entry.is_dir():
            continue
        match = re.match(r"checkpoint-step-(\d+)$", entry.name)
        if match:
            candidates.append((int(match.group(1)), entry))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_resume_checkpoint(cfg: ExperimentConfig, out_dir: Path) -> Path | None:
    if cfg.training.resume_from_checkpoint:
        explicit = Path(cfg.training.resume_from_checkpoint)
        if not explicit.exists():
            raise FileNotFoundError(f"Requested checkpoint does not exist: {explicit}")
        return explicit
    if cfg.training.auto_resume_latest_checkpoint:
        return _latest_checkpoint_dir(out_dir)
    return None


def _parse_step_from_checkpoint_name(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.match(r"checkpoint-step-(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return 0


def _move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool
) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=non_blocking) if hasattr(v, "to") else v
        for k, v in batch.items()
    }


def _evaluate(
    model, loader, device, use_amp: bool, non_blocking: bool
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

            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_checkpoint = _resolve_resume_checkpoint(cfg, out_dir)
    model_source = str(resume_checkpoint) if resume_checkpoint else None

    tokenizer = build_tokenizer(cfg.model)
    model = build_classifier_model(cfg.model, pretrained_name_or_path=model_source).to(
        device
    )

    use_amp = device.type == "cuda" and cfg.training.fp16
    data_num_workers = max(0, cfg.training.num_workers)
    pin_memory = bool(cfg.training.pin_memory and device.type == "cuda")
    non_blocking = bool(pin_memory and device.type == "cuda")

    print(
        f"Using device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f" | fp16={use_amp} | num_workers={data_num_workers} | pin_memory={pin_memory}"
    )
    if resume_checkpoint is not None:
        print(f"Resuming weights from checkpoint: {resume_checkpoint}")

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

    train_loader_kwargs: dict[str, object] = {
        "batch_size": cfg.training.batch_size,
        "shuffle": True,
        "num_workers": data_num_workers,
        "pin_memory": pin_memory,
    }
    if data_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=data_num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=data_num_workers,
        pin_memory=pin_memory,
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

    global_step = _parse_step_from_checkpoint_name(resume_checkpoint)
    best_val_f1 = -1.0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch={epoch + 1}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch.pop("sample_id", None)
            batch.pop("raw_text", None)
            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                outputs = model(**batch)
                loss = outputs.loss / max(1, cfg.training.gradient_accumulation_steps)
            scaler.scale(loss).backward()

            if batch_idx % max(
                1, cfg.training.gradient_accumulation_steps
            ) == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * max(1, cfg.training.gradient_accumulation_steps)
            global_step += 1

            if global_step % cfg.training.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{(train_loss / max(1, global_step)):.4f}"})

            if global_step % cfg.training.save_every_n_steps == 0:
                ckpt_dir = out_dir / f"checkpoint-step-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        val_gold, val_pred, _, _ = _evaluate(
            model, val_loader, device, use_amp=use_amp, non_blocking=non_blocking
        )
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

    test_gold, test_pred, test_ids, test_texts = _evaluate(
        model, test_loader, device, use_amp=use_amp, non_blocking=non_blocking
    )
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
