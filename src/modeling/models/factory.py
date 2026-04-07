"""Factory functions for all supported model backbones."""

from __future__ import annotations

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ..config import ModelConfig


def build_tokenizer(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_classifier_model(
    model_cfg: ModelConfig, pretrained_name_or_path: str | None = None
):
    target = pretrained_name_or_path or model_cfg.name
    model = AutoModelForSequenceClassification.from_pretrained(
        target,
        num_labels=model_cfg.num_labels,
    )
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def build_dapt_model(model_cfg: ModelConfig):
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name)
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    return model
