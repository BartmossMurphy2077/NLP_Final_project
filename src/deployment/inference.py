"""Inference utilities for serving the final sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.cleaning.clean import _normalize_text
from src.preprocessing.preprocess import _analyze_slang, _build_variant_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "MODELS_FINAL"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(frozen=True)
class ModelVariantSpec:
    key: str
    title: str
    family_dir: str
    text_variant: str
    description: str


@dataclass(frozen=True)
class ResolvedModelVariant:
    spec: ModelVariantSpec
    model_dir: Path | None
    available: bool
    reason: str


@dataclass(frozen=True)
class PredictionResult:
    model_key: str
    model_title: str
    model_dir: str
    input_text: str
    cleaned_text: str
    prepared_text: str
    detected_slang_label: str
    predicted_label: str
    scores: dict[str, float]


VARIANT_SPECS = (
    ModelVariantSpec(
        key="bert_base",
        title="BERT Base",
        family_dir="bert_base",
        text_variant="original",
        description="Baseline sentiment model using cleaned text without slang masking.",
    ),
    ModelVariantSpec(
        key="bert_finetune_slang",
        title="BERT Slang Masked",
        family_dir="bert_finetune_slang",
        text_variant="slang_masked",
        description="Applies slang-aware masking before classification.",
    ),
    ModelVariantSpec(
        key="bert_finetune_slang_mixed",
        title="BERT Slang Mixed",
        family_dir="bert_finetune_slang_mixed",
        text_variant="mixed",
        description="Masks alternating slang-heavy tokens before classification.",
    ),
)


def _has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if not (model_dir / "config.json").exists():
        return False
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))


def _candidate_model_dirs(family_root: Path) -> list[Path]:
    candidates: list[Path] = []

    top_level = family_root / "final_model"
    if top_level.exists():
        candidates.append(top_level)

    seed_dirs = sorted(
        (
            seed_dir / "final_model"
            for seed_dir in family_root.glob("seed_*")
            if seed_dir.is_dir()
        ),
        key=lambda path: path.parent.name,
    )
    candidates.extend(seed_dirs)
    return candidates


def discover_model_variants(models_root: Path = MODELS_ROOT) -> list[ResolvedModelVariant]:
    resolved: list[ResolvedModelVariant] = []

    for spec in VARIANT_SPECS:
        family_root = models_root / spec.family_dir
        if not family_root.exists():
            resolved.append(
                ResolvedModelVariant(
                    spec=spec,
                    model_dir=None,
                    available=False,
                    reason=f"Missing family directory: {family_root}",
                )
            )
            continue

        chosen_dir: Path | None = None
        for candidate in _candidate_model_dirs(family_root):
            if _has_model_weights(candidate):
                chosen_dir = candidate
                break

        if chosen_dir is None:
            resolved.append(
                ResolvedModelVariant(
                    spec=spec,
                    model_dir=None,
                    available=False,
                    reason="No final_model directory with weights was found.",
                )
            )
            continue

        resolved.append(
            ResolvedModelVariant(
                spec=spec,
                model_dir=chosen_dir,
                available=True,
                reason="OK",
            )
        )

    return resolved


def available_model_variants(models_root: Path = MODELS_ROOT) -> list[ResolvedModelVariant]:
    return [variant for variant in discover_model_variants(models_root) if variant.available]


def unavailable_model_variants(models_root: Path = MODELS_ROOT) -> list[ResolvedModelVariant]:
    return [variant for variant in discover_model_variants(models_root) if not variant.available]


def _prepare_text(input_text: str, text_variant: str) -> tuple[str, str, str]:
    cleaned_text, _, _ = _normalize_text(input_text)
    if not cleaned_text:
        raise ValueError("Text is empty after removing URLs, mentions, and extra whitespace.")

    analysis = _analyze_slang(cleaned_text)
    prepared_text = _build_variant_text(
        analysis["tokens"],
        analysis["candidate_indices"],
        text_variant,
    )
    return cleaned_text, prepared_text, str(analysis["slang_label"])


class InferenceService:
    """Loads and caches local transformer models for sentiment inference."""

    def __init__(self, models_root: Path = MODELS_ROOT):
        self.models_root = models_root
        self._variants = {item.spec.key: item for item in discover_model_variants(models_root)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def variants(self) -> dict[str, ResolvedModelVariant]:
        return self._variants

    def available_variants(self) -> list[ResolvedModelVariant]:
        return [variant for variant in self._variants.values() if variant.available]

    def unavailable_variants(self) -> list[ResolvedModelVariant]:
        return [variant for variant in self._variants.values() if not variant.available]

    @lru_cache(maxsize=8)
    def _load_bundle(self, model_key: str):
        variant = self._variants.get(model_key)
        if variant is None:
            raise KeyError(f"Unknown model variant: {model_key}")
        if not variant.available or variant.model_dir is None:
            raise ValueError(f"Model variant is unavailable: {model_key}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(variant.model_dir),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(variant.model_dir),
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def predict(self, text: str, model_key: str) -> PredictionResult:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Please enter some text to classify.")

        variant = self._variants.get(model_key)
        if variant is None:
            raise KeyError(f"Unknown model variant: {model_key}")
        if not variant.available or variant.model_dir is None:
            raise ValueError(f"Model variant is unavailable: {variant.spec.title}")

        cleaned_text, prepared_text, slang_label = _prepare_text(
            stripped,
            variant.spec.text_variant,
        )
        tokenizer, model = self._load_bundle(model_key)

        encoded = tokenizer(
            prepared_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits[0], dim=-1).cpu().tolist()

        scores = {
            LABEL_NAMES[idx]: round(float(score), 4)
            for idx, score in enumerate(probabilities)
        }
        predicted_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])

        return PredictionResult(
            model_key=model_key,
            model_title=variant.spec.title,
            model_dir=str(variant.model_dir),
            input_text=stripped,
            cleaned_text=cleaned_text,
            prepared_text=prepared_text,
            detected_slang_label=slang_label,
            predicted_label=LABEL_NAMES[predicted_idx],
            scores=scores,
        )

