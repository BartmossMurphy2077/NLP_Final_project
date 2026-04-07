"""Configuration schema and loader for modeling experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    input_csv: str
    split_column: str = "split"
    label_column: str = "sentiment_label"
    slang_column: str = "slang_label"
    text_column: str = "text_for_model"
    variant_column: str = "text_variant"
    allowed_variants: list[str] = field(default_factory=lambda: ["original"])
    allowed_slang_labels: list[str] = field(default_factory=list)
    split_map: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "val": "val", "test": "test"}
    )


@dataclass
class AblationConfig:
    remove_profanity: bool = False
    remove_emojis: bool = False


@dataclass
class MaskingConfig:
    mode: str = "none"
    mask_token: str = "[SLANG]"


@dataclass
class ModelConfig:
    name: str
    model_type: str
    tokenizer_name: str
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    output_dir: str
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    fp16: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    resume_from_checkpoint: str | None = None
    auto_resume_latest_checkpoint: bool = True
    save_every_n_steps: int = 200
    log_every_n_steps: int = 50


@dataclass
class LoggingConfig:
    run_name: str
    output_dir: str
    save_predictions: bool = True
    save_misclassified: bool = True


@dataclass
class DaptConfig:
    enabled: bool = False
    corpus_csv: str = ""
    text_column: str = "text_clean"
    block_size: int = 128
    mlm_probability: float = 0.15


@dataclass
class ExperimentConfig:
    experiment_name: str
    task: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    ablation: AblationConfig = field(default_factory=AblationConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    dapt: DaptConfig = field(default_factory=DaptConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config parsing. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must parse into a mapping: {path}")
    return payload


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    raw = _read_yaml(cfg_path)

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    logging = LoggingConfig(**raw["logging"])
    ablation = AblationConfig(**raw.get("ablation", {}))
    masking = MaskingConfig(**raw.get("masking", {}))
    dapt = DaptConfig(**raw.get("dapt", {}))

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        task=raw.get("task", "classification"),
        data=data,
        model=model,
        training=training,
        logging=logging,
        ablation=ablation,
        masking=masking,
        dapt=dapt,
    )
