"""Environment loading helpers for modeling CLIs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> Path | None:
    """Load repository .env once and normalize HF token variable names."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if not env_path.exists():
        return None

    load_dotenv(dotenv_path=env_path, override=False)

    hf_token = os.getenv("HF_TOKEN")
    hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token and not hub_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if hub_token and not hf_token:
        os.environ["HF_TOKEN"] = hub_token

    return env_path
