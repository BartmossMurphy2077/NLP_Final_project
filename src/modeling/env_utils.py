"""Environment loading helpers for modeling CLIs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> Path | None:
    """Load repository .env once and normalize HF token variable names."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

    # Reduce CUDA memory fragmentation on long multi-run sessions unless user overrides.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Keep BLAS/OpenMP thread fan-out low to avoid native memory-allocation crashes.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    hf_token = os.getenv("HF_TOKEN")
    hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token and not hub_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if hub_token and not hf_token:
        os.environ["HF_TOKEN"] = hub_token

    return env_path if env_path.exists() else None
