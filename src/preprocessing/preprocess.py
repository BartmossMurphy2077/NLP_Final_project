"""Preprocessing entry point.

Builds final processed dataset and split manifests for modeling.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] Interim input directory: {INTERIM_DIR}")
    print(f"[preprocess] Processed output directory ready: {PROCESSED_DIR}")
    print(f"[preprocess] Splits directory ready: {SPLITS_DIR}")
    print("[preprocess] Add tokenization, labeling, masking, and split logic here.")


if __name__ == "__main__":
    main()
