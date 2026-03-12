"""Data cleaning entry point.

Transforms raw data into cleaned intermediate files under data/interim.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[clean] Raw input directory: {RAW_DIR}")
    print(f"[clean] Interim output directory ready: {INTERIM_DIR}")
    print("[clean] Add cleaning rules here (remove URLs/mentions, preserve emojis/slang/etc).")


if __name__ == "__main__":
    main()
