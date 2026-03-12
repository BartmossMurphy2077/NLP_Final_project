"""Data collection entry point.

Collects source data into data/raw and records collection metadata.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[collect] Raw data directory ready: {RAW_DIR}")
    print("[collect] Add source-specific collection logic here.")


if __name__ == "__main__":
    main()
