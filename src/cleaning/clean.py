"""Clean canonical ingested data into a modeling-ready interim file.

Rules implemented:
- Remove URLs and @mentions.
- Preserve emojis, hashtags, profanity, and expressive spelling.
- Normalize whitespace.
- Drop exact duplicates after cleaning.
- Add language metadata (dataset-level assumption).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INPUT_PATH = INTERIM_DIR / "canonical_ingested.csv"
OUTPUT_PATH = INTERIM_DIR / "canonical_cleaned.csv"
SUMMARY_PATH = INTERIM_DIR / "canonical_cleaning_summary.json"

INPUT_FIELDS = ["id", "text_original", "source", "task_label", "task_type"]
OUTPUT_FIELDS = INPUT_FIELDS + ["text_clean", "language", "language_metadata"]

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> tuple[str, int, int]:
    """Remove URLs/mentions, normalize whitespace, preserve other signals."""
    url_hits = len(URL_RE.findall(text))
    mention_hits = len(MENTION_RE.findall(text))

    cleaned = URL_RE.sub(" ", text)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned, url_hits, mention_hits


def _language_metadata_for_source(source: str) -> tuple[str, str]:
    """Return language metadata without external dependencies."""
    # Both datasets are English-focused by construction.
    if source.startswith("covid19_nlp"):
        return "en", "assumed_english_from_dataset_source:covid19_nlp_text_classification"
    if source.startswith("news_headlines_sarcasm"):
        return "en", "assumed_english_from_dataset_source:news_headlines_sarcasm_detection"
    return "unknown", "unknown_source_language"


def _validate_input_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV has no header.")
    missing = [col for col in INPUT_FIELDS if col not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run src/data_collection/collect.py first."
        )

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "rows_dropped_empty_after_cleaning": 0,
        "rows_dropped_exact_duplicates": 0,
        "url_tokens_removed": 0,
        "mentions_removed": 0,
        "rows_by_task_type_before": Counter(),
        "rows_by_task_type_after": Counter(),
        "rows_by_source_before": Counter(),
        "rows_by_source_after": Counter(),
    }

    dedupe_keys: set[tuple[str, str, str, str]] = set()

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as infile, OUTPUT_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        _validate_input_header(reader.fieldnames)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in reader:
            stats["rows_seen"] += 1

            source = (row.get("source") or "").strip()
            task_label = (row.get("task_label") or "").strip()
            task_type = (row.get("task_type") or "").strip()
            text_original = (row.get("text_original") or "").strip()

            stats["rows_by_task_type_before"][task_type] += 1
            stats["rows_by_source_before"][source] += 1

            text_clean, url_hits, mention_hits = _normalize_text(text_original)
            stats["url_tokens_removed"] += url_hits
            stats["mentions_removed"] += mention_hits

            if not text_clean:
                stats["rows_dropped_empty_after_cleaning"] += 1
                continue

            dedupe_key = (text_clean, source, task_label, task_type)
            if dedupe_key in dedupe_keys:
                stats["rows_dropped_exact_duplicates"] += 1
                continue
            dedupe_keys.add(dedupe_key)

            language, language_metadata = _language_metadata_for_source(source)
            writer.writerow(
                {
                    "id": row["id"],
                    "text_original": text_original,
                    "source": source,
                    "task_label": task_label,
                    "task_type": task_type,
                    "text_clean": text_clean,
                    "language": language,
                    "language_metadata": language_metadata,
                }
            )
            stats["rows_written"] += 1
            stats["rows_by_task_type_after"][task_type] += 1
            stats["rows_by_source_after"][source] += 1

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_PATH),
        "output_schema": OUTPUT_FIELDS,
        "rules": {
            "remove_urls": True,
            "remove_mentions": True,
            "normalize_whitespace": True,
            "drop_exact_duplicates": True,
            "dedupe_key": ["text_clean", "source", "task_label", "task_type"],
            "preserve_signals": [
                "emojis",
                "hashtags",
                "profanity",
                "expressive_spelling",
            ],
        },
        "counts": {
            "rows_seen": stats["rows_seen"],
            "rows_written": stats["rows_written"],
            "rows_dropped_empty_after_cleaning": stats["rows_dropped_empty_after_cleaning"],
            "rows_dropped_exact_duplicates": stats["rows_dropped_exact_duplicates"],
            "url_tokens_removed": stats["url_tokens_removed"],
            "mentions_removed": stats["mentions_removed"],
            "rows_by_task_type_before": dict(stats["rows_by_task_type_before"]),
            "rows_by_task_type_after": dict(stats["rows_by_task_type_after"]),
            "rows_by_source_before": dict(stats["rows_by_source_before"]),
            "rows_by_source_after": dict(stats["rows_by_source_after"]),
        },
        "language_metadata_note": (
            "Language is set using source-level English assumptions because no external "
            "language-ID model is used at this stage."
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(f"[clean] Cleaning complete: {OUTPUT_PATH}")
    print(f"[clean] Cleaning summary: {SUMMARY_PATH}")
    print(f"[clean] Rows seen: {stats['rows_seen']}")
    print(f"[clean] Rows written: {stats['rows_written']}")
    print(f"[clean] Rows dropped (empty after cleaning): {stats['rows_dropped_empty_after_cleaning']}")
    print(f"[clean] Rows dropped (exact duplicates): {stats['rows_dropped_exact_duplicates']}")
    print(f"[clean] URL tokens removed: {stats['url_tokens_removed']}")
    print(f"[clean] Mentions removed: {stats['mentions_removed']}")


if __name__ == "__main__":
    main()
