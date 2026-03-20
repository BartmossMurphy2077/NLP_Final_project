"""Canonical data ingestion from raw sources into a single interim schema."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

COVID_TRAIN_PATH = RAW_DIR / "covid19_nlp_text_classification" / "coronavirus" / "Corona_NLP_train.csv"
COVID_TEST_PATH = RAW_DIR / "covid19_nlp_text_classification" / "coronavirus" / "Corona_NLP_test.csv"
SARCASM_V2_PATH = (
    RAW_DIR
    / "news_headlines_sarcasm"
    / "newsheadlines"
    / "Sarcasm_Headlines_Dataset_v2.json"
)

CANONICAL_OUTPUT_PATH = INTERIM_DIR / "canonical_ingested.csv"
SUMMARY_OUTPUT_PATH = INTERIM_DIR / "canonical_ingestion_summary.json"

FIELDNAMES = ["id", "text_original", "source", "task_label", "task_type"]


def _ingest_covid_file(
    path: Path,
    source_name: str,
    writer: csv.DictWriter,
    summary: dict[str, dict[str, object]],
) -> None:
    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "rows_skipped_empty_text": 0,
        "label_counts": Counter(),
    }

    with path.open(encoding="latin1", newline="") as infile:
        reader = csv.DictReader(infile)
        for idx, row in enumerate(reader, start=1):
            stats["rows_seen"] += 1
            text = (row.get("OriginalTweet") or "").strip()
            if not text:
                stats["rows_skipped_empty_text"] += 1
                continue

            label = (row.get("Sentiment") or "").strip()
            writer.writerow(
                {
                    "id": f"{source_name}_{idx}",
                    "text_original": text,
                    "source": source_name,
                    "task_label": label,
                    "task_type": "sentiment",
                }
            )
            stats["rows_written"] += 1
            stats["label_counts"][label] += 1

    summary[source_name] = {
        "input_path": str(path),
        "rows_seen": stats["rows_seen"],
        "rows_written": stats["rows_written"],
        "rows_skipped_empty_text": stats["rows_skipped_empty_text"],
        "label_counts": dict(stats["label_counts"]),
        "task_type": "sentiment",
    }


def _ingest_sarcasm_file(
    path: Path,
    source_name: str,
    writer: csv.DictWriter,
    summary: dict[str, dict[str, object]],
) -> None:
    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "rows_skipped_empty_text": 0,
        "rows_skipped_invalid_label": 0,
        "label_counts": Counter(),
    }

    label_map = {0: "not_sarcastic", 1: "sarcastic"}

    with path.open(encoding="utf-8", errors="replace") as infile:
        for idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            stats["rows_seen"] += 1
            record = json.loads(line)

            text = (record.get("headline") or "").strip()
            if not text:
                stats["rows_skipped_empty_text"] += 1
                continue

            raw_label = record.get("is_sarcastic")
            if raw_label not in label_map:
                stats["rows_skipped_invalid_label"] += 1
                continue

            label = label_map[raw_label]
            writer.writerow(
                {
                    "id": f"{source_name}_{idx}",
                    "text_original": text,
                    "source": source_name,
                    "task_label": label,
                    "task_type": "sarcasm",
                }
            )
            stats["rows_written"] += 1
            stats["label_counts"][label] += 1

    summary[source_name] = {
        "input_path": str(path),
        "rows_seen": stats["rows_seen"],
        "rows_written": stats["rows_written"],
        "rows_skipped_empty_text": stats["rows_skipped_empty_text"],
        "rows_skipped_invalid_label": stats["rows_skipped_invalid_label"],
        "label_counts": dict(stats["label_counts"]),
        "task_type": "sarcasm",
        "usage_note": "Auxiliary dataset for informal language signals; not the main sentiment benchmark.",
    }


def _assert_inputs_exist() -> None:
    missing = [p for p in [COVID_TRAIN_PATH, COVID_TEST_PATH, SARCASM_V2_PATH] if not p.exists()]
    if missing:
        missing_text = "\n".join(f"- {m}" for m in missing)
        raise FileNotFoundError(f"Missing required raw input files:\n{missing_text}")


def main() -> None:
    _assert_inputs_exist()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    source_summary: dict[str, dict[str, object]] = {}

    with CANONICAL_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        _ingest_covid_file(COVID_TRAIN_PATH, "covid19_nlp_train", writer, source_summary)
        _ingest_covid_file(COVID_TEST_PATH, "covid19_nlp_test", writer, source_summary)
        _ingest_sarcasm_file(SARCASM_V2_PATH, "news_headlines_sarcasm_v2", writer, source_summary)

    total_rows = sum(int(v["rows_written"]) for v in source_summary.values())
    by_task_type = Counter(v["task_type"] for v in source_summary.values())
    by_task_rows = Counter()
    for source_stats in source_summary.values():
        by_task_rows[source_stats["task_type"]] += int(source_stats["rows_written"])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(CANONICAL_OUTPUT_PATH),
        "schema": FIELDNAMES,
        "sources": source_summary,
        "totals": {
            "rows_written": total_rows,
            "sources_by_task_type": dict(by_task_type),
            "rows_by_task_type": dict(by_task_rows),
            "main_benchmark_task_type": "sentiment",
            "auxiliary_task_type": "sarcasm",
        },
    }
    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(f"[collect] Canonical ingestion complete: {CANONICAL_OUTPUT_PATH}")
    print(f"[collect] Ingestion summary: {SUMMARY_OUTPUT_PATH}")
    print(f"[collect] Total rows written: {total_rows}")
    print(f"[collect] Rows by task type: {dict(by_task_rows)}")


if __name__ == "__main__":
    main()
