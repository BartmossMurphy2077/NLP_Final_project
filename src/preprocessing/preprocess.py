"""Preprocess cleaned text for sentiment experiments with slang-focused variants."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

INPUT_PATH = INTERIM_DIR / "canonical_cleaned.csv"
SENTIMENT_OUTPUT_PATH = PROCESSED_DIR / "sentiment_preprocessed.csv"
SENTIMENT_VARIANTS_OUTPUT_PATH = PROCESSED_DIR / "sentiment_preprocessed_variants.csv"
SARCASM_AUX_OUTPUT_PATH = PROCESSED_DIR / "sarcasm_auxiliary_preprocessed.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DIR / "preprocessing_summary.json"
SPLIT_MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"
TRAIN_IDS_PATH = SPLITS_DIR / "train_ids.txt"
VAL_IDS_PATH = SPLITS_DIR / "val_ids.txt"
TEST_IDS_PATH = SPLITS_DIR / "test_ids.txt"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

SENTIMENT_FIELDS = [
    "id",
    "text_original",
    "text_clean",
    "source",
    "sentiment_label",
    "slang_label",
    "split",
    "task_type",
    "slang_term_count",
    "informal_signal_count",
]

SENTIMENT_VARIANT_FIELDS = [
    "id",
    "base_id",
    "text_original",
    "text_clean",
    "source",
    "sentiment_label",
    "slang_label",
    "split",
    "task_type",
    "text_variant",
    "text_for_model",
]

SARCASM_AUX_FIELDS = [
    "id",
    "text_original",
    "text_clean",
    "source",
    "task_label",
    "slang_label",
    "split",
    "task_type",
    "slang_term_count",
    "informal_signal_count",
]

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']*")
HASHTAG_RE = re.compile(r"#\w+")
ELONGATED_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b", re.IGNORECASE)
REPEATED_PUNCT_RE = re.compile(r"[!?]{2,}")
# Covers common emoji blocks used in modern social text.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)

SLANG_LEXICON = {
    "af",
    "aint",
    "bae",
    "bc",
    "bet",
    "brb",
    "bro",
    "bruh",
    "btw",
    "cap",
    "clapback",
    "dm",
    "dope",
    "finna",
    "fire",
    "fr",
    "fomo",
    "goat",
    "gonna",
    "idc",
    "idk",
    "ikr",
    "imo",
    "imho",
    "irl",
    "lit",
    "lmao",
    "lmfao",
    "lol",
    "lowkey",
    "mid",
    "nah",
    "ngl",
    "noob",
    "omg",
    "pls",
    "rn",
    "salty",
    "sis",
    "slay",
    "smh",
    "stan",
    "sus",
    "tho",
    "thx",
    "tbh",
    "tryna",
    "u",
    "ur",
    "vibe",
    "vibing",
    "wanna",
    "wildin",
    "wtf",
    "yall",
    "yeet",
}


def _normalize_token(token: str) -> str:
    lowered = token.lower()
    return re.sub(r"(^[^\w#']+|[^\w#']+$)", "", lowered)


def _is_mask_candidate(token: str) -> bool:
    norm = _normalize_token(token)
    core = norm[1:] if norm.startswith("#") else norm

    if not norm:
        return False
    if EMOJI_RE.search(token):
        return True
    if norm.startswith("#"):
        return True
    if core in SLANG_LEXICON:
        return True
    if ELONGATED_RE.search(core):
        return True
    if REPEATED_PUNCT_RE.search(token):
        return True
    return False


def _analyze_slang(text_clean: str) -> dict[str, object]:
    tokens = text_clean.split()
    candidate_indices: list[int] = []
    slang_terms = 0

    for idx, token in enumerate(tokens):
        norm = _normalize_token(token)
        core = norm[1:] if norm.startswith("#") else norm
        if core in SLANG_LEXICON:
            slang_terms += 1
        if _is_mask_candidate(token):
            candidate_indices.append(idx)

    emoji_hits = len(EMOJI_RE.findall(text_clean))
    hashtag_hits = len(HASHTAG_RE.findall(text_clean))
    expressive_hits = len(ELONGATED_RE.findall(text_clean))
    repeated_punct_hits = len(REPEATED_PUNCT_RE.findall(text_clean))

    informal_signal_count = (
        slang_terms + emoji_hits + expressive_hits + repeated_punct_hits + hashtag_hits
    )

    slang_label = (
        "slang_heavy"
        if (
            slang_terms >= 1
            or emoji_hits >= 1
            or expressive_hits >= 1
            or repeated_punct_hits >= 1
            or hashtag_hits >= 2
        )
        else "formal"
    )

    return {
        "slang_label": slang_label,
        "slang_term_count": slang_terms,
        "informal_signal_count": informal_signal_count,
        "tokens": tokens,
        "candidate_indices": candidate_indices,
    }


def _build_variant_text(tokens: list[str], candidate_indices: list[int], mode: str) -> str:
    if mode == "original":
        return " ".join(tokens).strip()

    replaced = tokens[:]
    if mode == "slang_masked":
        to_mask = set(candidate_indices)
    elif mode == "mixed":
        to_mask = {idx for pos, idx in enumerate(candidate_indices) if pos % 2 == 0}
    else:
        raise ValueError(f"Unknown variant mode: {mode}")

    for idx in to_mask:
        replaced[idx] = "[SLANG]"
    return " ".join(replaced).strip()


def _stable_hash(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _assign_splits_stratified(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["task_label"], row["slang_label"])
        grouped[key].append(row)

    assignments: dict[str, str] = {}
    for key in sorted(grouped.keys()):
        group_rows = grouped[key]
        group_rows_sorted = sorted(group_rows, key=lambda r: _stable_hash(seed, r["id"]))
        n = len(group_rows_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        for row in group_rows_sorted[:n_train]:
            assignments[row["id"]] = "train"
        for row in group_rows_sorted[n_train : n_train + n_val]:
            assignments[row["id"]] = "val"
        for row in group_rows_sorted[n_train + n_val : n_train + n_val + n_test]:
            assignments[row["id"]] = "test"

    if len(assignments) != len(rows):
        raise RuntimeError("Split assignment count mismatch.")
    return assignments


def _write_id_list(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row_id in ids:
            f.write(f"{row_id}\n")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}. Run src/cleaning/clean.py first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    sentiment_rows: list[dict[str, str]] = []
    sarcasm_rows: list[dict[str, str]] = []
    task_type_counts = Counter()

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        required = {"id", "text_original", "text_clean", "source", "task_label", "task_type"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        for row in reader:
            task_type = (row["task_type"] or "").strip()
            text_clean = (row["text_clean"] or "").strip()
            if not text_clean:
                continue

            analysis = _analyze_slang(text_clean)
            row_base = {
                "id": row["id"],
                "text_original": row["text_original"],
                "text_clean": text_clean,
                "source": row["source"],
                "task_label": row["task_label"],
                "task_type": task_type,
                "slang_label": analysis["slang_label"],
                "slang_term_count": str(analysis["slang_term_count"]),
                "informal_signal_count": str(analysis["informal_signal_count"]),
                "tokens": analysis["tokens"],
                "candidate_indices": analysis["candidate_indices"],
            }

            if task_type == "sentiment":
                sentiment_rows.append(row_base)
            elif task_type == "sarcasm":
                sarcasm_rows.append(row_base)
            task_type_counts[task_type] += 1

    split_assignments = _assign_splits_stratified(
        rows=sentiment_rows,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    for row in sentiment_rows:
        row["split"] = split_assignments[row["id"]]
        row["sentiment_label"] = row["task_label"]

    for row in sarcasm_rows:
        row["split"] = "auxiliary"

    train_ids = sorted([row["id"] for row in sentiment_rows if row["split"] == "train"])
    val_ids = sorted([row["id"] for row in sentiment_rows if row["split"] == "val"])
    test_ids = sorted([row["id"] for row in sentiment_rows if row["split"] == "test"])

    _write_id_list(TRAIN_IDS_PATH, train_ids)
    _write_id_list(VAL_IDS_PATH, val_ids)
    _write_id_list(TEST_IDS_PATH, test_ids)

    with SENTIMENT_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SENTIMENT_FIELDS)
        writer.writeheader()
        for row in sentiment_rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "text_original": row["text_original"],
                    "text_clean": row["text_clean"],
                    "source": row["source"],
                    "sentiment_label": row["sentiment_label"],
                    "slang_label": row["slang_label"],
                    "split": row["split"],
                    "task_type": row["task_type"],
                    "slang_term_count": row["slang_term_count"],
                    "informal_signal_count": row["informal_signal_count"],
                }
            )

    with SENTIMENT_VARIANTS_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SENTIMENT_VARIANT_FIELDS)
        writer.writeheader()
        for row in sentiment_rows:
            tokens = row["tokens"]
            candidate_indices = row["candidate_indices"]
            for variant in ["original", "slang_masked", "mixed"]:
                text_for_model = _build_variant_text(tokens, candidate_indices, variant)
                writer.writerow(
                    {
                        "id": f"{row['id']}__{variant}",
                        "base_id": row["id"],
                        "text_original": row["text_original"],
                        "text_clean": row["text_clean"],
                        "source": row["source"],
                        "sentiment_label": row["sentiment_label"],
                        "slang_label": row["slang_label"],
                        "split": row["split"],
                        "task_type": row["task_type"],
                        "text_variant": variant,
                        "text_for_model": text_for_model,
                    }
                )

    with SARCASM_AUX_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SARCASM_AUX_FIELDS)
        writer.writeheader()
        for row in sarcasm_rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "text_original": row["text_original"],
                    "text_clean": row["text_clean"],
                    "source": row["source"],
                    "task_label": row["task_label"],
                    "slang_label": row["slang_label"],
                    "split": row["split"],
                    "task_type": row["task_type"],
                    "slang_term_count": row["slang_term_count"],
                    "informal_signal_count": row["informal_signal_count"],
                }
            )

    split_manifest = {
        "dataset_version": "v0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "stratify_by": ["sentiment_label", "slang_label"],
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "files": {
            "train": str(TRAIN_IDS_PATH.relative_to(PROJECT_ROOT)),
            "val": str(VAL_IDS_PATH.relative_to(PROJECT_ROOT)),
            "test": str(TEST_IDS_PATH.relative_to(PROJECT_ROOT)),
        },
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total_sentiment_rows": len(sentiment_rows),
        },
    }
    with SPLIT_MANIFEST_PATH.open("w", encoding="utf-8") as outfile:
        json.dump(split_manifest, outfile, indent=2)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_PATH),
        "outputs": {
            "sentiment_base": str(SENTIMENT_OUTPUT_PATH),
            "sentiment_variants": str(SENTIMENT_VARIANTS_OUTPUT_PATH),
            "sarcasm_auxiliary": str(SARCASM_AUX_OUTPUT_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        },
        "counts": {
            "input_rows_by_task_type": dict(task_type_counts),
            "sentiment_rows": len(sentiment_rows),
            "sarcasm_rows": len(sarcasm_rows),
            "sentiment_variant_rows": len(sentiment_rows) * 3,
            "split_counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "slang_label_distribution_sentiment": dict(
                Counter(row["slang_label"] for row in sentiment_rows)
            ),
            "slang_label_distribution_sarcasm": dict(
                Counter(row["slang_label"] for row in sarcasm_rows)
            ),
        },
        "notes": {
            "main_benchmark_task_type": "sentiment",
            "auxiliary_task_type": "sarcasm",
            "variants": ["original", "slang_masked", "mixed"],
            "mask_token": "[SLANG]",
        },
    }
    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)

    print(f"[preprocess] Input file: {INPUT_PATH}")
    print(f"[preprocess] Sentiment base output: {SENTIMENT_OUTPUT_PATH}")
    print(f"[preprocess] Sentiment variants output: {SENTIMENT_VARIANTS_OUTPUT_PATH}")
    print(f"[preprocess] Sarcasm auxiliary output: {SARCASM_AUX_OUTPUT_PATH}")
    print(f"[preprocess] Split manifest: {SPLIT_MANIFEST_PATH}")
    print(
        f"[preprocess] Sentiment splits -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}"
    )


if __name__ == "__main__":
    main()
