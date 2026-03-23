"""Extract TF-IDF features from preprocessed sentiment data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SENTIMENT_INPUT_PATH = PROCESSED_DIR / "sentiment_preprocessed.csv"
TRAIN_IDS_PATH = SPLITS_DIR / "train_ids.txt"
VAL_IDS_PATH = SPLITS_DIR / "val_ids.txt"
TEST_IDS_PATH = SPLITS_DIR / "test_ids.txt"

FEATURE_OUTPUT_DIR = PROCESSED_DIR / "features"


def _load_split_ids(path: Path) -> set[str]:
    """Load split IDs from text file."""
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def _load_sentiment_data() -> dict[str, dict]:
    """Load sentiment preprocessed CSV into memory keyed by ID."""
    data = {}
    with SENTIMENT_INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["id"]] = row
    return data


def main() -> None:
    # Verify input exists
    if not SENTIMENT_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {SENTIMENT_INPUT_PATH}. Run src/preprocessing/preprocess.py first."
        )

    # Load split IDs
    train_ids = _load_split_ids(TRAIN_IDS_PATH)
    val_ids = _load_split_ids(VAL_IDS_PATH)
    test_ids = _load_split_ids(TEST_IDS_PATH)

    # Load all sentiment data
    all_data = _load_sentiment_data()

    # Determine which text column to use
    # From preprocessing_report.md and preprocess.py, text_clean is the main modeling text
    text_column = "text_clean"
    
    # Separate by split
    train_data = [all_data[id_] for id_ in train_ids if id_ in all_data]
    val_data = [all_data[id_] for id_ in val_ids if id_ in all_data]
    test_data = [all_data[id_] for id_ in test_ids if id_ in all_data]

    # Extract texts and labels
    X_train_texts = [row[text_column] for row in train_data]
    X_val_texts = [row[text_column] for row in val_data]
    X_test_texts = [row[text_column] for row in test_data]

    y_train = np.array([row["sentiment_label"] for row in train_data])
    y_val = np.array([row["sentiment_label"] for row in val_data])
    y_test = np.array([row["sentiment_label"] for row in test_data])

    # Fit TF-IDF vectorizer on training data (unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)
    X_test = vectorizer.transform(X_test_texts)

    # Create output directory
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature matrices using scipy sparse format
    from scipy.sparse import save_npz

    save_npz(FEATURE_OUTPUT_DIR / "X_train.npz", X_train)
    save_npz(FEATURE_OUTPUT_DIR / "X_val.npz", X_val)
    save_npz(FEATURE_OUTPUT_DIR / "X_test.npz", X_test)

    # Save labels
    np.save(FEATURE_OUTPUT_DIR / "y_train.npy", y_train)
    np.save(FEATURE_OUTPUT_DIR / "y_val.npy", y_val)
    np.save(FEATURE_OUTPUT_DIR / "y_test.npy", y_test)

    # Save vectorizer metadata
    import json

    vectorizer_meta = {
        "text_column_used": text_column,
        "ngram_range": (1, 2),
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
        "n_features": X_train.shape[1],
        "feature_names_sample": vectorizer.get_feature_names_out()[:20].tolist(),
    }
    with (FEATURE_OUTPUT_DIR / "vectorizer_meta.json").open("w") as f:
        json.dump(vectorizer_meta, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"\nText column used: {text_column}")
    print(f"  (From: data/processed/sentiment_preprocessed.csv)")
    print(f"\nTF-IDF Vectorizer Configuration:")
    print(f"  - N-gram range: (1, 2) [unigrams + bigrams]")
    print(f"  - Max features: 5,000")
    print(f"  - Min document frequency: 2")
    print(f"  - Max document frequency: 0.95")
    print(f"  - Final feature count: {X_train.shape[1]}")
    print(f"\nSplit Sizes:")
    print(f"  - Train: {len(train_ids):,} samples -> shape {X_train.shape}")
    print(f"  - Val:   {len(val_ids):,} samples -> shape {X_val.shape}")
    print(f"  - Test:  {len(test_ids):,} samples -> shape {X_test.shape}")
    print(f"\nSentiment Label Distribution:")
    unique_labels, train_counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, train_counts):
        pct = 100.0 * count / len(y_train)
        print(f"  - {label}: {count:,} ({pct:.1f}%)")
    print(f"\nOutput Location: {FEATURE_OUTPUT_DIR}/")
    print(f"  - X_train.npz, X_val.npz, X_test.npz (sparse TF-IDF matrices)")
    print(f"  - y_train.npy, y_val.npy, y_test.npy (labels)")
    print(f"  - vectorizer_meta.json (feature metadata)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()