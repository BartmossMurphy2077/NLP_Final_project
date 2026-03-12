# NLP Final Project

## Workflow Layout

```text
data/
  raw/        # Original source files (never modified)
  interim/    # Cleaned intermediate outputs
  processed/  # Final modeling-ready datasets + schema
  splits/     # Fixed train/val/test split manifests
src/
  data_collection/
  cleaning/
  preprocessing/
configs/
reports/
```

## Quick Start

1. Add source datasets to `data/raw/`.
2. Implement collection logic in `src/data_collection/collect.py`.
3. Run cleaning in `src/cleaning/clean.py` to produce `data/interim/`.
4. Run preprocessing in `src/preprocessing/preprocess.py` to produce `data/processed/` and `data/splits/`.
5. Document each step in the `reports/` templates.

## Data Contract (handoff)

Final modeling-ready data should include:

- `id`
- `text_original`
- `text_clean`
- `text_variant` (`original`, `slang_masked`, `mixed`)
- `sentiment_label`
- `slang_label` (`slang_heavy`, `formal`)
- `source`
- `split` (`train`, `val`, `test`)
