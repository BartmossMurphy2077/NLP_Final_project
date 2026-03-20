# Preprocessing Report

## 1. Objective
- Goal: Convert cleaned canonical data into modeling-ready sentiment datasets and controlled slang-masking variants for comparative experiments.
- Inputs from cleaning stage:
  - `data/interim/canonical_cleaned.csv`
  - Rows by task type in input: `44,893` sentiment, `28,503` sarcasm.
- Outputs for modeling stage:
  - Sentiment base dataset with split and slang labels.
  - Sentiment controlled variants (`original`, `slang_masked`, `mixed`).
  - Auxiliary sarcasm preprocessed dataset.
  - Fixed split ID files and split manifest.

## 2. Preprocessing Steps
- Text normalization rules:
  - Reused `text_clean` from cleaning stage as primary modeling text.
  - No additional aggressive normalization (to retain informal/social signals).
- Token-level handling decisions:
  - Tokenization: whitespace split (`text_clean.split()`).
  - Tokens are analyzed in lowercase-normalized form for slang detection.
  - Punctuation is retained in text outputs except where masking replaces candidate tokens.
- Slang marker handling:
  - Rule-based detection combines:
    - Slang lexicon matches (e.g., `lol`, `lmao`, `fr`, `ngl`, `sus`, `idk`, `wtf`, `yall`, etc.).
    - Emoji presence.
    - Hashtag presence.
    - Elongated spelling patterns (repeated letters, e.g., `soooo`).
    - Repeated punctuation markers (`!!`, `???`, `?!`).
  - Candidate slang markers are used for controlled masking variants.
- Emoji/hashtag/profanity handling:
  - Preserved as meaningful semantic signals.
  - Emojis and hashtags are never stripped in preprocessing.
  - Profanity is preserved (no censoring/filtering step).

## 3. Slang vs Formal Labeling
- Operational definition of slang-heavy:
  - A sample is labeled `slang_heavy` if it contains informal markers likely to encode social-media style or sentiment cues.
- Rule-based criteria:
  - `slang_heavy` if any condition is true:
    - `slang_term_count >= 1`, or
    - `emoji_count >= 1`, or
    - `elongated_spelling_count >= 1`, or
    - `repeated_punctuation_count >= 1`, or
    - `hashtag_count >= 2`.
  - Otherwise label as `formal`.
- Manual validation protocol:
  - Planned for next iteration: annotate a random sample from both `formal` and `slang_heavy` and estimate precision/recall of heuristic labels.
  - Current run uses heuristic-only labeling (no manual adjudication yet).
- Inter-annotator agreement (if used):
  - Not applicable in current run (no multi-annotator pass yet).

## 4. Controlled Variants for Experiments
- `original`:
  - `text_for_model = text_clean` (no masking).
- `slang_masked`:
  - All detected slang/informal candidate tokens replaced with `[SLANG]`.
- `mixed`:
  - Deterministic partial masking: every other candidate token (left-to-right) replaced with `[SLANG]`.
- Variant volume:
  - Sentiment base rows: `44,893`
  - Variant rows: `134,679` (`44,893 x 3`)

## 5. Split Strategy
- Train/val/test ratios:
  - Train: `0.8`
  - Val: `0.1`
  - Test: `0.1`
- Stratification criteria:
  - Joint stratification over `sentiment_label` and `slang_label`.
- Random seed:
  - `42`
- Deterministic assignment method:
  - Stable hash (`sha256(seed:id)`) per sample, then stratified partitioning by group.
- Final split counts (sentiment only):
  - Train: `35,910`
  - Val: `4,485`
  - Test: `4,498`
  - Total: `44,893`
- Leakage prevention checks:
  - Split files are disjoint by ID.
  - Verified overlap counts: train-val `0`, train-test `0`, val-test `0`.

## 6. Output Schema and Artifacts
- Processed file locations:
  - `data/processed/sentiment_preprocessed.csv`
  - `data/processed/sentiment_preprocessed_variants.csv`
  - `data/processed/sarcasm_auxiliary_preprocessed.csv`
  - `data/processed/preprocessing_summary.json`
- Required columns (data contract):
  - Sentiment base:
    - `id`, `text_original`, `text_clean`, `source`, `sentiment_label`, `slang_label`, `split`, `task_type`, `slang_term_count`, `informal_signal_count`
  - Sentiment variants:
    - `id`, `base_id`, `text_original`, `text_clean`, `source`, `sentiment_label`, `slang_label`, `split`, `task_type`, `text_variant`, `text_for_model`
  - Sarcasm auxiliary:
    - `id`, `text_original`, `text_clean`, `source`, `task_label`, `slang_label`, `split`, `task_type`, `slang_term_count`, `informal_signal_count`
- Split manifest:
  - `data/splits/split_manifest.json`
  - ID files:
    - `data/splits/train_ids.txt`
    - `data/splits/val_ids.txt`
    - `data/splits/test_ids.txt`

## 7. Validation and Sanity Checks
- Class balance by split (sentiment labels):
  - Train: Neutral `6,650`, Positive `9,880`, Extremely Negative `4,856`, Negative `8,753`, Extremely Positive `5,771`
  - Val: Neutral `830`, Positive `1,234`, Extremely Negative `606`, Negative `1,094`, Extremely Positive `721`
  - Test: Neutral `834`, Positive `1,237`, Extremely Negative `609`, Negative `1,095`, Extremely Positive `723`
- Slang/formal balance by split (sentiment rows):
  - Train: formal `18,509`, slang_heavy `17,401`
  - Val: formal `2,312`, slang_heavy `2,173`
  - Test: formal `2,318`, slang_heavy `2,180`
- Sample inspection results:
  - `slang_masked` replaces hashtags/informal markers with `[SLANG]` as designed.
  - `mixed` retains partial informal signal while masking a deterministic subset.
  - Example behavior observed on hashtag-heavy COVID tweets (`#COVID19`, `#StayAtHome`, etc.).

## 8. Handoff Notes for Downstream Teams
- Feature extraction assumptions:
  - Use `text_for_model` from the variants file for ablation experiments.
  - Use `text_clean` from sentiment base for standard baselines.
  - Use `slang_term_count` and `informal_signal_count` as optional engineered features.
- Modeling constraints:
  - Main benchmark task is sentiment (`task_type=sentiment`) only.
  - Sarcasm dataset is auxiliary (`task_type=sarcasm`, `split=auxiliary`), not part of primary sentiment metrics.
  - Keep split IDs fixed for all model families (BERT, BERTweet, GPT).
- Evaluation compatibility notes:
  - Metrics should be computed on identical split IDs using the same label set.
  - Controlled comparison should contrast `original` vs `slang_masked` vs `mixed`.
  - Statistical significance tests should use paired predictions over matched test IDs.
