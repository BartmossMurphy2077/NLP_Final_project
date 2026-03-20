# Data Collection and Cleaning Report

## 1. Objective
- Goal: Inventory and validate all raw datasets required for the project before cleaning/preprocessing.
- Research question alignment: Build a sentiment-focused corpus and an auxiliary informal-language corpus to study how modern slang/informal style affects model performance.

## 2. Data Sources
- Source 1: Kaggle `datatattle/covid-19-nlp-text-classification`
- Local path: `data/raw/covid19_nlp_text_classification/coronavirus/`
- Files:
  - `Corona_NLP_train.csv` (41,157 rows)
  - `Corona_NLP_test.csv` (3,798 rows)
- Schema:
  - `UserName` (string/id-like)
  - `ScreenName` (string/id-like)
  - `Location` (string, nullable)
  - `TweetAt` (date string `%d-%m-%Y`)
  - `OriginalTweet` (string text)
  - `Sentiment` (5-class label: `Extremely Negative`, `Negative`, `Neutral`, `Positive`, `Extremely Positive`)
- Sentiment records total (train + test): 44,955
- Tweet date range in raw data: 2020-03-02 to 2020-04-14

- Source 2: Kaggle `rmisra/news-headlines-dataset-for-sarcasm-detection`
- Local path: `data/raw/news_headlines_sarcasm/newsheadlines/`
- Files:
  - `Sarcasm_Headlines_Dataset.json` (26,709 JSONL records)
  - `Sarcasm_Headlines_Dataset_v2.json` (28,619 JSONL records)
- Schema:
  - `article_link` (string URL)
  - `headline` (string text)
  - `is_sarcastic` (binary integer: `0` or `1`)
- Data selection note:
  - `v1` is a subset of `v2`; for modeling we will use `v2` as the canonical sarcasm file to avoid double counting.

- Access method (both sources): Manual download from Kaggle dataset pages, then upload into `data/raw/`.
- Date range collected (local): March 2026.
- Inclusion criteria:
  - English text-based entries.
  - Records with non-empty text field (`OriginalTweet` or `headline`).
  - For sarcasm data, keep binary labels in `{0,1}` only.
- Exclusion criteria:
  - None applied yet at raw stage except selecting canonical sarcasm file (`v2`).
- Licensing/ethics notes:
  - No separate LICENSE files were included inside downloaded folders.
  - Usage is subject to each Kaggle dataset's license and Kaggle Terms of Use.
  - Action item before publication/redistribution: confirm and document the exact license shown on each Kaggle dataset page and include citation requirements in final report/repo.
  - Ethics note: raw tweet/headline text may contain profanity, sensitive language, and potential demographic/geographic bias.

## 3. Collection Procedure
- Query terms/filters: Not applicable (datasets were pre-curated by providers on Kaggle).
- Sampling strategy:
  - Source-defined dataset splits for Corona NLP (`train`/`test`) retained as downloaded.
  - Sarcasm dataset variants audited; `v2` chosen as canonical file for downstream tasks.
- Raw records collected:
  - Corona NLP (train + test): 44,955
  - Sarcasm headlines canonical (`v2`): 28,619
  - Combined canonical raw total: 73,574

## 4. Cleaning Pipeline
- Removed: URLs, @mentions
- Preserved: emojis, hashtags, profanity, expressive spelling
- Deduplication approach:
  - Inventory check completed.
  - Full deduplication will be applied in cleaning stage (exact-match on cleaned text + source-aware record id).
- Language filtering approach:
  - Datasets are English-oriented by design.
  - Additional language filtering is planned only if non-English leakage is detected during cleaning.

## 5. Quality Checks
- Null/empty text handling:
  - `Corona_NLP_train/test`: `OriginalTweet` empty count = 0
  - `Sarcasm_Headlines_Dataset/v2`: `headline` empty count = 0
- Duplicate rate (raw-stage audit only):
  - `Corona_NLP_train.csv`: 9 duplicate tweet texts out of 41,157
  - `Corona_NLP_test.csv`: 0 duplicate tweet texts out of 3,798
  - `Sarcasm_Headlines_Dataset.json`: 1 duplicate full record out of 26,709
  - `Sarcasm_Headlines_Dataset_v2.json`: 2 duplicate full records out of 28,619
- Label integrity checks:
  - Corona sentiment labels observed: exactly 5 expected classes.
  - Sarcasm labels observed: only `0` and `1`.
- Manual spot checks:
  - Spot-checked sample rows from each source for schema readability and text content presence.
  - Found URLs/mentions/social-media artifacts as expected for planned cleaning rules.

## 6. Dataset Size Summary
| Stage | Rows |
|---|---:|
| Raw (canonical source choice: Corona train+test + Sarcasm v2) | 73,574 |
| After cleaning | Pending |
| Final usable | Pending |

## 7. Risks and Limitations
- Bias considerations:
  - Corona tweets are time-bound (early pandemic period) and topic-specific.
  - Sarcasm headlines are news-domain specific and may not represent social-media slang directly.
  - Label subjectivity exists in both sentiment intensity and sarcasm annotations.
- Remaining noise:
  - Raw text contains URLs, mentions, and likely duplicated text fragments.
  - Informal language signals may correlate with platform/source rather than sentiment alone.
- Known failure modes:
  - Potential domain mismatch between tweets (sentiment) and headlines (sarcasm/informality).
  - If both sarcasm `v1` and `v2` are merged naively, duplicate inflation occurs.
  - License ambiguity until Kaggle-page license fields are explicitly recorded.
