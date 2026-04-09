# Code Documentation

## Purpose

This document provides technical documentation for the project codebase, including module responsibilities, execution flow, configuration strategy, and usage instructions.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- YAML-based experiment configuration

Dependencies are listed in `src/modeling/requirements_modeling.txt`.

## Runtime Assumptions

- Commands are executed from repository root.
- Virtual environment is activated.
- Raw datasets are placed under `data/raw/`.
- Optional Hugging Face authentication is set through `.env`.

## Final Methodology Alignment

- Final reported models: BERT-base and BERTweet.
- GPT was retained as a baseline comparison run.
- GPT DAPT and GPT slang fine-tuning remain in the repository as optional/archived experiments and were not used in the final reported results.

## Pipeline Overview

1. Data collection consolidates raw sources into a canonical CSV schema.
2. Cleaning removes noisy tokens (URLs/mentions), normalizes whitespace, deduplicates, and preserves social-language signals.
3. Preprocessing creates sentiment datasets, slang-aware labels, text variants, and stratified splits.
4. Modeling trains baseline and slang-aware transformer models.
5. Evaluation utilities summarize metrics and run paired significance tests.

## Module-Level Documentation

### `src/data_collection/collect.py`

Responsibilities:

- Ingests COVID sentiment train/test CSVs and sarcasm JSONL source.
- Converts records into a canonical schema.
- Writes ingestion summary with per-source counts.

Inputs:

- `data/raw/covid19_nlp_text_classification/coronavirus/Corona_NLP_train.csv`
- `data/raw/covid19_nlp_text_classification/coronavirus/Corona_NLP_test.csv`
- `data/raw/news_headlines_sarcasm/newsheadlines/Sarcasm_Headlines_Dataset_v2.json`

Outputs:

- `data/interim/canonical_ingested.csv`
- `data/interim/canonical_ingestion_summary.json`

Run:

```powershell
python -m src.data_collection.collect
```

### `src/cleaning/clean.py`

Responsibilities:

- Applies deterministic cleaning rules to canonical ingested rows.
- Removes URLs and mentions.
- Preserves emojis, hashtags, profanity, and expressive spelling.
- Performs exact duplicate removal using a stable key.
- Adds source-based language metadata.

Input:

- `data/interim/canonical_ingested.csv`

Outputs:

- `data/interim/canonical_cleaned.csv`
- `data/interim/canonical_cleaning_summary.json`

Run:

```powershell
python -m src.cleaning.clean
```

### `src/preprocessing/preprocess.py`

Responsibilities:

- Converts cleaned data into sentiment-focused modeling datasets.
- Computes informal/slang signals and assigns `slang_label`.
- Builds text variants: `original`, `slang_masked`, `mixed`.
- Performs deterministic stratified split assignment for sentiment rows.
- Creates train-only DAPT corpus (optional artifact, not required for final reported results).

Input:

- `data/interim/canonical_cleaned.csv`

Outputs:

- `data/processed/sentiment_preprocessed.csv`
- `data/processed/sentiment_preprocessed_variants.csv`
- `data/processed/sarcasm_auxiliary_preprocessed.csv`
- `data/processed/dapt_train_corpus.csv`
- `data/processed/preprocessing_summary.json`
- `data/splits/train_ids.txt`
- `data/splits/val_ids.txt`
- `data/splits/test_ids.txt`
- `data/splits/split_manifest.json`

Run:

```powershell
python -m src.preprocessing.preprocess
```

### `src/modeling/config.py`

Responsibilities:

- Defines and loads experiment configuration schema from YAML.
- Centralizes model, data, training, logging, and optional DAPT settings.

### `src/modeling/dataset.py`

Responsibilities:

- Loads processed datasets for train/val/test.
- Applies filtering by split, variant, and slang label constraints.
- Provides model-ready tensors and sample metadata for reporting.

### `src/modeling/train_classification.py`

Responsibilities:

- Orchestrates transformer training and validation.
- Tracks metrics and writes run artifacts.
- Evaluates on test set and persists prediction-level outputs.

### `src/modeling/run_classification.py`

Responsibilities:

- CLI wrapper for single-config training runs.

Run:

```powershell
python -m src.modeling.run_classification --config configs/modeling/bert_base_baseline.yaml
```

### `src/modeling/run_multiseed.py`

Responsibilities:

- Executes one config across multiple seeds.
- Aggregates mean/std metrics into a single summary JSON.

Run:

```powershell
python -m src.modeling.run_multiseed --config configs/modeling/bert_base_baseline.yaml --seeds 42 43 44
```

### `src/modeling/run_experiment_matrix.py`

Responsibilities:

- Runs the default 3-model x 3-variant config matrix.
- Writes execution manifest.

Note:

- In final reporting, this matrix may be filtered to non-GPT runs using `run_full_training_pipeline.py --non-gpt-only`.

Run:

```powershell
python -m src.modeling.run_experiment_matrix --seeds 42 43 44
```

Dry run:

```powershell
python -m src.modeling.run_experiment_matrix --seeds 42 43 44 --dry-run
```

### `src/modeling/run_full_training_pipeline.py`

Responsibilities:

- Runs resumable sequential training stages.
- Supports skipping completed seeds.
- Optionally includes GPT DAPT and GPT finetuning stages.
- Refreshes summary tables after successful execution.

Final-results command:

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error --non-gpt-only
```

Run:

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error
```

Include GPT stages:

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error --include-dapt --include-gpt-finetune
```

Note:

- The command above is optional/archived and not required for the final reported methodology.

### `src/modeling/dapt.py`

Responsibilities:

- Runs GPT domain-adaptive pretraining on train-only corpus.
- Saves adapted model and tokenizer artifacts.

Usage status:

- Optional/archived; not part of the final reported results.

Run:

```powershell
python -m src.modeling.dapt --config configs/modeling/gpt_dapt.yaml
```

### `src/modeling/run_gpt_finetune.py`

Responsibilities:

- Executes GPT classifier fine-tuning using adapted base model and finetune config.

Usage status:

- Optional/archived; not part of the final reported results.

Run:

```powershell
python -m src.modeling.run_gpt_finetune --config configs/modeling/gpt_finetune_slang.yaml
```

### `src/modeling/summarize_runs.py`

Responsibilities:

- Recursively collects `run_summary.json` files and generates a flat CSV table.

Run:

```powershell
python -m src.modeling.summarize_runs --root outputs/modeling --output outputs/modeling/run_summary_table.csv
```

### `src/modeling/summarize_multiseed.py`

Responsibilities:

- Recursively collects `multiseed_summary.json` files and generates aggregate CSV.

Run:

```powershell
python -m src.modeling.summarize_multiseed --root outputs/modeling --output outputs/modeling/multiseed_summary_table.csv
```

### `src/modeling/paired_significance.py`

Responsibilities:

- Matches predictions across two runs (`base_id` preferred, fallback `id`).
- Computes paired accuracy comparison.
- Runs McNemar and sign-test p-values.
- Computes bootstrap confidence interval for accuracy difference.

Run:

```powershell
python -m src.modeling.paired_significance --a outputs/modeling/bert_base/logs/seed_42/predictions_test.csv --b outputs/modeling/bert_base_slang_masked/logs/seed_42/predictions_test.csv --output outputs/modeling/paired_significance_bert_vs_slang_seed42.json
```

## Configuration Strategy

Model experiments are configured in `configs/modeling/*.yaml`.

Common config concerns:

- model checkpoint and tokenizer selection
- data file and split filtering
- allowed text variants (`original`, `slang_masked`, `mixed`)
- optional slang-only filtering (`allowed_slang_labels`)
- training hyperparameters and checkpoint retention (`save_total_limit`)
- output/log directories

## Artifact Reference

Common output artifacts:

- `predictions_test.csv`
- `misclassified_test.csv`
- `test_metrics_detailed.json`
- `test_metrics_by_group.json`
- `run_summary.json`
- `multiseed/multiseed_summary.json`
- run manifests under `outputs/modeling/matrix_runs/` and `outputs/modeling/pipeline_runs/`

## Reproducibility Notes

- Data splits are deterministic and seed-controlled.
- Multi-seed scripts report aggregate mean/std values.
- Environment metadata (Python/Torch/Transformers/device) is tracked in aggregate outputs.

## Error Handling and Common Issues

- Missing raw files: ensure source datasets are placed in expected `data/raw/...` locations before running collection.
- Missing processed files: run collection -> cleaning -> preprocessing in order.
- Authentication errors for gated model access: set `HF_TOKEN` in `.env`.
- Large output directories: keep checkpoint retention small (for example, `save_total_limit: 2` in config).

## Suggested Submission Usage

For the Code Documentation deliverable, submit:

- `README.md` for setup and execution instructions.
- `CODE_DOCUMENTATION.md` for technical implementation details.
- Optional: convert these Markdown files to PDF if your instructor prefers PDF attachments.

Final scope note for submission:

- State that final model development focused on BERT-base and BERTweet, while GPT was used as baseline comparison only.
