# NLP Final Project

This repository contains the full code pipeline for a sentiment analysis project with slang-aware experimentation:

- canonical data collection and cleaning
- preprocessing with slang feature signals and text variants
- transformer modeling (BERT-base, BERTweet, plus GPT baseline for comparison)
- multi-seed evaluation, summaries, and paired significance tests

## Final Project Scope (What Was Actually Used)

For the final reported modeling results:

- Primary models: BERT-base and BERTweet.
- GPT was trained as a baseline comparison only.
- GPT DAPT and GPT slang fine-tuning were not part of the final reported methodology.

## Code Documentation Index

- Main usage guide: `README.md`
- Detailed module and API-level documentation: `CODE_DOCUMENTATION.md`
- Data/process reports used in writeups:
  - `reports/data_collection_cleaning_report.md`
  - `reports/preprocessing_report.md`
  - `reports/dataset_card.md`

## Repository Layout

```text
data/
  raw/        # Original source files (never modified)
  interim/    # Ingestion and cleaning outputs
  processed/  # Modeling-ready datasets and manifests
  splits/     # train/val/test IDs and split manifest
src/
  data_collection/
  cleaning/
  preprocessing/
  modeling/
configs/
  data_pipeline.yaml
  modeling/
outputs/
  modeling/
analysis/
reports/
```

## Environment Setup

### 1) Activate your virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r src/modeling/requirements_modeling.txt
```

### 3) Optional Hugging Face token

If running GPT baseline comparisons that need Hub access:

1. Copy `.env.example` to `.env`.
2. Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`).

All modeling entrypoints load `.env` automatically.

## End-to-End Pipeline Commands

Run from repository root.

### Data Collection

```powershell
python -m src.data_collection.collect
```

Outputs:

- `data/interim/canonical_ingested.csv`
- `data/interim/canonical_ingestion_summary.json`

### Data Cleaning

```powershell
python -m src.cleaning.clean
```

Outputs:

- `data/interim/canonical_cleaned.csv`
- `data/interim/canonical_cleaning_summary.json`

### Preprocessing

```powershell
python -m src.preprocessing.preprocess
```

Outputs:

- `data/processed/sentiment_preprocessed.csv`
- `data/processed/sentiment_preprocessed_variants.csv`
- `data/processed/sarcasm_auxiliary_preprocessed.csv`
- `data/processed/dapt_train_corpus.csv` (generated but not required for final reported results)
- `data/processed/preprocessing_summary.json`
- `data/splits/split_manifest.json`

## Modeling Commands

### Single config run

```powershell
python -m src.modeling.run_classification --config configs/modeling/bert_base_baseline.yaml
```

### Multi-seed run

```powershell
python -m src.modeling.run_multiseed --config configs/modeling/bert_base_baseline.yaml --seeds 42 43 44
```

### Final Core Matrix (What We Report)

This is the recommended command for the final BERT/BERTweet-focused matrix:

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error --non-gpt-only
```

### Full 3x3 model-variant matrix (includes GPT baseline)

```powershell
python -m src.modeling.run_experiment_matrix --seeds 42 43 44
```

Dry-run only:

```powershell
python -m src.modeling.run_experiment_matrix --seeds 42 43 44 --dry-run
```

### Resumable full training pipeline

Core classification matrix only:

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error
```

Run only BERT/BERTweet (recommended for final report):

```powershell
python -m src.modeling.run_full_training_pipeline --seeds 42 43 44 --skip-completed --continue-on-error --non-gpt-only
```

### GPT Baseline (Comparison-Only)

```powershell
python -m src.modeling.run_classification --config configs/modeling/gpt_classification_baseline.yaml
```

### Archived/Optional GPT Adaptation Commands (Not Used In Final Results)

```powershell
python -m src.modeling.dapt --config configs/modeling/gpt_dapt.yaml
```

### Archived/Optional GPT slang-focused fine-tuning

```powershell
python -m src.modeling.run_gpt_finetune --config configs/modeling/gpt_finetune_slang.yaml
python -m src.modeling.run_gpt_finetune --config configs/modeling/gpt_finetune_slang_mixed.yaml
```

## Evaluation Utilities

### Aggregate all run summaries

```powershell
python -m src.modeling.summarize_runs --root outputs/modeling --output outputs/modeling/run_summary_table.csv
python -m src.modeling.summarize_multiseed --root outputs/modeling --output outputs/modeling/multiseed_summary_table.csv
```

### Paired significance testing

```powershell
python -m src.modeling.paired_significance --a outputs/modeling/bert_base/logs/seed_42/predictions_test.csv --b outputs/modeling/bert_base_slang_masked/logs/seed_42/predictions_test.csv --output outputs/modeling/paired_significance_bert_vs_slang_seed42.json
```

## Expected Modeling Artifacts

Each run in `outputs/modeling/...` can contain:

- checkpoints and final model weights
- `predictions_test.csv`
- `misclassified_test.csv`
- `test_metrics_detailed.json`
- `test_metrics_by_group.json`
- `run_summary.json`
- multi-seed aggregate files (`multiseed_summary.json`)

## Notebook and Analysis

- Main analysis notebook: `analysis/deep_results_analysis.ipynb`
- Analysis script: `analysis/generate_analysis.py`
- Generated analysis outputs: `analysis/outputs/`

## Submission-Facing Notes

- This repository already contains code for Data Collection, Cleaning, Preprocessing, Feature Extraction (slang/informal signals and text variants), Modeling, and Evaluation.
- Final reported modeling scope is BERT-base + BERTweet, with GPT baseline as comparison only.
- A deployment demo is now included through `app.py` and `src/deployment/`.
- For the Code Documentation deliverable, use:
  - `README.md` (setup and execution instructions)
  - `CODE_DOCUMENTATION.md` (detailed code/module documentation)

## Deployment Demo

This repository now includes a simple multi-model deployment demo built with Gradio.

What it does:

- loads deployable final models from `MODELS_FINAL/`
- exposes a single web UI for comparing model variants
- applies the same URL/mention cleaning used in training
- applies variant-specific slang masking before inference
- remaps raw classifier outputs to `negative`, `neutral`, and `positive`

### Run locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Launch the demo:

```powershell
python app.py
```

Then open the local Gradio URL shown in the terminal.

### Model discovery behavior

The deployment app auto-discovers model folders under `MODELS_FINAL/` and only serves variants that contain actual weight files.

Expected families:

- `bert_base`
- `bert_finetune_slang`
- `bert_finetune_slang_mixed`

If one family is missing weights, the app marks it as unavailable instead of failing at startup.

### Deploy to Hugging Face Spaces

Recommended for final-project sharing:

1. Create a new Gradio Space on Hugging Face.
2. Upload this repository.
3. Ensure `requirements.txt` is present at the repo root.
4. Keep `app.py` at the repo root as the Space entrypoint.
5. Include the desired `MODELS_FINAL/` folders in the Space repository or storage.

Once the Space builds, the demo UI will be publicly shareable by URL.
