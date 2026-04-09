# Final Results Analysis (Outputs/Modeling)

## Scope and Data Sources

This analysis is based on artifacts under `outputs/modeling`:

- `outputs/modeling/run_summary_table.csv` (seed-level records)
- `outputs/modeling/multiseed_summary_table.csv` (aggregated multiseed metrics)
- `outputs/modeling/paired_significance_bert_vs_finetune_slang_seed42.json`
- `outputs/modeling/paired_significance_bertmixed_vs_finetune_mixed_seed42.json`

Interpretation here prioritizes:

- robust model ranking (mean + uncertainty)
- stability across seeds
- statistically tested impact of slang-heavy finetuning

## Coverage Snapshot

- BERT-base: baseline, mixed, slang_masked (3 seeds each)
- BERTweet: baseline, mixed, slang_masked (3 seeds each)
- BERT slang-heavy finetunes: original and mixed (3 seeds each)
- GPT baseline: single-seed entry in run-level table (comparison-only)

## Multi-Seed Leaderboard

Ranked by `test_macro_f1_mean` from `multiseed_summary_table.csv`.

| Rank | Experiment                        | Accuracy Mean | Accuracy Std | Macro-F1 Mean | Macro-F1 Std | Macro-F1 CV % |
| ---- | --------------------------------- | ------------: | -----------: | ------------: | -----------: | ------------: |
| 1    | `bert_base_baseline`              |        0.9322 |       0.0016 |        0.9248 |       0.0025 |          0.27 |
| 2    | `bert_base_slang_masked`          |        0.9284 |       0.0021 |        0.9212 |       0.0019 |          0.20 |
| 3    | `bert_base_mixed`                 |        0.9276 |       0.0031 |        0.9205 |       0.0027 |          0.30 |
| 4    | `bertweet_mixed`                  |        0.9204 |       0.0014 |        0.9113 |       0.0022 |          0.24 |
| 5    | `bertweet_baseline`               |        0.9164 |       0.0104 |        0.9092 |       0.0104 |          1.14 |
| 6    | `bert_finetune_slang_heavy`       |        0.8965 |       0.0014 |        0.8909 |       0.0009 |          0.11 |
| 7    | `bert_finetune_slang_heavy_mixed` |        0.8941 |       0.0005 |        0.8880 |       0.0005 |          0.06 |
| 8    | `bertweet_slang_masked`           |        0.5773 |       0.2958 |        0.4313 |       0.4133 |         95.83 |

Interpretation:

- `bert_base_baseline` is the strongest overall performer.
- The top three entries are all BERT-base variants with narrow uncertainty.
- `bertweet_slang_masked` is a clear instability outlier.

## Family and Variant Comparison (Core Experiments)

Core experiments exclude finetune rows to keep model-family comparisons fair.

| Model Family | Original |  Mixed | Slang Masked |
| ------------ | -------: | -----: | -----------: |
| BERT         |   0.9248 | 0.9205 |       0.9212 |
| BERTweet     |   0.9092 | 0.9113 |       0.4313 |

BERT minus BERTweet macro-F1 gap:

| Variant      | Delta (BERT - BERTweet) |
| ------------ | ----------------------: |
| Original     |                 +0.0156 |
| Mixed        |                 +0.0092 |
| Slang Masked |                 +0.4899 |

Interpretation:

- BERT is consistently stronger than BERTweet on original and mixed variants.
- Under aggressive slang masking, BERTweet collapses while BERT remains stable.

## Seed Stability and Reproducibility Risk

Stability indicators from multiseed uncertainty and seed-level diagnostics:

- Most stable among high performers: `bert_base_slang_masked` (Macro-F1 std 0.0019)
- Moderate sensitivity: `bertweet_baseline` (Macro-F1 std 0.0104)
- Severe instability: `bertweet_slang_masked` (Macro-F1 std 0.4133; CV 95.83%)

This means mean performance alone is not enough. Reporting should include uncertainty for every main comparison.

## Slang-Heavy Finetuning: Statistical Result

Matched-example paired tests show that slang-heavy finetuning underperforms baseline in both checked settings.

| Comparison                                             | Matched N | Accuracy Delta (Finetune - Baseline) | McNemar p | Sign Test p | Bootstrap 95% CI   |
| ------------------------------------------------------ | --------: | -----------------------------------: | --------: | ----------: | ------------------ |
| `bert_base_baseline` vs `bert_finetune_slang_heavy`    |      2178 |                              -0.0340 |  1.53e-09 |    6.55e-10 | [-0.0445, -0.0230] |
| `bert_base_mixed` vs `bert_finetune_slang_heavy_mixed` |      2178 |                              -0.0266 |  7.20e-08 |    3.58e-08 | [-0.0367, -0.0170] |

Interpretation:

- Both deltas are negative.
- Both significance tests are strongly below 0.05.
- Confidence intervals are fully below zero.

Conclusion: slang-heavy-only finetuning is a robust negative finding in current outputs.

## Practical Write-Up Guidance

Use this framing in the final report:

1. BERT-base is the best-performing and most reliable family in this experiment matrix.
2. BERTweet is competitive on mixed/original, but fails under slang-masked setup.
3. Slang-heavy finetuning is not beneficial in this configuration and significantly degrades paired performance.
4. GPT baseline can be shown as contextual reference only (single-seed evidence).

## Reproducibility

Regenerate summary artifacts:

```bash
python analysis/generate_analysis.py
```

Interactive deep-dive notebook:

- `analysis/deep_results_analysis.ipynb`
