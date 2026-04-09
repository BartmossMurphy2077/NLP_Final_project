# Final Results Analysis

## Scope and Source of Truth

This analysis is based on model logs and summary tables in `outputs/modeling/`:

- `outputs/modeling/run_summary_table.csv` (seed-level runs)
- `outputs/modeling/multiseed_summary_table.csv` (3-seed aggregates)
- paired significance outputs for baseline vs slang-heavy finetune

Context reflected here matches your final project direction:

- Main model development: BERT-base and BERTweet
- GPT used as baseline comparison only
- GPT DAPT/fine-tuning not part of final reported core results

## What Is Available in Outputs

- BERT-base: baseline, slang_masked, mixed (3 seeds each)
- BERTweet: baseline, slang_masked, mixed (3 seeds each)
- BERT slang-heavy finetuning: slang_masked and mixed (3 seeds each, slang-heavy subgroup)
- GPT baseline: original variant, seed 42

## Per-Experiment Performance (Multi-Seed)

Ranked by `test_macro_f1_mean` from `multiseed_summary_table.csv`:

1. `bert_base_baseline` -> Accuracy 0.9322, Macro-F1 0.9248 (+/- 0.0025)
2. `bert_base_slang_masked` -> Accuracy 0.9284, Macro-F1 0.9212 (+/- 0.0019)
3. `bert_base_mixed` -> Accuracy 0.9276, Macro-F1 0.9205 (+/- 0.0027)
4. `bertweet_mixed` -> Accuracy 0.9204, Macro-F1 0.9113 (+/- 0.0022)
5. `bertweet_baseline` -> Accuracy 0.9164, Macro-F1 0.9092 (+/- 0.0104)
6. `bert_finetune_slang_heavy` -> Accuracy 0.8965, Macro-F1 0.8909 (+/- 0.0009)
7. `bert_finetune_slang_heavy_mixed` -> Accuracy 0.8941, Macro-F1 0.8880 (+/- 0.0005)
8. `bertweet_slang_masked` -> Accuracy 0.5773, Macro-F1 0.4313 (+/- 0.4133)

## Difference Between Models

### BERT-base vs BERTweet (same variants)

Macro-F1 means:

- Original: BERT-base 0.9248 vs BERTweet 0.9092 (BERT-base +0.0156)
- Mixed: BERT-base 0.9205 vs BERTweet 0.9113 (BERT-base +0.0092)
- Slang-masked: BERT-base 0.9212 vs BERTweet 0.4313 (BERT-base +0.4899)

Interpretation:

- BERT-base is consistently stronger across variants.
- The largest gap is in slang-masked setting due to BERTweet instability.

### Within-model variant sensitivity

BERT-base:

- Best variant: original (Macro-F1 0.9248)
- Variant drop from original:
  - slang_masked: -0.0036
  - mixed: -0.0043
- Interpretation: mild degradation only; fairly robust.

BERTweet:

- Best variant: mixed (Macro-F1 0.9113)
- Original is close (0.9092)
- Slang_masked collapses (0.4313, very high std)
- Interpretation: sensitive/unstable under full slang masking.

### GPT baseline vs BERT leaders

GPT baseline (seed 42 only):

- Accuracy 0.9137
- Macro-F1 0.9064

Compared to best BERT multiseed (`bert_base_baseline` Macro-F1 0.9248), GPT baseline is lower by about 0.0184 Macro-F1. Because GPT has only one seed and no full matrix completion, this remains comparison-only evidence.

## Per-Model Performance (Seed-Level Aggregates)

Computed from seed-level runs in `run_summary_table.csv` (excluding non-seeded legacy row):

- `bert-base-uncased`: 15 runs, avg Accuracy 0.9158, avg Macro-F1 0.9091
- `vinai/bertweet-base`: 9 runs, avg Accuracy 0.8047, avg Macro-F1 0.7506
- `openai-community/gpt2`: 1 run, Accuracy 0.9137, Macro-F1 0.9064

Important caveat:

- BERTweet average is pulled down heavily by `bertweet_slang_masked` failures. For fair reading, compare per-variant numbers above.

## Slang-Heavy Finetuning Result (Negative Finding)

When compared on matched slang-heavy test examples, slang-heavy-only finetuning underperforms baseline:

- baseline original vs `bert_finetune_slang_heavy`:
  - accuracy diff (finetune - baseline): -0.0340
  - McNemar p: 1.53e-09
  - Sign test p: 6.55e-10
  - bootstrap 95% CI: [-0.0445, -0.0230]

- baseline mixed vs `bert_finetune_slang_heavy_mixed`:
  - accuracy diff (finetune - baseline): -0.0266
  - McNemar p: 7.20e-08
  - Sign test p: 3.58e-08
  - bootstrap 95% CI: [-0.0367, -0.0170]

Interpretation:

- Performance drop is statistically significant in both comparisons.
- This is a valid and useful negative result: narrowing training to slang-heavy only reduced general sentiment classification quality on that subgroup.

## High-Level Takeaways

1. Best overall model in current outputs: `bert_base_baseline`.
2. BERT-base consistently outperforms BERTweet across comparable variants.
3. BERTweet with full slang masking is unstable and should be treated as a limitation/risk finding.
4. GPT baseline is competitive but not superior to top BERT result, and evidence is limited to one seed.
5. Slang-heavy-only finetuning did not improve results and significantly degraded performance.

## Recommended Wording for Final Write-Up

- Present BERT-base and BERTweet as the core model families.
- Report GPT as supplementary baseline comparison.
- Explicitly include the negative finetuning result and significance tests as part of the contribution.
- Discuss `bertweet_slang_masked` instability as an observed failure mode under aggressive masking.

## Reproducibility

To regenerate analysis artifacts:

```bash
python analysis/generate_analysis.py
```

Interactive deep dive:

- `analysis/deep_results_analysis.ipynb`
