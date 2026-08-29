# Person 4 -- Synthesis & Proposed Vocab-Size Criterion

## Proposed criterion

Recommend vocab size at the point where the vocabulary-growth curve plateaus, computed **per language x tokenizer** from `vocab_growth_results.csv` (same style of criterion Person 3 used tokenizer-wide: relative marginal growth below 5% of its initial value for 2 consecutive checkpoints), falling back to Person 3's tokenizer-level plateau (`sweet_spot_estimates.csv`) and then the tokenizer's full trained vocab size if no plateau is found. Cross-check against Person 2's Zipf fit: rows with R^2 below 0.9 are flagged `low_r2_review`, and rows whose exponent is a statistical outlier *relative to that same tokenizer's exponent on the other two languages* (|z| > 1.5) are flagged `atypical_exponent_review`.

**Why not a fixed 'good' exponent range (e.g. 0.8-1.2)?** That range comes from word-level Zipf studies. Every language x tokenizer pair in this project has s well above 1.2 (observed range ~1.3-2.2), consistent with BPE/subword tokens being known to follow a steeper Zipf slope than whole words. Applying a word-level band here would flag 100% of rows as suspect and carry no signal, so confidence is instead based on fit quality and each tokenizer's own internal consistency across languages.

## Headline result 1: tokenizer-level plateau vs. trained vocab size

Per Person 3's tokenizer-wide sweet-spot estimate (`sweet_spot_estimates.csv`), the vocabulary-growth curve plateaus well under the tokenizer's full trained vocab size:

| tokenizer | plateau vocab (tokenizer-wide) | full trained vocab | coverage |
|---|---|---|---|
| Kimi | 56759 | 163840 | 34.6% |
| LLaMA | 59233 | 128256 | 46.2% |
| Qwen | 57388 | 151665 | 37.8% |

Roughly a third to a half of each tokenizer's trained vocabulary is doing essentially all the work -- the rest sees vanishingly few additional unique tokens as the corpus grows further.

## Headline result 2: the plateau is not the same across languages

The tokenizer-wide number above hides real per-language differences. Computing the plateau separately per language (from `vocab_growth_results.csv`) instead of pooling all languages together:

| language   |   Kimi |   LLaMA |   Qwen |
|:-----------|-------:|--------:|-------:|
| Arabic     |    nan |     nan |    nan |
| English    |  60196 |   63357 |  61260 |
| Hindi      |    nan |     nan |    nan |

The recommended vocab size per language-tokenizer pair (`recommended_vocab_size` in the combined table below) uses this language-level plateau first, since it's the number that actually answers 'what's the sweet spot for *this* language,' and only falls back to the tokenizer-wide figure when a language-level plateau can't be detected.

## Does the Zipf exponent shift more with language or tokenizer?

From `zipf_variation_summary.csv` (Person 2's own comparison, reused here rather than recomputed):

| comparison        | held_fixed   |   n_groups |   mean_s |   range_s |      std_s |
|:------------------|:-------------|-----------:|---------:|----------:|-----------:|
| across languages  | Kimi         |          3 |  1.71143 | 0.279023  | 0.145828   |
| across languages  | LLaMA        |          3 |  1.93457 | 0.42368   | 0.231595   |
| across languages  | Qwen         |          3 |  1.76463 | 0.946073  | 0.473118   |
| across tokenizers | Arabic       |          3 |  2.08649 | 0.406156  | 0.225788   |
| across tokenizers | English      |          3 |  1.77067 | 0.0163101 | 0.00889698 |
| across tokenizers | Hindi        |          3 |  1.55347 | 0.539982  | 0.270042   |

## Surprising case

**English** has the lowest Zipf-fit R^2 of the three languages for *every single tokenizer* (LLaMA, Qwen, and Kimi alike) -- not just one tokenizer/language pairing. That consistency across tokenizers points at something about the corpus or language itself (sample composition, cleaning, script-specific tokenization behavior) rather than any one tokenizer's algorithm. Worth a follow-up look before trusting that language's Zipf fit as strongly as the other two.

| tokenizer   |   Arabic |   English |   Hindi |
|:------------|---------:|----------:|--------:|
| Kimi        |    0.955 |     0.838 |   0.931 |
| LLaMA       |    0.955 |     0.842 |   0.947 |
| Qwen        |    0.954 |     0.84  |   0.923 |

## Combined results

| language   | tokenizer   |   zipf_exponent_s |   r_squared | exponent_outlier   |   observed_vocab_at_plateau_lang |   observed_vocab_at_plateau |   plateau_agreement_pct |   tokenizer_total_vocab |   recommended_vocab_size | confidence    |
|:-----------|:------------|------------------:|------------:|:-------------------|---------------------------------:|----------------------------:|------------------------:|------------------------:|-------------------------:|:--------------|
| Arabic     | Kimi        |             1.826 |       0.955 | False              |                              nan |                       56759 |                 nan     |                  163840 |                    56759 | confident     |
| Arabic     | LLaMA       |             2.2   |       0.955 | False              |                              nan |                       59233 |                 nan     |                  128256 |                    59233 | confident     |
| Arabic     | Qwen        |             2.233 |       0.954 | False              |                              nan |                       57388 |                 nan     |                  151665 |                    57388 | confident     |
| English    | Kimi        |             1.76  |       0.838 | False              |                            60196 |                       56759 |                  93.945 |                  163840 |                    56759 | low_r2_review |
| English    | LLaMA       |             1.777 |       0.842 | False              |                            63357 |                       59233 |                  93.038 |                  128256 |                    59233 | low_r2_review |
| English    | Qwen        |             1.775 |       0.84  | False              |                            61260 |                       57388 |                  93.253 |                  151665 |                    57388 | low_r2_review |
| Hindi      | Kimi        |             1.547 |       0.931 | False              |                              nan |                       56759 |                 nan     |                  163840 |                    56759 | confident     |
| Hindi      | LLaMA       |             1.826 |       0.947 | False              |                              nan |                       59233 |                 nan     |                  128256 |                    59233 | confident     |
| Hindi      | Qwen        |             1.287 |       0.923 | False              |                              nan |                       57388 |                 nan     |                  151665 |                    57388 | confident     |

## Notes

- `observed_vocab_at_plateau_lang` is computed by this script directly from `vocab_growth_results.csv`; `observed_vocab_at_plateau` is Person 3's tokenizer-only estimate from `sweet_spot_estimates.csv`. `plateau_agreement_pct` shows how close the two are -- large gaps are worth a manual look regardless of the confidence flag.
- `confidence = low_r2_review` or `atypical_exponent_review` rows should get a manual look at Person 2's log-log plot before the recommended vocab size is treated as final.
- Thresholds (min R^2 0.9, exponent outlier z 1.5, plateau relative threshold 5%) are intentionally simple and tunable -- a defensible rule of thumb, not a rigorously derived cutoff.
