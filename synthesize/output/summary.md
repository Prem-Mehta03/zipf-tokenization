# Person 4 -- Synthesis & Proposed Vocab-Size Criterion

*Written directly from `combined_table.csv`, `recommended_vocab_sizes.csv`,
and `zipf_variation_summary.csv` -- no invented or fixture data.*

## Proposed criterion

Recommend vocab size at the point where the vocabulary-growth curve
plateaus for a given language x tokenizer (relative marginal growth below
5% of its initial value, for 2 consecutive checkpoints -- the same
criterion Person 3 used tokenizer-wide), falling back to Person 3's
tokenizer-level plateau when no language-level plateau is detected.
Cross-check against Person 2's Zipf fit: flag `low_r2_review` if R^2 <
0.90, and flag `atypical_exponent_review` if a language's exponent is a
statistical outlier (|z| > 1.5) relative to that *same tokenizer's*
exponent on the other two languages.

We deliberately did not use a fixed "good" exponent range (e.g. 0.8-1.2,
the classic word-level Zipf band) -- every language x tokenizer pair here
has s in the range 1.29-2.23, consistent with BPE/subword tokens having
steeper Zipf slopes than whole words. Confidence is instead grounded in
fit quality and each tokenizer's own cross-language consistency.

## Result 1 -- tokenizer-wide plateau vs. full trained vocab

| tokenizer | plateau vocab | full trained vocab | coverage |
|---|---|---|---|
| LLaMA | 59,233 | 128,256 | 46.2% |
| Qwen | 57,388 | 151,665 | 37.8% |
| Kimi | 56,759 | 163,840 | 34.6% |

Across all three tokenizers, roughly a third to a half of the full trained
vocabulary accounts for essentially all coverage this corpus exercises.

## Result 2 -- the per-language criterion doesn't currently change the recommendation

This is a limitation worth stating plainly rather than glossing over.
Applying the same plateau criterion *per language* (from
`vocab_growth_results.csv`) instead of pooled/tokenizer-wide:

- **Hindi and Arabic** have no rows in `vocab_growth_results.csv` at all
  (consistent with the assignment only requiring that plot for one
  language), so there's nothing to compute a language-level plateau from.
- **English** does have growth data, but its own curve never satisfies the
  plateau criterion on its own (`lang_plateau_detected = False` for all
  three tokenizers) -- English's vocabulary is still measurably growing at
  the last sampled checkpoint, even though the same criterion applied to
  the pooled/tokenizer-wide curve does detect a plateau.

As a result, `recommended_vocab_size` in `recommended_vocab_sizes.csv` is
currently identical to Person 3's original tokenizer-wide plateau for all
nine language x tokenizer rows -- the per-language logic hasn't yet
diverged from the tokenizer-only numbers with the data available. Closing
this would need either denser corpus-fraction checkpoints for English (to
give the criterion more chances to see two consecutive low-growth points)
or vocab-growth data for Hindi and Arabic as well.

## Does the Zipf exponent shift more with language or tokenizer?

From `zipf_variation_summary.csv`:

| comparison | held fixed | mean s | range s | std s |
|---|---|---|---|---|
| across languages | Kimi | 1.711 | 0.279 | 0.146 |
| across languages | LLaMA | 1.935 | 0.424 | 0.232 |
| across languages | Qwen | 1.765 | 0.946 | 0.473 |
| across tokenizers | Arabic | 2.086 | 0.406 | 0.226 |
| across tokenizers | English | 1.771 | 0.016 | 0.009 |
| across tokenizers | Hindi | 1.553 | 0.540 | 0.270 |

Averaging each side: holding tokenizer fixed and varying language gives an
average range of **0.55** (avg std 0.28); holding language fixed and
varying tokenizer gives an average range of **0.32** (avg std 0.17). **The
exponent shifts more with language than with tokenizer choice.**

Two things stand out within that:
- **Qwen's exponent swings the most across languages** (range 0.946: 1.29
  for Hindi up to 2.23 for Arabic) -- much more than LLaMA (0.42) or Kimi
  (0.28). Qwen's tokenization behavior is the least language-consistent of
  the three.
- **English is almost untouched by tokenizer choice** (range 0.016 across
  all three tokenizers -- s sits at 1.76-1.78 regardless of which
  tokenizer is used), while Hindi (range 0.54) and Arabic (range 0.41) vary
  substantially by tokenizer. Tokenizer choice appears to matter far more
  for Hindi and Arabic than for English.

## Surprising case

English has the lowest Zipf-fit R^2 of the three languages for *every*
tokenizer -- not one tokenizer/language pairing, all three:

| tokenizer | Arabic | English | Hindi |
|---|---|---|---|
| Kimi | 0.955 | 0.838 | 0.931 |
| LLaMA | 0.955 | 0.842 | 0.947 |
| Qwen | 0.954 | 0.840 | 0.923 |

That consistency across tokenizers points at something about the English
corpus/sample or its cleaning rather than any one tokenizer's algorithm --
worth a follow-up look before trusting English's Zipf fit as strongly as
Hindi's or Arabic's.

## Combined results

| language | tokenizer | s | R^2 | plateau vocab | confidence |
|---|---|---|---|---|---|
| Arabic | Kimi | 1.826 | 0.955 | 56,759 | confident |
| Arabic | LLaMA | 2.200 | 0.955 | 59,233 | confident |
| Arabic | Qwen | 2.233 | 0.954 | 57,388 | confident |
| English | Kimi | 1.760 | 0.838 | 56,759 | low_r2_review |
| English | LLaMA | 1.777 | 0.842 | 59,233 | low_r2_review |
| English | Qwen | 1.775 | 0.840 | 57,388 | low_r2_review |
| Hindi | Kimi | 1.547 | 0.931 | 56,759 | confident |
| Hindi | LLaMA | 1.826 | 0.947 | 59,233 | confident |
| Hindi | Qwen | 1.287 | 0.923 | 57,388 | confident |

No row was flagged as an exponent outlier (all |z| < 1.5; the closest were
Arabic/LLaMA at z=1.15 and Hindi/Kimi at z=-1.12).

## Notes

- Kimi's tokenization algorithm is confirmed `BPE (byte-level /
  byte-pair encoding)` in this run's `tokenizer_comparison.csv`, matching
  LLaMA and Qwen -- an earlier upload had it as "Unknown -- manual
  verification required"; Person 3 appears to have since resolved that.
- `confidence = low_r2_review` rows (all of English) should get a manual
  look at Person 2's log-log plot before the recommended vocab size is
  treated as final.
- Thresholds (min R^2 0.90, exponent outlier z 1.5, plateau relative
  threshold 5%) are intentionally simple and tunable -- a defensible rule
  of thumb, not a rigorously derived cutoff.