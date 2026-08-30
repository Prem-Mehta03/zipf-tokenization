# Zipf's Law and Tokenization -- Final Report

## 1. Corpus & Tokenization Setup (Person 1)

- Languages: English, Hindi, Arabic
- Sample sizes per language: _fill in_
- Cleaning steps: Unicode normalization, markup/noise stripping
- Tokenizers: LLaMA, Qwen, Kimi
- Output: 9 token-frequency lists (3 languages x 3 tokenizers)

_Link/summarize Person 1's `Corpus+Tokenization/` folder here._

## 2. Zipf's Law Analysis (Person 2)

- Fitted f(r) = C * r^-s on each of the 9 frequency lists (`ZipfAnalysis/output/zipf_results.csv`)

**Comparison table (language x tokenizer -> s, R^2):**

| language | tokenizer | s | R^2 |
|---|---|---|---|
| Arabic | Kimi | 1.826 | 0.955 |
| Arabic | LLaMA | 2.200 | 0.955 |
| Arabic | Qwen | 2.233 | 0.954 |
| English | Kimi | 1.760 | 0.838 |
| English | LLaMA | 1.777 | 0.842 |
| English | Qwen | 1.775 | 0.840 |
| Hindi | Kimi | 1.547 | 0.931 |
| Hindi | LLaMA | 1.826 | 0.947 |
| Hindi | Qwen | 1.287 | 0.923 |

- Log-log rank-frequency plots: _embed/link plots from `ZipfAnalysis/output/plots/`_

**Does the exponent shift more with language or tokenizer?** From
`zipf_variation_summary.csv`: holding tokenizer fixed and varying language
gives an average exponent range of **0.55** (avg std 0.28); holding
language fixed and varying tokenizer gives an average range of **0.32**
(avg std 0.17). **Language drives more variation in s than tokenizer
choice does.** Within that: Qwen's exponent swings the most across
languages (range 0.95, from 1.29 for Hindi to 2.23 for Arabic), while
English is almost untouched by which tokenizer is used (range 0.016
across all three).

**Surprising case:** English has the lowest Zipf-fit R^2 of the three
languages for *every single tokenizer* (0.838-0.842, vs. 0.92-0.96 for
Hindi and Arabic). That consistency across all three tokenizers points at
something about the English corpus/sample or its cleaning, rather than any
one tokenizer's algorithm -- worth a follow-up look before trusting
English's Zipf fit as strongly as the other two.

## 3. Vocabulary & Tokenizer Comparison (Person 3)

- Vocab size and tokenization algorithm per tokenizer (`tokenizer_comparison.csv`):

| tokenizer | vocab size | algorithm |
|---|---|---|
| LLaMA | 128,256 | BPE (byte-level / byte-pair encoding) |
| Qwen | 151,665 | BPE (byte-level / byte-pair encoding) |
| Kimi | 163,840 | BPE (byte-level / byte-pair encoding) |

- Vocabulary growth vs. corpus size plot: _embed/link Person 3's plot(s)
  (`vocab_growth.png`, `marginal_vocab_growth.png`)_
- Sweet-spot vocab size estimate (tokenizer-wide, `sweet_spot_estimates.csv`):
  plateaus at ~35-46% of each tokenizer's full trained vocab (see Section 4
  for the exact numbers). Note: `vocab_growth_results.csv` currently only
  covers English -- consistent with the assignment requiring this plot for
  "at least one language" -- so Person 4's per-language plateau check in
  Section 4 only has English to work with.

## 4. Synthesis & Proposed Criterion (Person 4)

### 4.1 Combining Zipf stabilization and vocab plateau points

`synthesize/synthesize.py` reads Person 2's and Person 3's output directly
and combines them (`synthesize/output/combined_table.csv`):
- Person 2's exponent s and R^2 per language x tokenizer (`zipf_results.csv`)
- Person 3's tokenizer-wide vocab-growth plateau (`sweet_spot_estimates.csv`)
- A **per-language** vocab-growth plateau computed directly from
  `vocab_growth_results.csv`, since Person 3's sweet-spot file only has one
  row per tokenizer and can't answer "sweet spot per language" (assignment
  goal 4) on its own

### 4.2 Proposed criterion

**Rule of thumb:** recommend vocab size at the point where the
vocabulary-growth curve plateaus for that specific language x tokenizer
(relative marginal growth below 5% of its initial value for 2 consecutive
checkpoints -- the same style of criterion Person 3 used tokenizer-wide),
falling back to Person 3's tokenizer-level plateau, and then the
tokenizer's full trained vocab size, if no plateau is detected.

Cross-check that recommendation against Person 2's Zipf fit:
- Flag `low_r2_review` if R^2 < 0.90 (the power-law fit itself isn't trustworthy)
- Flag `atypical_exponent_review` if the exponent is a statistical outlier
  (|z| > 1.5) *relative to that same tokenizer's exponent on the other two
  languages* -- not against a fixed external range

We explicitly did **not** use a fixed "good" exponent range like 0.8-1.2
from classic word-level Zipf studies. Every language x tokenizer pair in
this project has s well above 1.2 (observed range 1.29-2.23), consistent
with BPE/subword token distributions being known to have steeper Zipf
slopes than whole words. A word-level band applied to subword-token data
would flag effectively every row as suspect and carry no signal -- so
confidence is instead grounded in fit quality and each tokenizer's own
internal consistency, both derived from this project's actual data.

Rationale: the vocab-growth plateau tells you when adding more merges stops
buying new coverage; the Zipf exponent (and its consistency across
languages, for a fixed tokenizer) tells you whether the resulting token
distribution still behaves the way that tokenizer typically behaves. Using
both guards against trusting a plateau that's an artifact of too small a
corpus sample or unusual fit quality for that one language.

### 4.3 Results

**Result 1 -- plateau vocab vs. trained vocab size (tokenizer-wide):**

| tokenizer | plateau vocab | full trained vocab | coverage |
|---|---|---|---|
| LLaMA | 59,233 | 128,256 | 46.2% |
| Qwen | 57,388 | 151,665 | 37.8% |
| Kimi | 56,759 | 163,840 | 34.6% |

Roughly a third to a half of each tokenizer's full trained vocabulary
accounts for essentially all observed coverage in this corpus.

**Result 2 -- the per-language criterion doesn't currently change the
recommendation, and that's worth stating plainly.** Applying the same
plateau criterion separately per language, instead of pooled/tokenizer-wide:
- Hindi and Arabic have no rows in `vocab_growth_results.csv` at all
  (consistent with the brief only requiring that plot for one language)
- English *does* have growth data, but its own curve never satisfies the
  plateau criterion on its own -- it's still measurably growing at the
  last sampled checkpoint, even though the same criterion applied to the
  pooled/tokenizer-wide curve does detect a plateau

As a result, `recommended_vocab_size` in `recommended_vocab_sizes.csv` is
currently identical to Person 3's tokenizer-wide plateau for all nine
language x tokenizer rows -- the per-language logic hasn't diverged from
the tokenizer-only numbers with the data available yet. Closing this gap
would need denser corpus-fraction checkpoints for English and/or
vocab-growth data for Hindi and Arabic.

**Combined results and confidence flags:**

| language | tokenizer | s | R^2 | recommended vocab | confidence |
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

No row was flagged as an exponent outlier (largest |z| = 1.15, under the
1.5 threshold). All of English is flagged `low_r2_review`, consistent with
Section 2's surprising case.

## 5. Conclusion

The proposed criterion -- vocab-growth plateau, cross-checked against Zipf
fit quality and each tokenizer's own cross-language exponent consistency --
is cheap to compute (it only needs outputs Person 2 and Person 3 already
produce) and gives a defensible tokenizer-wide answer: 35-46% of each
tokenizer's full vocab does essentially all the work in this corpus. The
one place it falls short of the full assignment ask is the *per-language*
sweet spot (goal 4): with only English having vocab-growth checkpoints, and
English's own growth curve not yet satisfying the plateau condition within
the sampled range, the criterion currently can't say whether Hindi and
Arabic would plateau at a different vocab size than English does -- that
would need either more corpus-fraction checkpoints or vocab-growth data for
all three languages to answer properly, and is the clearest next step for
this project.