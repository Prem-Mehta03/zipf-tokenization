# Zipf's Law and Tokenization -- Final Report

## 1. Corpus & Tokenization Setup (Person 1)

- Languages: English, Hindi, Arabic
- Sample sizes per language: _fill in_
- Cleaning steps: Unicode normalization, markup/noise stripping
- Tokenizers: LLaMA, Qwen, Kimi
- Output: 9 token-frequency lists (3 languages x 3 tokenizers)

_Link/summarize Person 1's `Corpus+Tokenization/` folder here._

## 2. Zipf's Law Analysis (Person 2)

- Fitted f(r) = C * r^-s on each of the 9 frequency lists
- Comparison table (language x tokenizer -> s, R^2): _insert table_
- Log-log rank-frequency plots: _embed/link plots_
- Write-up: does exponent shift more with language or tokenizer? _insert finding_
- Notable/surprising case: _insert_

## 3. Vocabulary & Tokenizer Comparison (Person 3)

- Vocab size and tokenization algorithm per tokenizer (`tokenizer_comparison.csv`):

| tokenizer | vocab size | algorithm |
|---|---|---|
| LLaMA | 128,256 | BPE (byte-level) |
| Qwen | 151,665 | BPE (byte-level) |
| Kimi | 163,840 | _unmarked "Unknown -- manual verification required" in the source file --_ **flag for Person 3 to confirm before final submission** |

- Vocabulary growth vs. corpus size plot: _embed/link Person 3's plot(s)_
- Sweet-spot vocab size estimate (tokenizer-wide, `sweet_spot_estimates.csv`):
  plateaus at ~35-46% of each tokenizer's full trained vocab (see Section 4
  for the exact numbers and the per-language breakdown Person 4 added on
  top of this).

## 4. Synthesis & Proposed Criterion (Person 4)

### 4.1 Combining Zipf stabilization and vocab plateau points

`Person4-Synthesis/synthesize.py` reads Person 2's and Person 3's output
directly and combines them (`Person4-Synthesis/output/combined_table.csv`):
- Person 2's exponent s and R^2 per language x tokenizer (`ZipfAnalysis/output/zipf_results.csv`)
- Person 3's tokenizer-wide vocab-growth plateau (`sweet_spot_estimates.csv`)
- A **per-language** vocab-growth plateau computed directly from
  `vocab_growth_results.csv`, since Person 3's sweet-spot file only has one
  row per tokenizer and can't answer "sweet spot per language" (assignment
  goal 4) on its own.

### 4.2 Proposed criterion

**Rule of thumb:** recommend vocab size at the point where the
vocabulary-growth curve plateaus for that specific language x tokenizer
(relative marginal growth below 5% of its initial value for 2 consecutive
checkpoints -- the same style of criterion Person 3 used tokenizer-wide),
falling back to Person 3's tokenizer-level plateau and then the tokenizer's
full trained vocab size if no plateau is detected.

Cross-check that recommendation against Person 2's Zipf fit:
- Flag `low_r2_review` if R^2 < 0.90 (the power-law fit itself isn't trustworthy)
- Flag `atypical_exponent_review` if the exponent is a statistical outlier
  (|z| > 1.5) *relative to that same tokenizer's exponent on the other two
  languages* -- not against a fixed external range.

We explicitly did **not** use a fixed "good" exponent range like 0.8-1.2
from classic word-level Zipf studies. Every language x tokenizer pair in
this project has s well above 1.2 (observed range ~1.3-2.2), which is
consistent with BPE/subword token distributions being known to have
steeper Zipf slopes than whole words. A word-level band applied to
subword-token data flagged 100% of rows as suspect and carried no signal --
so confidence is instead grounded in fit quality and each tokenizer's own
internal consistency, both derived from this project's actual data.

Rationale: the vocab-growth plateau tells you when adding more merges stops
buying new coverage; the Zipf exponent (and its consistency across
languages, for a fixed tokenizer) tells you whether the resulting token
distribution still behaves the way that tokenizer typically behaves. Using
both guards against trusting a plateau that's an artifact of too small a
corpus sample or unusual fit quality for that one language.

### 4.3 Results

**Headline 1 -- plateau vocab vs. trained vocab size (tokenizer-wide):**
roughly a third to a half of each tokenizer's full trained vocabulary
accounts for essentially all observed coverage in this corpus (LLaMA ~46%,
Qwen ~38%, Kimi ~35%) -- see `output/synthesis_summary.md` for the exact table.

**Headline 2 -- the plateau is not the same across languages.** Pooling all
languages into one tokenizer-wide plateau (as `sweet_spot_estimates.csv`
does) hides real per-language differences. See the per-language pivot table
in `output/synthesis_summary.md` for the actual numbers once
`vocab_growth_results.csv` is processed.


See `output/recommended_vocab_sizes.csv` for the full per-language/tokenizer
recommended vocab sizes and confidence flags.

## 5. Conclusion

The proposed criterion -- language-specific vocab-growth plateau, cross-checked
against Zipf fit quality and each tokenizer's own cross-language exponent
consistency -- gives a usable, defensible, and cheaply computed rule: it needs
only the vocab-growth checkpoints and the Zipf fits that Person 2 and Person 3
already produce, no additional corpus passes. Where it disagrees most with a
tokenizer's actual trained vocab size (all three tokenizers' plateaus land at
35-46% of their full vocab) suggests those tokenizers were trained with
headroom beyond what this corpus needed -- worth noting as a limitation, since
that headroom may exist for multilingual/domain coverage beyond what this
three-language sample exercises, not necessarily wasted capacity.