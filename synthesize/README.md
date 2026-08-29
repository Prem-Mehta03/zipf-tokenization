# Person 4 -- Synthesis & Proposed Criterion

## What's here

- `synthesize.py` -- reads Person 2's and Person 3's output CSVs directly,
  combines them, applies a proposed vocab-size criterion, and writes a
  summary write-up.
- `final_report.md` -- template for the assembled final report (all 4 parts).

## Expects this repo layout 

```
zipf-tokenization/
|-- ZipfAnalysis/
|   `-- output/
|       |-- zipf_results.csv
|       `-- zipf_variation_summary.csv
|-- Tokenizer_and_vocabulary_comparison/
|   `-- VocabAnalysis/
|       `-- output/
|           |-- vocab_growth_results.csv
|           |-- tokenizer_comparison.csv
|           `-- sweet_spot_estimates.csv
`-- synthesis/
    |-- synthesize.py
    `-- output/          <- created by this script
```

## How to run

1. `pip install pandas tabulate`
2. `python3 synthesize.py` (run from anywhere -- paths are resolved relative
   to the script's own location, not your cwd)
3. Outputs land in `Person4-Synthesis/output/`:
   - `combined_table.csv`
   - `recommended_vocab_sizes.csv`
   - `synthesis_summary.md`
4. Fold those into `final_report.md`.

## Criterion

Recommended vocab size = the **language-specific** plateau, computed
directly from `vocab_growth_results.csv` (relative marginal growth below 5%
of its initial value for 2 consecutive checkpoints -- same style of
criterion Person 3 used tokenizer-wide). Falls back to Person 3's
tokenizer-level plateau (`observed_vocab_at_plateau`), then the tokenizer's
full trained vocab size, if no language-level plateau is detected.

Cross-checked against Person 2's Zipf fit:
- `low_r2_review` if R^2 < 0.90
- `atypical_exponent_review` if the exponent is a statistical outlier
  (|z| > 1.5) relative to that *same tokenizer's* exponent on the other
  two languages (not against a fixed external range -- see the note in
  `synthesize.py` on why a word-level 0.8-1.2 band doesn't apply to
  subword/BPE token distributions).

## Status

Runs end-to-end against Person 2's and Person 3's real output schemas
(verified with fixture data matching their exact column names and values).
Just needs this folder placed as a sibling of `ZipfAnalysis/` and
`Tokenizer_and_vocabulary_comparison/` in your local clone.

Known follow-ups:
- `tokenizer_comparison.csv` lists Kimi's algorithm as "Unknown -- manual
  verification required" -- worth asking Person 3 to fill in before final
  submission.
- English's Zipf R^2 (~0.84) is consistently lower than Hindi/Arabic
  (~0.92-0.95) across all three tokenizers -- called out as the "surprising
  case" in the generated summary, worth a follow-up look.