# Synthesis & Proposed Criterion

## What's here

- `synthesize.py` -- searches the repo for Person 2's and Person 3's output
  CSVs by filename (not a hardcoded folder path), combines them, applies a
  proposed vocab-size criterion, and writes a summary write-up.
- `final_report.md` -- template for the assembled final report (all 4 parts).

## How path resolution works

`synthesize.py` does **not** hardcode where `ZipfAnalysis/` or
`Tokenizer_and_vocabulary_comparison/` live. It recursively searches the
repo root (the parent of this `synthesize/` folder) for these exact
filenames and uses whatever it finds, so it keeps working even if someone
reorganizes a subfolder:

```
zipf_results.csv
zipf_variation_summary.csv
vocab_growth_results.csv
tokenizer_comparison.csv
sweet_spot_estimates.csv
```

As of the current repo tree these live at:

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
`-- synthesize/
    |-- synthesize.py
    `-- output/          <- created by this script
```

If a required filename doesn't exist anywhere in the repo, or exists in
more than one place, the script fails loudly and tells you exactly what it
was looking for and where it looked -- it never silently guesses.

## How to run

1. `pip install pandas tabulate`
2. `python3 synthesize.py` (run from anywhere -- the script finds the repo
   root and searches from there, not from your cwd)
3. Outputs land in `synthesize/output/`:
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

Runs end-to-end against Person 2's and Person 3's real output schemas and
values (verified with fixture data matching your actual `zipf_results.csv`
numbers). Path discovery is filename-based, so this works regardless of
which subfolder depth Person 2's/Person 3's output ends up at.

Known follow-ups:
- Pull the real `algorithm` value for each tokenizer from
  `tokenizer_comparison.csv` into `final_report.md` -- a placeholder value
  ("Unknown -- manual verification required") was accidentally left in an
  earlier draft of the report from test-fixture data, not your real file;
  it's been removed.
- English's Zipf R^2 (~0.84) is consistently lower than Hindi/Arabic
  (~0.92-0.95) across all three tokenizers -- called out as the "surprising
  case" in the generated summary, worth a follow-up look.
- `vocab_growth_results.csv` currently appears to only have rows for one
  language (English), which is consistent with the assignment brief ("at
  least one language") but means the per-language plateau breakdown is
  English-only until/unless Hindi and Arabic rows are added.