# IR Assignment 2 — Zipf's Law and Tokenization

Investigating whether tokens follow Zipf's Law across languages and tokenizers, and
whether Zipf/vocabulary-growth behavior can inform how to choose a tokenizer's
vocabulary size.

**Full report:** [`IR_Assignment2_Report.pdf`](./IR_Assignment2_Report.pdf)

## Team & Task Split

| Member | Area | Status |
|---|---|---|
| Mehta Prem | Corpus acquisition & tokenization (EN/HI/AR × LLaMA/Qwen/Kimi) | ✅ Done |
| Yash Saxena | Zipf's Law analysis across languages & tokenizers | ✅ Done |
| Rishi Kumar Motwani | Tokenizer & vocabulary comparison, vocab growth curves | ✅ Done |
| Asvin Sunderiyal | Synthesis, proposed vocab-size criterion, final report assembly | ✅ Done |

## Repo Structure

```
IR_Assignment2/
├── IR_Assignment2_Report.pdf          # Final report (LaTeX-built)
├── Corpus+Tokenization/               # Mehta Prem — see its own README for details
│   ├── corpus_download.py
│   ├── tokenize_text.py
│   └── data/
│       ├── raw_text/                  # Cleaned Wikipedia text per language
│       └── token_frequency/           # Token frequency data per language × tokenizer
├── Tokenizer_and_vocabulary_comparison/
│   ├── ZipfAnalysis/output/           # Yash Saxena — Zipf's Law analysis
│   │   ├── plots/                     # 9 rank-frequency plots (3 languages × 3 tokenizers)
│   │   ├── zipf_results.csv
│   │   └── zipf_variation_summary.csv
│   ├── VocabAnalysis/output/          # Rishi Kumar Motwani — vocab/tokenizer comparison
│   │   ├── vocab_growth.png
│   │   ├── marginal_vocab_growth.png
│   │   ├── vocab_growth_results.csv
│   │   ├── tokenizer_comparison.csv
│   │   ├── sweet_spot_estimates.csv
│   │   └── vocab_tokenizer_analysis.py
│   └── data/
└── synthesize/                        # Asvin Sunderiyal — synthesis & criterion
    ├── synthesize.py
    ├── synthesis_summary.md
    └── output/
        ├── combined_table.csv
        └── recommended_vocab_sizes.csv
```

## Assignment Goals

1. Test whether tokens follow Zipf's Law, across English, Hindi, and Arabic
2. Compare token distributions across tokenizers (LLaMA, Qwen, Kimi) — vocab size and
   tokenization strategy differences
3. Analyze how language and tokenizer choice jointly affect Zipf-law behavior
4. Investigate whether there's a "sweet spot" vocabulary size per language
5. Explore whether Zipf-law stabilization can signal when to stop tokenizer training
6. Propose an algorithm/criterion for choosing vocabulary size for a language + corpus

## Key Findings

**Corpus & tokenization (Mehta Prem):** 5,000 Wikipedia articles per language (English,
Hindi, Arabic), tokenized with LLaMA (128K vocab), Qwen (151.6K vocab), and Kimi (163.8K
vocab). Hindi and Arabic consistently produce far fewer unique tokens than English across
all three tokenizers; Qwen fragments Hindi text especially heavily (18.6M total tokens
vs. LLaMA's 10.5M on the same text, despite fewer unique tokens).

**Zipf's Law (Yash Saxena):** A single global exponent fits English poorly on the full
rank range (R²≈0.84) but very well on the head only (R²>0.99) — the opposite pattern
holds for Hindi and Arabic. English's exponent is remarkably stable across tokenizers
(range s=0.016), while Hindi's is the most volatile (range s=0.540). Averaged across
both groupings, language choice drives more exponent variation than tokenizer choice
does (mean range 0.55 vs. 0.32).

**Vocabulary & tokenizer comparison (Rishi Kumar Motwani):** English vocabulary growth
plateaus at ~40% of the processed corpus across all three tokenizers. LLaMA reaches a
higher plateau vocabulary (59,233 types, 46.2% of its trained vocab) than Qwen (37.8%) or
Kimi (34.6%), suggesting LLaMA's smaller vocabulary is more English-concentrated.

**Synthesis & criterion (Asvin Sunderiyal):** Proposed criterion — recommend vocabulary
size where relative marginal growth falls below 5% of its initial value for two
consecutive checkpoints — validated on English and cross-checked against the Zipf fits.
Full derivation and recommended vocab sizes per language×tokenizer are in
`synthesize/output/` and Section 5 of the final report.

## How to Reproduce

```bash
# 1. Corpus + tokenization
cd Corpus+Tokenization
pip install datasets transformers sentencepiece
python corpus_download.py
python tokenize_text.py

# 2. Zipf & vocab analysis
cd ../Tokenizer_and_vocabulary_comparison
python ZipfAnalysis/zipf_analysis.py         # produces ZipfAnalysis/output/
python VocabAnalysis/vocab_tokenizer_analysis.py   # produces VocabAnalysis/output/

# 3. Synthesis
cd ../synthesize
pip install pandas tabulate
python synthesize.py                          # produces synthesize/output/
```

---