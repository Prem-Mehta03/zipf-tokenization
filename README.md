# Zipf's Law and Tokenization

Investigating whether tokens follow Zipf's Law across languages and tokenizers, and
whether Zipf/vocabulary-growth behavior can inform how to choose a tokenizer's
vocabulary size.

## Team & Task Split

| Person | Area | Status |
|---|---|---|
| Mehta Prem | Corpus acquisition & tokenization (EN/HI/AR × LLaMA/Qwen/Kimi) | ✅ Done |
| Yash Saxena | Zipf's Law analysis across languages & tokenizers | ⏳ In progress |
| Rishi Kumar Motwani | Tokenizer & vocabulary comparison, vocab growth curves | ⏳ In progress |
| Asvin Sunderiyal | Synthesis, proposed vocab-size criterion, final report | ⏳ Not started |

## Repo Structure

```
IR_Assignment2/
└── Corpus+Tokenization/    
    ├── corpus_download.py
    ├── tokenize_text.py
    └── data/
        ├── raw_text/         # Cleaned Wikipedia text per language
        └── token_frequency/  # Token frequency data per language × tokenizer
```
*(Folders for Person 2/3/4's work will be added here as they're completed.)*

## Assignment Goals

1. Test whether tokens follow Zipf's Law, across English, Hindi, and Arabic
2. Compare token distributions across tokenizers (LLaMA, Qwen, Kimi) — vocab size and
   tokenization strategy differences
3. Analyze how language and tokenizer choice jointly affect Zipf-law behavior
4. Investigate whether there's a "sweet spot" vocabulary size per language
5. Explore whether Zipf-law stabilization can signal when to stop tokenizer training
6. Propose an algorithm/criterion for choosing vocabulary size for a language + corpus


*This README will be updated as Person 2, 3, and 4 add their work.*