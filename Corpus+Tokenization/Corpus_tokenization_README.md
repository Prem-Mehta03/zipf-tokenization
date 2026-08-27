# Corpus + Tokenization
 
This module covers building the multilingual Wikipedia corpus
and tokenizing it with three different tokenizers, to produce token frequency data for
downstream Zipf's Law and vocabulary analysis.
 
## What's in here
 
```
Corpus+Tokenization/
├── corpus_download.py       # Streams & cleans Wikipedia text
├── tokenize_text.py         # Tokenizes text, saves frequency data
data/
├── raw_text/             # Cleaned Wikipedia articles (1 article per line)
    ├── en_wiki.txt
    ├── hi_wiki.txt
    └── ar_wiki.txt
└── token_frequency/      # Token → frequency counts, sorted descending
   ├── en_llama_freq.json
   ├── en_qwen_freq.json
   ├── en_kimi_freq.json
   ├── hi_llama_freq.json
   ├── hi_qwen_freq.json
   ├── hi_kimi_freq.json
   ├── ar_llama_freq.json
   ├── ar_qwen_freq.json
   └── ar_kimi_freq.json
```
 
## Corpus acquisition (`corpus_download.py`)
 
- Source: `wikimedia/wikipedia` dataset on Hugging Face, streamed (not fully downloaded)
- Languages: English (`en`), Hindi (`hi`), Arabic (`ar`)
- 5,000 articles per language
- Cleaning: Unicode NFC normalization, whitespace collapsing, drop near-empty articles (<50 chars)
| Language | Articles | Total characters |
|---|---|---|
| English  | 5,000 | 69,550,405 |
| Hindi    | 5,000 | 20,389,721 |
| Arabic   | 5,000 | 52,815,316 |
 
## Tokenization (`tokenize_text.py`)
 
Each language file was tokenized using three tokenizers (tokenizer files only — no full
model weights were downloaded or run):
 
| Tokenizer | HF repo used | Note |
|---|---|---|
| LLaMA | `NousResearch/Meta-Llama-3.1-8B` | Public mirror of Meta's gated tokenizer (identical tokenizer) |
| Qwen  | `Qwen/Qwen2.5-7B` | Public |
| Kimi  | `moonshotai/Kimi-K2-Instruct` | Public |
 
**Tokenizer vocab sizes:**
 
| Tokenizer | Vocab size |
|---|---|
| LLaMA | 128,000 |
| Qwen  | 151,643 |
| Kimi  | 163,840 |
 
**Output token counts per language × tokenizer:**
 
| Language | Tokenizer | Unique tokens | Total tokens |
|---|---|---|---|
| English | LLaMA | 63,357 | 15,257,211 |
| English | Qwen  | 61,260 | 15,885,254 |
| English | Kimi  | 60,197 | 15,303,152 |
| Hindi   | LLaMA | 22,233 | 10,574,535 |
| Hindi   | Qwen  | 20,195 | 18,647,984 |
| Hindi   | Kimi  | 21,800 | 11,228,758 |
| Arabic  | LLaMA | 30,588 | 20,296,548 |
| Arabic  | Qwen  | 30,013 | 20,761,442 |
| Arabic  | Kimi  | 27,791 | 25,078,315 |
 
### Observation worth flagging
 
Hindi and Arabic consistently produce far fewer unique tokens than English across all
three tokenizers — expected, since these tokenizers are trained on English-heavy data
and represent non-Latin scripts with smaller/more fragmented subword pieces.
 
Qwen on Hindi is a notable outlier: only 20,195 unique tokens but 18.6M total tokens —
almost double LLaMA's total token count (10.5M) for the same text. This means Qwen is
breaking Hindi words into many more, smaller pieces than LLaMA does — a real efficiency
difference worth discussing in the tokenizer comparison section, not a data error.
 
## Output file format
 
Each `*_freq.json` file is a list of `[token, frequency]` pairs, sorted from most to
least frequent (rank 1 = most common token). Example:
 
```json
[
  ["the", 823145],
  [" of", 601233],
  ...
]
```
 
This format is ready to feed directly into rank-frequency (Zipf) plotting — rank = index
in the list + 1.
 
## How to reproduce
 
```bash
pip install datasets transformers sentencepiece
python corpus_download.py     # produces data/raw_text/*.txt
python tokenize_text.py       # produces data/token_frequency/*.json
```
 
Note: the first run of `tokenize_text.py` downloads tokenizer files (~hundreds of MB
total) and caches them locally; subsequent runs are faster.