"""
Step 2: Tokenize EN/HI/AR Wikipedia text with 3 tokenizers

What this does:
1. Loads the three tokenizers (LLaMA, Qwen, Kimi) from Hugging Face
   - We only download the tokenizer files here, NOT the full models,
     so this is fast even though some of these are huge models.
2. For each of the 3 language files from Step 1, runs each tokenizer
   over the text and counts how often each token appears.
3. Saves 9 output files (3 languages x 3 tokenizers), each a list of
   (token, frequency) pairs sorted from most to least common.
   This is exactly what's needed to fit Zipf's law later.

Requirements:
    pip install transformers sentencepiece

Run:
    python tokenize.py

Note: the first time you run this, each tokenizer will be downloaded
and cached locally (a few hundred MB total). Subsequent runs will be
much faster since they're read from the local cache.
"""

import json
from collections import Counter
from transformers import AutoTokenizer

# ---- CONFIG ----

# Tokenizer repos on Hugging Face. We use NousResearch's mirror for LLaMA
# because Meta's official repo is "gated" (requires manual approval that
# can take days). NousResearch hosts an identical, publicly accessible copy.
TOKENIZERS = {
    "llama": "NousResearch/Meta-Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
    "kimi": "moonshotai/Kimi-K2-Instruct",
}

# Wikipedia text files produced by corpus_download.py
LANGUAGE_FILES = {
    "en": "./data/raw_text/en_wiki.txt",
    "hi": "./data/raw_text/hi_wiki.txt",
    "ar": "./data/raw_text/ar_wiki.txt",
}


def load_all_tokenizers():
    """Download/load all 3 tokenizers once, reuse across languages."""
    loaded = {}
    for name, repo in TOKENIZERS.items():
        print(f"Loading tokenizer: {name} ({repo})...")
        # trust_remote_code=True is needed for some newer tokenizers
        # (like Kimi's) that ship custom tokenization code.
        loaded[name] = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        print(f"  -> done. Vocab size: {loaded[name].vocab_size}")
    return loaded


def tokenize_and_count(text: str, tokenizer, chunk_size: int = 2000) -> Counter:
    """
    Tokenize text and count token frequencies.

    We process in chunks (line by line, batched) instead of all at once
    because tokenizing tens of millions of characters in a single call
    can be slow/memory-heavy. Chunking also avoids any max-length limits
    some tokenizers impose.
    """
    freq = Counter()
    lines = text.split("\n")

    for i in range(0, len(lines), chunk_size):
        batch = lines[i:i + chunk_size]
        joined = " ".join(batch)
        if not joined.strip():
            continue
        tokens = tokenizer.tokenize(joined)
        freq.update(tokens)

    return freq


def main():
    tokenizers = load_all_tokenizers()

    for lang_code, filepath in LANGUAGE_FILES.items():
        print(f"\n=== Language: {lang_code} ===")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        for tok_name, tokenizer in tokenizers.items():
            print(f"  Tokenizing with {tok_name}...")
            freq = tokenize_and_count(text, tokenizer)

            out_name = f"{lang_code}_{tok_name}_freq.json"
            # most_common() sorts descending by frequency - ready for
            # Zipf rank-frequency analysis (rank 1 = most frequent token)
            with open(out_name, "w", encoding="utf-8") as out_f:
                json.dump(freq.most_common(), out_f, ensure_ascii=False, indent=2)

            print(f"    -> saved {out_name} ({len(freq)} unique tokens, "
                  f"{sum(freq.values())} total tokens)")

    print("\nAll done. You should have 9 files named like: en_llama_freq.json")


if __name__ == "__main__":
    main()