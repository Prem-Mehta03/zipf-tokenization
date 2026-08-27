"""
Step 1: Download & clean Wikipedia text for EN, HI, AR

What this does:
1. Streams articles from the Hugging Face 'wikimedia/wikipedia' dataset
   (streaming=True means it doesn't download the whole dataset - it pulls
   articles one at a time)
2. Cleans each article's text lightly (normalize unicode, strip extra whitespace)
3. Saves one .txt file per language

Requirements:
    pip install datasets

Run:
    python corpus_download.py
"""

import re
import unicodedata
from datasets import load_dataset

# ---- CONFIG ----
# How many articles to pull per language. 5000 is a reasonable sample size
# for this assignment - not the full Wikipedia, just enough for meaningful
# token frequency statistics.
ARTICLES_PER_LANGUAGE = 5000

# language code -> (HF config name, output filename)
LANGUAGES = {
    "en": ("20231101.en", "en_wiki.txt"),
    "hi": ("20231101.hi", "hi_wiki.txt"),
    "ar": ("20231101.ar", "ar_wiki.txt"),
}


def clean_text(text: str) -> str:
    """Light cleaning: normalize unicode, collapse whitespace."""
    # NFC normalization -
    # This makes sure characters that can be represented in multiple ways
    # (e.g. accented letters) are stored consistently.
    text = unicodedata.normalize("NFC", text)

    # Collapse multiple spaces/newlines into a single space.
    # Wikipedia articles often have extra blank lines from formatting.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def download_and_clean(lang_code: str, hf_config: str, output_file: str, n_articles: int):
    print(f"\n--- Processing {lang_code} ---")
    print(f"Streaming from wikimedia/wikipedia config: {hf_config}")

    # streaming=True: pulls articles on demand instead of downloading
    # the entire dataset file upfront.
    dataset = load_dataset(
        "wikimedia/wikipedia",
        hf_config,
        split="train",
        streaming=True,
    )

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for article in dataset:
            # Each article is a dict with fields like 'title' and 'text'.
            raw_text = article.get("text", "")
            if not raw_text:
                continue

            cleaned = clean_text(raw_text)
            if len(cleaned) < 50:
                # Skip near-empty articles (redirects, stubs, etc.)
                continue

            f.write(cleaned + "\n")
            count += 1

            if count % 500 == 0:
                print(f"  {count}/{n_articles} articles saved...")

            if count >= n_articles:
                break

    print(f"Done: saved {count} articles to {output_file}")


if __name__ == "__main__":
    for lang_code, (hf_config, output_file) in LANGUAGES.items():
        download_and_clean(lang_code, hf_config, output_file, ARTICLES_PER_LANGUAGE)

    print("\nAll languages done. You should now have:")
    for _, output_file in LANGUAGES.values():
        print(f"  - {output_file}")