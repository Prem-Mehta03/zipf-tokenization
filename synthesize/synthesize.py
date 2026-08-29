"""
Person 4 -- Synthesis & Proposed Vocab-Size Criterion
=====================================================

Reads Person 2's Zipf analysis output and Person 3's vocab/tokenizer
comparison output directly, combines them, and proposes a simple rule of
thumb for picking vocab size.

Expected repo layout (this script lives in Person4-Synthesis/, a sibling of
the other two folders):

    zipf-tokenization/
    |-- Corpus+Tokenization/
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
    `-- Person4-Synthesis/
        |-- synthesize.py   <- this file
        `-- output/         <- created by this script

Run from anywhere; paths below are resolved relative to this script's
location, not the current working directory.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script's own location, so `python synthesize.py`
# works regardless of the caller's cwd)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

ZIPF_DIR = os.path.join(REPO_ROOT, "ZipfAnalysis", "output")
VOCAB_DIR = os.path.join(REPO_ROOT, "Tokenizer_and_vocabulary_comparison", "VocabAnalysis", "output")

ZIPF_RESULTS_PATH = os.path.join(ZIPF_DIR, "zipf_results.csv")
ZIPF_VARIATION_PATH = os.path.join(ZIPF_DIR, "zipf_variation_summary.csv")
VOCAB_GROWTH_PATH = os.path.join(VOCAB_DIR, "vocab_growth_results.csv")
TOKENIZER_COMPARISON_PATH = os.path.join(VOCAB_DIR, "tokenizer_comparison.csv")
SWEET_SPOT_PATH = os.path.join(VOCAB_DIR, "sweet_spot_estimates.csv")

OUT_DIR = os.path.join(HERE, "output")

# ---------------------------------------------------------------------------
# Criterion parameters
#
# NOTE: an earlier version of this script used a fixed "confident" exponent
# range of (0.8, 1.2), borrowed from classic word-level Zipf studies. Running
# it against the real data showed that was wrong for this project: BPE/
# subword token distributions are well documented to have steeper Zipf
# slopes than word-level ones, and every single language x tokenizer row
# here has s well above 1.2 (observed range ~1.3-2.2). A fixed literature
# band for the wrong unit of analysis flagged 100% of rows as "review" --
# not a useful signal. Confidence below is instead based on (a) fit quality
# (R^2) and (b) whether a tokenizer's exponent for a given language is an
# outlier relative to that *same* tokenizer's exponents on the other
# languages, which is a same-tokenizer conceptual band derived from the
# data itself rather than an external constant that doesn't apply here.
# ---------------------------------------------------------------------------
MIN_R_SQUARED = 0.90                # below this, don't trust the exponent fit
EXPONENT_OUTLIER_Z = 1.5            # |z-score| within a tokenizer's own languages
PLATEAU_RELATIVE_THRESHOLD = 0.05   # mirrors Person 3's own plateau criterion
PLATEAU_CONSECUTIVE_CHECKPOINTS = 2


def _require(path, who):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} -- expected from {who}. Pull latest main and re-run.")


def load_inputs():
    _require(ZIPF_RESULTS_PATH, "Person 2 (ZipfAnalysis)")
    _require(ZIPF_VARIATION_PATH, "Person 2 (ZipfAnalysis)")
    _require(VOCAB_GROWTH_PATH, "Person 3 (Tokenizer_and_vocabulary_comparison)")
    _require(TOKENIZER_COMPARISON_PATH, "Person 3 (Tokenizer_and_vocabulary_comparison)")
    _require(SWEET_SPOT_PATH, "Person 3 (Tokenizer_and_vocabulary_comparison)")

    zipf_results = pd.read_csv(ZIPF_RESULTS_PATH)
    zipf_variation = pd.read_csv(ZIPF_VARIATION_PATH)
    vocab_growth = pd.read_csv(VOCAB_GROWTH_PATH)
    tokenizer_comparison = pd.read_csv(TOKENIZER_COMPARISON_PATH)
    sweet_spot = pd.read_csv(SWEET_SPOT_PATH)

    return zipf_results, zipf_variation, vocab_growth, tokenizer_comparison, sweet_spot


def compute_language_level_plateau(
    vocab_growth,
    relative_threshold=PLATEAU_RELATIVE_THRESHOLD,
    consecutive=PLATEAU_CONSECUTIVE_CHECKPOINTS,
):
    """
    sweet_spot_estimates.csv (Person 3) is keyed by tokenizer only, so it
    can't answer the assignment's "sweet spot per language" question (goal
    4). vocab_growth_results.csv *does* have a language column and enough
    checkpoints to compute a plateau per language x tokenizer, so we derive
    one here as a supplementary, language-aware estimate -- using the same
    style of criterion Person 3 used ("relative marginal growth below a
    threshold of its initial value, for N consecutive checkpoints"), so the
    two are comparable.
    """
    results = []
    for (lang, tok), g in vocab_growth.groupby(["language", "tokenizer"]):
        g = g.sort_values("corpus_fraction").reset_index(drop=True)

        initial = g["initial_marginal_growth"].iloc[0]
        threshold = relative_threshold * initial if pd.notna(initial) else None

        plateau_idx = None
        if threshold is not None:
            below = g["marginal_growth_relative"] < threshold
            for i in range(len(g) - consecutive + 1):
                if below.iloc[i:i + consecutive].all():
                    plateau_idx = i
                    break

        if plateau_idx is not None:
            row = g.iloc[plateau_idx]
            results.append({
                "language": lang,
                "tokenizer": tok,
                "lang_plateau_detected": True,
                "lang_plateau_corpus_fraction": row["corpus_fraction"],
                "observed_vocab_at_plateau_lang": row["unique_tokens_observed"],
            })
        else:
            row = g.iloc[-1]
            results.append({
                "language": lang,
                "tokenizer": tok,
                "lang_plateau_detected": False,
                "lang_plateau_corpus_fraction": row["corpus_fraction"],
                "observed_vocab_at_plateau_lang": row["unique_tokens_observed"],
            })

    return pd.DataFrame(results)


def flag_exponent_outliers(zipf_results, z_threshold=EXPONENT_OUTLIER_Z):
    """
    Per tokenizer, z-score each language's exponent against that tokenizer's
    own mean/std across languages. Flags a language x tokenizer row whose
    exponent is unusually far from how that tokenizer behaves elsewhere --
    a same-tokenizer, data-derived band instead of an external constant.
    """
    df = zipf_results[["language", "tokenizer", "zipf_exponent_s"]].copy()
    stats = df.groupby("tokenizer")["zipf_exponent_s"].agg(["mean", "std"]).rename(
        columns={"mean": "tok_mean_s", "std": "tok_std_s"}
    )
    df = df.merge(stats, on="tokenizer", how="left")
    df["exponent_z"] = (df["zipf_exponent_s"] - df["tok_mean_s"]) / df["tok_std_s"]
    df["exponent_outlier"] = df["exponent_z"].abs() > z_threshold
    return df[["language", "tokenizer", "exponent_z", "exponent_outlier"]]


def detect_consistent_low_fit_language(zipf_results):
    """
    Checks whether one language has the lowest R^2 across *every* tokenizer
    -- a pattern that points at the corpus/language itself rather than any
    one tokenizer, and is worth calling out as Person 2's "surprising case."
    Returns the language name if such a pattern exists, else None.
    """
    pivot = zipf_results.pivot(index="tokenizer", columns="language", values="r_squared")
    min_lang_per_tokenizer = pivot.idxmin(axis=1)
    if min_lang_per_tokenizer.nunique() == 1:
        return min_lang_per_tokenizer.iloc[0], pivot
    return None, pivot


def build_combined_table(zipf_results, tokenizer_comparison, sweet_spot, vocab_growth):
    """
    Merge language x tokenizer Zipf fits with:
      - the tokenizer-level vocab plateau Person 3 found (sweet_spot_estimates.csv)
      - a supplementary language-level plateau computed here from
        vocab_growth_results.csv, since Person 3's sweet-spot file has no
        language column
      - a same-tokenizer exponent-outlier flag (see flag_exponent_outliers)
    """
    zipf_cols = [
        "language", "tokenizer", "unique_tokens", "total_tokens",
        "zipf_exponent_s", "r_squared",
    ]
    combined = zipf_results[zipf_cols].copy()

    sweet_spot_cols = [
        "tokenizer", "plateau_detected", "plateau_corpus_fraction",
        "observed_vocab_at_plateau", "tokenizer_total_vocab",
        "vocab_coverage_percent", "criterion_used",
    ]
    combined = combined.merge(sweet_spot[sweet_spot_cols], on="tokenizer", how="left")

    tok_cols = ["tokenizer", "vocab_size", "algorithm"]
    combined = combined.merge(tokenizer_comparison[tok_cols], on="tokenizer", how="left")

    lang_plateau = compute_language_level_plateau(vocab_growth)
    combined = combined.merge(lang_plateau, on=["language", "tokenizer"], how="left")

    outliers = flag_exponent_outliers(zipf_results)
    combined = combined.merge(outliers, on=["language", "tokenizer"], how="left")

    return combined


def recommend_vocab_sizes(combined):
    df = combined.copy()

    def pick_recommended(row):
        # prefer the language-specific plateau (computed from
        # vocab_growth_results.csv) over the tokenizer-only one, since it
        # actually answers "sweet spot per language" -- fall back to
        # Person 3's tokenizer-level plateau, then the full trained vocab.
        if bool(row.get("lang_plateau_detected")) and pd.notna(row.get("observed_vocab_at_plateau_lang")):
            return row["observed_vocab_at_plateau_lang"]
        if bool(row.get("plateau_detected")) and pd.notna(row.get("observed_vocab_at_plateau")):
            return row["observed_vocab_at_plateau"]
        return row.get("tokenizer_total_vocab", row.get("vocab_size"))

    def confidence(row):
        r2 = row.get("r_squared")
        if pd.isna(r2):
            return "insufficient_data"
        if r2 < MIN_R_SQUARED:
            return "low_r2_review"
        if bool(row.get("exponent_outlier")):
            return "atypical_exponent_review"
        return "confident"

    df["recommended_vocab_size"] = df.apply(pick_recommended, axis=1)
    df["confidence"] = df.apply(confidence, axis=1)

    # how far apart are the language-level and tokenizer-level plateau
    # estimates? large gaps are worth a manual look regardless of confidence
    df["plateau_agreement_pct"] = (
        1 - (df["observed_vocab_at_plateau_lang"] - df["observed_vocab_at_plateau"]).abs()
        / df["observed_vocab_at_plateau"]
    ) * 100

    return df[[
        "language", "tokenizer", "zipf_exponent_s", "r_squared", "exponent_outlier",
        "observed_vocab_at_plateau_lang", "observed_vocab_at_plateau", "plateau_agreement_pct",
        "tokenizer_total_vocab", "recommended_vocab_size", "confidence",
    ]].sort_values(["language", "tokenizer"])


def write_summary_md(recommendation_df, zipf_variation, zipf_results, path):
    low_fit_lang, r2_pivot = detect_consistent_low_fit_language(zipf_results)

    with open(path, "w") as f:
        f.write("# Person 4 -- Synthesis & Proposed Vocab-Size Criterion\n\n")

        f.write("## Proposed criterion\n\n")
        f.write(
            "Recommend vocab size at the point where the vocabulary-growth "
            "curve plateaus, computed **per language x tokenizer** from "
            "`vocab_growth_results.csv` (same style of criterion Person 3 "
            "used tokenizer-wide: relative marginal growth below "
            f"{PLATEAU_RELATIVE_THRESHOLD:.0%} of its initial value for "
            f"{PLATEAU_CONSECUTIVE_CHECKPOINTS} consecutive checkpoints), "
            "falling back to Person 3's tokenizer-level plateau "
            "(`sweet_spot_estimates.csv`) and then the tokenizer's full "
            "trained vocab size if no plateau is found. Cross-check against "
            "Person 2's Zipf fit: rows with R^2 below "
            f"{MIN_R_SQUARED} are flagged `low_r2_review`, and rows whose "
            "exponent is a statistical outlier *relative to that same "
            "tokenizer's exponent on the other two languages* "
            f"(|z| > {EXPONENT_OUTLIER_Z}) are flagged `atypical_exponent_review`.\n\n"
        )
        f.write(
            "**Why not a fixed 'good' exponent range (e.g. 0.8-1.2)?** That "
            "range comes from word-level Zipf studies. Every language x "
            "tokenizer pair in this project has s well above 1.2 (observed "
            "range ~1.3-2.2), consistent with BPE/subword tokens being known "
            "to follow a steeper Zipf slope than whole words. Applying a "
            "word-level band here would flag 100% of rows as suspect and "
            "carry no signal, so confidence is instead based on fit quality "
            "and each tokenizer's own internal consistency across "
            "languages.\n\n"
        )

        f.write("## Headline result 1: tokenizer-level plateau vs. trained vocab size\n\n")
        f.write(
            "Per Person 3's tokenizer-wide sweet-spot estimate "
            "(`sweet_spot_estimates.csv`), the vocabulary-growth curve "
            "plateaus well under the tokenizer's full trained vocab size:\n\n"
        )
        f.write("| tokenizer | plateau vocab (tokenizer-wide) | full trained vocab | coverage |\n")
        f.write("|---|---|---|---|\n")
        for tok, row in recommendation_df.groupby("tokenizer").first().iterrows():
            full = row["tokenizer_total_vocab"]
            plateau = row["observed_vocab_at_plateau"]
            pct = plateau / full * 100 if pd.notna(full) and full else float("nan")
            f.write(f"| {tok} | {plateau:.0f} | {full:.0f} | {pct:.1f}% |\n")
        f.write(
            "\nRoughly a third to a half of each tokenizer's trained "
            "vocabulary is doing essentially all the work -- the rest sees "
            "vanishingly few additional unique tokens as the corpus grows "
            "further.\n\n"
        )

        f.write("## Headline result 2: the plateau is not the same across languages\n\n")
        f.write(
            "The tokenizer-wide number above hides real per-language "
            "differences. Computing the plateau separately per language "
            "(from `vocab_growth_results.csv`) instead of pooling all "
            "languages together:\n\n"
        )
        lang_pivot = recommendation_df.pivot(
            index="language", columns="tokenizer", values="observed_vocab_at_plateau_lang"
        )
        f.write(lang_pivot.round(0).to_markdown())
        f.write(
            "\n\nThe recommended vocab size per language-tokenizer pair "
            "(`recommended_vocab_size` in the combined table below) uses "
            "this language-level plateau first, since it's the number that "
            "actually answers 'what's the sweet spot for *this* language,' "
            "and only falls back to the tokenizer-wide figure when a "
            "language-level plateau can't be detected.\n\n"
        )

        f.write("## Does the Zipf exponent shift more with language or tokenizer?\n\n")
        f.write(
            "From `zipf_variation_summary.csv` (Person 2's own comparison, "
            "reused here rather than recomputed):\n\n"
        )
        f.write(zipf_variation.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Surprising case\n\n")
        if low_fit_lang is not None:
            f.write(
                f"**{low_fit_lang}** has the lowest Zipf-fit R^2 of the three "
                f"languages for *every single tokenizer* (LLaMA, Qwen, and "
                "Kimi alike) -- not just one tokenizer/language pairing. "
                "That consistency across tokenizers points at something "
                "about the corpus or language itself (sample composition, "
                "cleaning, script-specific tokenization behavior) rather "
                "than any one tokenizer's algorithm. Worth a follow-up look "
                "before trusting that language's Zipf fit as strongly as "
                "the other two.\n\n"
            )
            f.write(r2_pivot.round(3).to_markdown())
            f.write("\n\n")
        else:
            f.write(
                "No single language had the lowest R^2 across all three "
                "tokenizers -- the fit-quality pattern isn't consistent "
                "enough to call out one language as an outlier.\n\n"
            )
            f.write(r2_pivot.round(3).to_markdown())
            f.write("\n\n")

        f.write("## Combined results\n\n")
        f.write(recommendation_df.round(3).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Notes\n\n")
        f.write(
            "- `observed_vocab_at_plateau_lang` is computed by this script "
            "directly from `vocab_growth_results.csv`; `observed_vocab_at_plateau` "
            "is Person 3's tokenizer-only estimate from `sweet_spot_estimates.csv`. "
            "`plateau_agreement_pct` shows how close the two are -- large gaps "
            "are worth a manual look regardless of the confidence flag.\n"
            "- `confidence = low_r2_review` or `atypical_exponent_review` rows "
            "should get a manual look at Person 2's log-log plot before the "
            "recommended vocab size is treated as final.\n"
            f"- Thresholds (min R^2 {MIN_R_SQUARED}, exponent outlier z "
            f"{EXPONENT_OUTLIER_Z}, plateau relative threshold "
            f"{PLATEAU_RELATIVE_THRESHOLD:.0%}) are intentionally simple and "
            "tunable -- a defensible rule of thumb, not a rigorously derived "
            "cutoff.\n"
        )
    return path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    zipf_results, zipf_variation, vocab_growth, tokenizer_comparison, sweet_spot = load_inputs()

    combined = build_combined_table(zipf_results, tokenizer_comparison, sweet_spot, vocab_growth)
    combined.to_csv(os.path.join(OUT_DIR, "combined_table.csv"), index=False)

    recs = recommend_vocab_sizes(combined)
    recs.to_csv(os.path.join(OUT_DIR, "recommended_vocab_sizes.csv"), index=False)

    summary_path = write_summary_md(recs, zipf_variation, zipf_results, os.path.join(OUT_DIR, "synthesis_summary.md"))

    print(f"Wrote {OUT_DIR}/combined_table.csv")
    print(f"Wrote {OUT_DIR}/recommended_vocab_sizes.csv")
    print(f"Wrote {summary_path}")