"""
Person 4 -- Synthesis & Proposed Vocab-Size Criterion
=====================================================

Reads Person 2's Zipf analysis output and Person 3's vocab/tokenizer
comparison output, combines them, and proposes a simple rule of thumb for
picking vocab size.

Path resolution is generic: rather than hardcoding the exact folder depth
each input file lives at (which is an assumption that breaks the moment
someone reorganizes a subfolder), this script recursively searches the
repo for each required filename by name and uses whatever it finds. The
only contract this script depends on is the filenames themselves:

    zipf_results.csv
    zipf_variation_summary.csv
    vocab_growth_results.csv
    tokenizer_comparison.csv
    sweet_spot_estimates.csv

as of the current repo tree these live at:

    ZipfAnalysis/output/zipf_results.csv
    ZipfAnalysis/output/zipf_variation_summary.csv
    Tokenizer_and_vocabulary_comparison/VocabAnalysis/output/vocab_growth_results.csv
    Tokenizer_and_vocabulary_comparison/VocabAnalysis/output/tokenizer_comparison.csv
    Tokenizer_and_vocabulary_comparison/VocabAnalysis/output/sweet_spot_estimates.csv

but if Person 2 or Person 3 move their output folder, this script keeps
working as long as the filenames stay the same. If a filename can't be
found anywhere under the repo root, or is found in more than one place,
the script fails loudly with the path(s) it looked at rather than silently
picking one.

This script itself lives in `synthesize/` at the repo root, and writes its
own output to `synthesize/output/`:
    combined_table.csv
    recommended_vocab_sizes.csv
    zipf_variation_summary.csv   (passed through unchanged, for reference)

This script only produces CSVs -- it does not generate a markdown
write-up. Upload the CSVs from `synthesize/output/` to get a written
summary/report; that step happens separately, from the real numbers, not
from this script's own guesses about them.
"""

import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution -- generic, filename-based discovery under the repo root
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "output")

REQUIRED_FILES = {
    "zipf_results": "zipf_results.csv",
    "zipf_variation": "zipf_variation_summary.csv",
    "vocab_growth": "vocab_growth_results.csv",
    "tokenizer_comparison": "tokenizer_comparison.csv",
    "sweet_spot": "sweet_spot_estimates.csv",
}


def find_file(filename, search_root=REPO_ROOT, exclude_dir=OUT_DIR):
    """
    Recursively search search_root for a file named `filename`, ignoring
    this script's own output directory (so re-running the script doesn't
    accidentally pick up its own previous output). Fails loudly rather than
    silently guessing if the file is missing or ambiguous.
    """
    matches = [
        p for p in glob.glob(os.path.join(search_root, "**", filename), recursive=True)
        if not os.path.abspath(p).startswith(os.path.abspath(exclude_dir))
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under {search_root}. "
            f"Make sure Person 2's/Person 3's output has been pulled into your clone."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found more than one '{filename}' under {search_root}:\n  "
            + "\n  ".join(matches)
            + "\nRemove the stale copy or rename one so this script knows which to use."
        )
    return matches[0]


def resolve_input_paths():
    return {key: find_file(filename) for key, filename in REQUIRED_FILES.items()}


# ---------------------------------------------------------------------------
# Criterion parameters
#
# NOTE: an earlier version of this script used a fixed "confident" exponent
# range of (0.8, 1.2), borrowed from classic word-level Zipf studies. Running
# it against the real data showed that was wrong for this project: BPE/
# subword token distributions are well documented to have steeper Zipf
# slopes than word-level ones, and every language x tokenizer row observed
# so far has s well above 1.2 (~1.3-2.2). A fixed literature band for the
# wrong unit of analysis flagged effectively every row as "review" -- not a
# useful signal. Confidence below is instead based on (a) fit quality (R^2)
# and (b) whether a tokenizer's exponent for a given language is an outlier
# relative to that *same* tokenizer's exponents on the other languages,
# which is a data-derived band rather than an external constant.
# ---------------------------------------------------------------------------
MIN_R_SQUARED = 0.90                # below this, don't trust the exponent fit
EXPONENT_OUTLIER_Z = 1.5            # |z-score| within a tokenizer's own languages
PLATEAU_RELATIVE_THRESHOLD = 0.05   # mirrors Person 3's own plateau criterion
PLATEAU_CONSECUTIVE_CHECKPOINTS = 2


def load_inputs():
    paths = resolve_input_paths()
    zipf_results = pd.read_csv(paths["zipf_results"])
    zipf_variation = pd.read_csv(paths["zipf_variation"])
    vocab_growth = pd.read_csv(paths["vocab_growth"])
    tokenizer_comparison = pd.read_csv(paths["tokenizer_comparison"])
    sweet_spot = pd.read_csv(paths["sweet_spot"])
    return zipf_results, zipf_variation, vocab_growth, tokenizer_comparison, sweet_spot, paths


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
    one tokenizer. Kept as a helper for ad-hoc use; not called by default
    since this script only writes CSVs now (see module docstring).
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




if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    zipf_results, zipf_variation, vocab_growth, tokenizer_comparison, sweet_spot, paths = load_inputs()

    print("Resolved input files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    combined = build_combined_table(zipf_results, tokenizer_comparison, sweet_spot, vocab_growth)
    combined.to_csv(os.path.join(OUT_DIR, "combined_table.csv"), index=False)

    recs = recommend_vocab_sizes(combined)
    recs.to_csv(os.path.join(OUT_DIR, "recommended_vocab_sizes.csv"), index=False)

    # copy zipf_variation_summary.csv through unchanged -- useful context for
    # whoever writes the final summary/report, even though this script
    # doesn't generate that write-up itself
    zipf_variation.to_csv(os.path.join(OUT_DIR, "zipf_variation_summary.csv"), index=False)

    print(f"Wrote {OUT_DIR}/combined_table.csv")
    print(f"Wrote {OUT_DIR}/recommended_vocab_sizes.csv")
    print(f"Wrote {OUT_DIR}/zipf_variation_summary.csv")
    print()
    print("This script only writes CSVs -- no markdown summary is generated.")
    print("Upload the three CSVs above to get a written summary/report.")