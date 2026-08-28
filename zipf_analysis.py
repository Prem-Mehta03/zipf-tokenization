"""
zipf_analysis.py  --  Person 2: Zipf's Law analysis across languages x tokenizers
================================================================================

WHAT THIS DOES
--------------
Consumes the token-frequency files produced by Person 1's `tokenize_text.py`
and, for every (language x tokenizer) combination:

  1. Loads the rank-frequency data.
  2. Builds the empirical rank-frequency distribution.
  3. Fits Zipf's law   f(r) = C * r^(-s)   by log-log linear regression.
  4. Saves a log-log plot per combination.
  5. Writes a comparison CSV + prints a summary table.
  6. Quantitatively compares "variation across languages" vs
     "variation across tokenizers" in the fitted exponent s.
  7. Flags one automatically-selected interesting case.

EXPECTED REPOSITORY STRUCTURE
-----------------------------
This script auto-detects the repository root by walking upwards from its own
location looking for a `data/token_frequency` (or
`Corpus+Tokenization/data/token_frequency`) directory.

NOTE ON THE ACTUAL REPO LAYOUT (verified):
The project README documents the data as living at
`Corpus+Tokenization/data/...`, but in the committed repository the data
actually sits at the REPO ROOT:

    zipf-tokenization/
    |-- Corpus+Tokenization/
    |   |-- corpus_download.py
    |   |-- tokenize_text.py
    |   `-- Corpus_tokenization_README.md
    |-- data/
    |   |-- raw_text/            en_wiki.txt, hi_wiki.txt, ar_wiki.txt
    |   `-- token_frequency/     {en,hi,ar}_{llama,qwen,kimi}_freq.json
    |-- zipf_analysis.py          <-- this file
    `-- vocab_tokenizer_analysis.py

Both layouts are searched, so the script keeps working if the data folder is
ever moved under `Corpus+Tokenization/`.

INPUT FILE FORMAT (verified against the repo)
---------------------------------------------
Each of the 9 files is JSON produced by `Counter.most_common()`:

    [ ["\u0120the", 670893], [",", 668412], ["." , 471219], ... ]

i.e. a list of [token_string, frequency] pairs already sorted descending.
Tokens are DECODED SUBWORD STRINGS (byte-level BPE, so "\u0120" == leading
space); token IDs were NOT saved by Person 1.

Filename convention: <lang>_<tokenizer>_freq.json  with
lang in {en, hi, ar} and tokenizer in {llama, qwen, kimi}.

DEPENDENCIES
------------
    pip install numpy pandas scipy matplotlib
    (optional: pip install tabulate  -- for nicer terminal tables)

HOW TO RUN
----------
    python zipf_analysis.py

OUTPUTS
-------
    ZipfAnalysis/output/zipf_results.csv
    ZipfAnalysis/output/zipf_variation_summary.csv
    ZipfAnalysis/output/plots/<language>_<tokenizer>_zipf.png
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe: write PNGs without a display server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Map the short codes used in Person 1's filenames to display names.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ar": "Arabic",
}

TOKENIZER_NAMES: Dict[str, str] = {
    "llama": "LLaMA",
    "qwen": "Qwen",
    "kimi": "Kimi",
}

# Model IDs, copied from Corpus+Tokenization/tokenize_text.py (for reporting only).
TOKENIZER_MODEL_IDS: Dict[str, str] = {
    "llama": "NousResearch/Meta-Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
    "kimi": "moonshotai/Kimi-K2-Instruct",
}

# Where Person 1's frequency files might live, relative to the repo root.
TOKEN_FREQ_DIR_CANDIDATES: List[str] = [
    "data/token_frequency",
    "Corpus+Tokenization/data/token_frequency",
]

# Output locations (relative to repo root).
OUTPUT_DIR_NAME = "ZipfAnalysis/output"
PLOTS_DIR_NAME = "ZipfAnalysis/output/plots"

# --- Fitting configuration ---------------------------------------------------
# Zipf fits are conventionally reported over the full rank range, which is what
# the assignment asks for, so FIT_MIN_RANK/FIT_MAX_RANK default to "everything".
# They are exposed because the extreme tail (tokens seen once or twice) is
# dominated by sampling noise and flattens the log-log curve; setting e.g.
# FIT_MAX_RANK = 10000 gives a cleaner "core vocabulary" fit if you want to
# report a sensitivity check in the write-up.
FIT_MIN_RANK: int = 1
FIT_MAX_RANK: Optional[int] = None  # None => use all ranks

# --- Secondary "head" fit ----------------------------------------------------
# Ordinary least squares in log-log space weights every rank equally, and there
# are vastly more high-rank points than low-rank ones (tens of thousands of
# tokens seen once or twice vs a handful of very frequent tokens). The tail
# therefore dominates the regression, which inflates s and pushes the fitted
# line above the head of the distribution.
#
# As a sensitivity check we also fit the "head" -- the most frequent
# HEAD_FIT_MAX_RANK token types -- which is the region Zipf's law is normally
# claimed to describe. Both fits are reported so the write-up can compare them.
# Set HEAD_FIT_MAX_RANK = None to disable the secondary fit.
HEAD_FIT_MAX_RANK: Optional[int] = 5000

# Frequency floor. Tokens below this are dropped before ranking.
MIN_FREQUENCY: int = 1

# Plot appearance
PLOT_DPI = 150
PLOT_FIGSIZE = (7.0, 5.5)
SCATTER_MAX_POINTS = 20000  # subsample points for plotting only (never for fitting)

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logger = logging.getLogger("zipf_analysis")


# =============================================================================
# REPOSITORY / FILE DISCOVERY
# =============================================================================

def find_repo_root(start: Path) -> Path:
    """
    Walk upwards from `start` looking for a directory that contains one of the
    known token_frequency locations. Falls back to `start` itself.
    """
    for candidate in [start] + list(start.parents):
        for rel in TOKEN_FREQ_DIR_CANDIDATES:
            if (candidate / rel).is_dir():
                return candidate
        # Secondary signal: the repo root holds Person 1's folder.
        if (candidate / "Corpus+Tokenization").is_dir():
            return candidate
    return start


def find_token_frequency_dir(repo_root: Path) -> Path:
    """Return the first existing token_frequency directory, else raise."""
    tried = []
    for rel in TOKEN_FREQ_DIR_CANDIDATES:
        path = repo_root / rel
        tried.append(str(path))
        if path.is_dir():
            logger.info("Using token frequency directory: %s", path)
            return path

    # Last resort: search the tree for *_freq.json
    matches = sorted(repo_root.rglob("*_freq.json"))
    if matches:
        found = matches[0].parent
        logger.warning(
            "Standard directories not found; falling back to discovered "
            "directory: %s", found
        )
        return found

    raise FileNotFoundError(
        "Could not locate the token frequency directory.\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n"
        "Run Person 1's Corpus+Tokenization/tokenize_text.py first, and make "
        "sure the *_freq.json files are committed."
    )


# Filename pattern: <lang>_<tokenizer>_freq.json
FREQ_FILENAME_RE = re.compile(r"^(?P<lang>[a-z]{2})_(?P<tok>[a-z0-9]+)_freq\.json$", re.I)


def discover_frequency_files(freq_dir: Path) -> List[Dict[str, object]]:
    """
    Scan `freq_dir` and infer (language, tokenizer) from each filename.

    Returns a list of dicts: {lang_code, tokenizer_code, language, tokenizer, path}
    Unrecognised files are logged and skipped (never silently ignored).
    """
    datasets: List[Dict[str, object]] = []

    for path in sorted(freq_dir.glob("*.json")):
        match = FREQ_FILENAME_RE.match(path.name)
        if not match:
            logger.warning("Skipping unrecognised filename (no <lang>_<tok>_freq.json "
                           "pattern): %s", path.name)
            continue

        lang_code = match.group("lang").lower()
        tok_code = match.group("tok").lower()

        if lang_code not in LANGUAGE_NAMES:
            logger.warning("Skipping %s: unknown language code '%s' (known: %s)",
                           path.name, lang_code, sorted(LANGUAGE_NAMES))
            continue
        if tok_code not in TOKENIZER_NAMES:
            logger.warning("Skipping %s: unknown tokenizer code '%s' (known: %s)",
                           path.name, tok_code, sorted(TOKENIZER_NAMES))
            continue

        datasets.append({
            "lang_code": lang_code,
            "tokenizer_code": tok_code,
            "language": LANGUAGE_NAMES[lang_code],
            "tokenizer": TOKENIZER_NAMES[tok_code],
            "path": path,
        })

    return datasets


def report_missing_combinations(datasets: List[Dict[str, object]]) -> None:
    """Log which of the 9 expected language x tokenizer combinations are absent."""
    present = {(d["lang_code"], d["tokenizer_code"]) for d in datasets}
    expected = {(l, t) for l in LANGUAGE_NAMES for t in TOKENIZER_NAMES}
    missing = sorted(expected - present)

    logger.info("Found %d/%d expected language x tokenizer combinations.",
                len(present & expected), len(expected))
    if missing:
        for lang, tok in missing:
            logger.warning("MISSING combination: %s + %s (expected file: %s_%s_freq.json)",
                           LANGUAGE_NAMES[lang], TOKENIZER_NAMES[tok], lang, tok)
    else:
        logger.info("All expected combinations are present.")


# =============================================================================
# LOADING
# =============================================================================

def load_frequency_file(path: Path) -> np.ndarray:
    """
    Load one token-frequency file and return frequencies sorted descending.

    Handles the format actually used in this repo -- a JSON list of
    [token, count] pairs -- and also tolerates a plain {token: count} JSON
    object, in case the tokenization step is ever re-run with a different dump.

    Raises ValueError on anything it cannot interpret (no silent skipping).
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        counts = list(data.values())
    elif isinstance(data, list):
        counts = []
        for i, item in enumerate(data):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                counts.append(item[1])
            else:
                raise ValueError(
                    f"{path.name}: entry {i} is not a [token, count] pair: {item!r}"
                )
    else:
        raise ValueError(
            f"{path.name}: expected a JSON list of [token, count] pairs or a "
            f"{{token: count}} object, got {type(data).__name__}"
        )

    freqs = np.asarray(counts, dtype=np.float64)

    if freqs.size == 0:
        raise ValueError(f"{path.name} contains no tokens.")
    if not np.all(np.isfinite(freqs)):
        raise ValueError(f"{path.name} contains non-numeric or infinite counts.")
    if np.any(freqs < 0):
        raise ValueError(f"{path.name} contains negative counts.")

    freqs = freqs[freqs >= MIN_FREQUENCY]
    if freqs.size == 0:
        raise ValueError(f"{path.name}: no tokens left after MIN_FREQUENCY filter.")

    # Person 1 already sorts via Counter.most_common(), but never rely on it.
    freqs = np.sort(freqs)[::-1]
    return freqs


# =============================================================================
# ZIPF FITTING
# =============================================================================

def build_rank_frequency(freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given descending frequencies, return (ranks, frequencies) with rank = 1,2,3..."""
    ranks = np.arange(1, freqs.size + 1, dtype=np.float64)
    return ranks, freqs


def fit_zipf(ranks: np.ndarray, freqs: np.ndarray,
             min_rank: int = FIT_MIN_RANK,
             max_rank: Optional[int] = FIT_MAX_RANK) -> Dict[str, float]:
    """
    Fit  log10(f) = log10(C) - s * log10(r)  by ordinary least squares over the
    rank window [min_rank, max_rank].

    Returns the exponent s (positive), the intercept in log10 space, the
    constant C, and R^2. The fit uses base-10 logs; the exponent s is
    log-base-invariant, and C = 10**intercept.
    """
    mask = np.ones(ranks.shape, dtype=bool)
    mask &= ranks >= min_rank
    if max_rank is not None:
        mask &= ranks <= max_rank
    mask &= freqs > 0  # log undefined at 0

    r_fit, f_fit = ranks[mask], freqs[mask]
    if r_fit.size < 3:
        raise ValueError(
            f"Only {r_fit.size} point(s) available for regression after applying "
            f"min_rank={min_rank}, max_rank={max_rank}. Need >= 3."
        )

    log_r = np.log10(r_fit)
    log_f = np.log10(f_fit)

    result = stats.linregress(log_r, log_f)

    slope = float(result.slope)
    intercept = float(result.intercept)

    return {
        "zipf_exponent_s": -slope,          # reported positive: s = -slope
        "intercept": intercept,             # log10(C)
        "zipf_constant_C": float(10.0 ** intercept),
        "r_squared": float(result.rvalue ** 2),
        "std_err_slope": float(result.stderr),
        "n_points_fitted": int(r_fit.size),
    }


def analyze_dataset(entry: Dict[str, object]) -> Dict[str, object]:
    """Load one file, fit Zipf, and return a flat result record (plus arrays)."""
    path: Path = entry["path"]  # type: ignore[assignment]
    logger.info("Processing %s  (%s + %s)", path.name, entry["language"], entry["tokenizer"])

    freqs = load_frequency_file(path)
    ranks, freqs = build_rank_frequency(freqs)

    fit = fit_zipf(ranks, freqs)

    # Secondary head-only fit (sensitivity check, see HEAD_FIT_MAX_RANK).
    head_fit: Dict[str, float] = {
        "head_max_rank": float("nan"),
        "zipf_exponent_s_head": float("nan"),
        "r_squared_head": float("nan"),
        "intercept_head": float("nan"),
    }
    if HEAD_FIT_MAX_RANK is not None and freqs.size >= 3:
        head_cap = min(HEAD_FIT_MAX_RANK, int(freqs.size))
        try:
            hf = fit_zipf(ranks, freqs, min_rank=FIT_MIN_RANK, max_rank=head_cap)
            head_fit = {
                "head_max_rank": float(head_cap),
                "zipf_exponent_s_head": hf["zipf_exponent_s"],
                "r_squared_head": hf["r_squared"],
                "intercept_head": hf["intercept"],
            }
        except ValueError as exc:
            logger.warning("    head fit skipped: %s", exc)

    record: Dict[str, object] = {
        "language": entry["language"],
        "tokenizer": entry["tokenizer"],
        "lang_code": entry["lang_code"],
        "tokenizer_code": entry["tokenizer_code"],
        "model_id": TOKENIZER_MODEL_IDS.get(str(entry["tokenizer_code"]), "unknown"),
        "source_file": path.name,
        "unique_tokens": int(freqs.size),
        "total_tokens": int(freqs.sum()),
        **fit,
        **head_fit,
    }

    logger.info("    unique=%d  total=%d  s=%.4f  R^2=%.4f  "
                "(head: s=%.4f  R^2=%.4f)",
                record["unique_tokens"], record["total_tokens"],
                record["zipf_exponent_s"], record["r_squared"],
                record["zipf_exponent_s_head"], record["r_squared_head"])

    record["_ranks"] = ranks
    record["_freqs"] = freqs
    return record


# =============================================================================
# PLOTTING
# =============================================================================

def _subsample_for_plot(ranks: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Log-uniform subsample so the PNG stays small while the curve shape (which is
    dominated by the low-rank head on a log axis) is preserved.
    """
    n = ranks.size
    if n <= SCATTER_MAX_POINTS:
        return ranks, freqs
    idx = np.unique(
        np.round(np.logspace(0, np.log10(n), SCATTER_MAX_POINTS)).astype(int) - 1
    )
    idx = idx[(idx >= 0) & (idx < n)]
    return ranks[idx], freqs[idx]


def plot_zipf(record: Dict[str, object], plots_dir: Path) -> Path:
    """Create and save one log-log rank-frequency plot with the fitted Zipf line."""
    ranks: np.ndarray = record["_ranks"]  # type: ignore[assignment]
    freqs: np.ndarray = record["_freqs"]  # type: ignore[assignment]

    s = float(record["zipf_exponent_s"])
    intercept = float(record["intercept"])
    r2 = float(record["r_squared"])

    plot_r, plot_f = _subsample_for_plot(ranks, freqs)

    # Fitted line evaluated across the fitted rank span.
    lo = max(FIT_MIN_RANK, 1)
    hi = int(ranks[-1]) if FIT_MAX_RANK is None else min(int(ranks[-1]), FIT_MAX_RANK)
    line_r = np.logspace(np.log10(lo), np.log10(hi), 200)
    line_f = (10.0 ** intercept) * line_r ** (-s)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    ax.scatter(plot_r, plot_f, s=6, alpha=0.45, edgecolors="none",
               color="#1f77b4", label="Empirical token frequencies")
    ax.plot(line_r, line_f, color="#d62728", linewidth=2.0,
            label=f"Full-range fit:  s = {s:.3f}, R² = {r2:.3f}")

    # Secondary head-only fit, if it was computed.
    s_head = float(record.get("zipf_exponent_s_head", float("nan")))
    int_head = float(record.get("intercept_head", float("nan")))
    r2_head = float(record.get("r_squared_head", float("nan")))
    head_cap = float(record.get("head_max_rank", float("nan")))
    if np.isfinite(s_head) and np.isfinite(int_head) and np.isfinite(head_cap):
        hline_r = np.logspace(np.log10(max(FIT_MIN_RANK, 1)), np.log10(head_cap), 200)
        hline_f = (10.0 ** int_head) * hline_r ** (-s_head)
        ax.plot(hline_r, hline_f, color="#2ca02c", linewidth=2.0, linestyle="--",
                label=f"Head fit (r ≤ {int(head_cap):,}):  s = {s_head:.3f}, "
                      f"R² = {r2_head:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank r (log scale)")
    ax.set_ylabel("Frequency f(r) (log scale)")
    ax.set_title(
        f"Zipf's Law — {record['language']} + {record['tokenizer']}\n"
        f"{record['model_id']}",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    stats_text = (
        f"s = {s:.4f}\n"
        f"R² = {r2:.4f}\n"
        f"C = {float(record['zipf_constant_C']):.3g}\n"
        f"unique tokens = {int(record['unique_tokens']):,}\n"
        f"total tokens = {int(record['total_tokens']):,}"
    )
    ax.text(0.03, 0.05, stats_text, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#999999", alpha=0.85))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out_path = plots_dir / f"{str(record['language']).lower()}_" \
                           f"{str(record['tokenizer']).lower()}_zipf.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)

    logger.info("    plot saved -> %s", out_path)
    return out_path


# =============================================================================
# TABLES
# =============================================================================

def print_table(df: pd.DataFrame, columns: List[str], title: str) -> None:
    """Print a DataFrame using tabulate if installed, otherwise pandas."""
    sub = df[columns]
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    try:
        from tabulate import tabulate  # optional dependency
        print(tabulate(sub, headers="keys", tablefmt="github",
                       showindex=False, floatfmt=".4f"))
    except ImportError:
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


# =============================================================================
# STEP 6 -- LANGUAGE EFFECT vs TOKENIZER EFFECT
# =============================================================================

def _spread(values: np.ndarray) -> Tuple[float, float]:
    """Return (range, sample standard deviation) of a 1-D array."""
    if values.size < 2:
        return float("nan"), float("nan")
    return float(np.max(values) - np.min(values)), float(np.std(values, ddof=1))


def analyze_variation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    """
    Quantify whether the Zipf exponent s moves more with LANGUAGE or with TOKENIZER.

    Across-language variation:
        hold the tokenizer fixed, measure the spread of s over the languages;
        average those spreads over tokenizers.
    Across-tokenizer variation:
        hold the language fixed, measure the spread of s over the tokenizers;
        average those spreads over languages.

    Returns (per-group detail table, summary dict, interpretation text).
    """
    rows: List[Dict[str, object]] = []

    # Vary language, hold tokenizer fixed.
    for tok, grp in df.groupby("tokenizer", sort=True):
        rng, sd = _spread(grp["zipf_exponent_s"].to_numpy())
        rows.append({
            "comparison": "across languages",
            "held_fixed": tok,
            "n_groups": int(len(grp)),
            "mean_s": float(grp["zipf_exponent_s"].mean()),
            "range_s": rng,
            "std_s": sd,
        })

    # Vary tokenizer, hold language fixed.
    for lang, grp in df.groupby("language", sort=True):
        rng, sd = _spread(grp["zipf_exponent_s"].to_numpy())
        rows.append({
            "comparison": "across tokenizers",
            "held_fixed": lang,
            "n_groups": int(len(grp)),
            "mean_s": float(grp["zipf_exponent_s"].mean()),
            "range_s": rng,
            "std_s": sd,
        })

    detail = pd.DataFrame(rows)

    lang_rows = detail[detail["comparison"] == "across languages"]
    tok_rows = detail[detail["comparison"] == "across tokenizers"]

    summary = {
        "mean_range_across_languages": float(lang_rows["range_s"].mean()),
        "mean_std_across_languages": float(lang_rows["std_s"].mean()),
        "mean_range_across_tokenizers": float(tok_rows["range_s"].mean()),
        "mean_std_across_tokenizers": float(tok_rows["std_s"].mean()),
    }

    lang_sd = summary["mean_std_across_languages"]
    tok_sd = summary["mean_std_across_tokenizers"]
    lang_rng = summary["mean_range_across_languages"]
    tok_rng = summary["mean_range_across_tokenizers"]

    if np.isnan(lang_sd) or np.isnan(tok_sd):
        interpretation = ("Not enough complete groups to compare language and "
                          "tokenizer effects. Check for missing combinations above.")
    else:
        if tok_sd > lang_sd:
            bigger, smaller = "tokenizer", "language"
            ratio = tok_sd / lang_sd if lang_sd > 0 else float("inf")
        elif lang_sd > tok_sd:
            bigger, smaller = "language", "tokenizer"
            ratio = lang_sd / tok_sd if tok_sd > 0 else float("inf")
        else:
            bigger = smaller = None
            ratio = 1.0

        if bigger is None:
            interpretation = (
                "Across-language and across-tokenizer variation in the Zipf exponent "
                "are effectively equal in this sample."
            )
        else:
            interpretation = (
                f"Across-{bigger} variation in the Zipf exponent is LARGER than "
                f"across-{smaller} variation "
                f"(mean SD {max(lang_sd, tok_sd):.4f} vs {min(lang_sd, tok_sd):.4f}, "
                f"a factor of {ratio:.2f}x; mean range "
                f"{max(lang_rng, tok_rng):.4f} vs {min(lang_rng, tok_rng):.4f}). "
                f"This suggests that, for this corpus, the choice of {bigger} shifts "
                f"the shape of the rank-frequency distribution more than the choice "
                f"of {smaller} does."
            )

    return detail, summary, interpretation


# =============================================================================
# STEP 7 -- INTERESTING CASE
# =============================================================================

def describe_interesting_case(df: pd.DataFrame) -> str:
    """
    Pick one automatically-selected interesting case and describe it soberly.

    Headline = the combination with the lowest R^2 (worst Zipf fit), because a
    poor fit is the most informative outcome for the write-up. Supporting lines
    report the exponent extremes and the largest deviation from the mean s.
    """
    lines: List[str] = []

    worst = df.loc[df["r_squared"].idxmin()]
    best = df.loc[df["r_squared"].idxmax()]
    high_s = df.loc[df["zipf_exponent_s"].idxmax()]
    low_s = df.loc[df["zipf_exponent_s"].idxmin()]

    mean_s = float(df["zipf_exponent_s"].mean())
    deviations = (df["zipf_exponent_s"] - mean_s).abs()
    outlier = df.loc[deviations.idxmax()]
    outlier_dev = float(deviations.max())

    lines.append(
        f"Interesting case:\n"
        f"  {worst['language']} + {worst['tokenizer']} produced the LOWEST R² "
        f"({worst['r_squared']:.4f}, s = {worst['zipf_exponent_s']:.4f}) of the "
        f"{len(df)} combinations analysed."
    )
    lines.append(
        f"  A lower R² means a single straight line in log-log space describes this "
        f"rank-frequency curve less well than it does the others. That is expected "
        f"behaviour rather than a failure: real token distributions typically bend "
        f"away from a pure power law in the high-rank tail, where many tokens occur "
        f"only once or twice, and the size of that bend depends on how finely the "
        f"tokenizer splits the text. It does not by itself show that this "
        f"language/tokenizer pair violates Zipf's law."
    )
    lines.append("")
    lines.append("Supporting observations:")
    lines.append(
        f"  Best fit:          {best['language']} + {best['tokenizer']} "
        f"(R² = {best['r_squared']:.4f})"
    )
    lines.append(
        f"  Largest exponent:  {high_s['language']} + {high_s['tokenizer']} "
        f"(s = {high_s['zipf_exponent_s']:.4f}) — steepest decay, frequency mass "
        f"concentrated in fewer token types"
    )
    lines.append(
        f"  Smallest exponent: {low_s['language']} + {low_s['tokenizer']} "
        f"(s = {low_s['zipf_exponent_s']:.4f}) — flattest decay, frequency mass "
        f"spread more evenly"
    )
    lines.append(
        f"  Furthest from the mean exponent (mean s = {mean_s:.4f}): "
        f"{outlier['language']} + {outlier['tokenizer']} "
        f"(s = {outlier['zipf_exponent_s']:.4f}, deviation {outlier_dev:.4f})"
    )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)

    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)
    logger.info("Repository root detected: %s", repo_root)

    try:
        freq_dir = find_token_frequency_dir(repo_root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    output_dir = repo_root / OUTPUT_DIR_NAME
    plots_dir = repo_root / PLOTS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    datasets = discover_frequency_files(freq_dir)
    if not datasets:
        logger.error("No usable *_freq.json files found in %s. Nothing to analyse.", freq_dir)
        return 1

    report_missing_combinations(datasets)

    # --- Steps 1-4: load, fit, plot -----------------------------------------
    records: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    for entry in datasets:
        try:
            record = analyze_dataset(entry)
            plot_zipf(record, plots_dir)
            records.append(record)
        except (ValueError, OSError) as exc:
            # Report loudly; do not pretend the combination succeeded.
            logger.error("FAILED on %s: %s", entry["path"], exc)
            failures.append((str(entry["path"]), str(exc)))

    if not records:
        logger.error("Every dataset failed to process. See errors above.")
        return 1

    # --- Step 5: results table ----------------------------------------------
    csv_columns = [
        "language", "tokenizer", "unique_tokens", "total_tokens",
        "zipf_exponent_s", "r_squared", "intercept", "zipf_constant_C",
    ]
    extra_columns = [
        "zipf_exponent_s_head", "r_squared_head", "intercept_head", "head_max_rank",
        "model_id", "source_file", "std_err_slope", "n_points_fitted",
    ]

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records])

    # Stable, human-friendly ordering.
    lang_order = {name: i for i, name in enumerate(LANGUAGE_NAMES.values())}
    tok_order = {name: i for i, name in enumerate(TOKENIZER_NAMES.values())}
    df["_lo"] = df["language"].map(lang_order).fillna(99)
    df["_to"] = df["tokenizer"].map(tok_order).fillna(99)
    df = df.sort_values(["_lo", "_to"]).drop(columns=["_lo", "_to"]).reset_index(drop=True)

    results_path = output_dir / "zipf_results.csv"
    df[csv_columns + extra_columns].to_csv(results_path, index=False)
    logger.info("Results CSV written -> %s", results_path)

    print_table(df, ["language", "tokenizer", "zipf_exponent_s", "r_squared",
                     "zipf_exponent_s_head", "r_squared_head"],
                "ZIPF FIT SUMMARY  (s = exponent, R² = goodness of log-log fit)")
    if HEAD_FIT_MAX_RANK is not None:
        print(f"  Note: '_head' columns refit using only the {HEAD_FIT_MAX_RANK:,} most "
              f"frequent token types.\n"
              f"  The full-range fit is dominated by the long tail of rare tokens, which\n"
              f"  inflates s; the head fit describes the region Zipf's law is normally\n"
              f"  claimed to hold over. Report both.\n")
    print_table(df, csv_columns, "FULL RESULTS (full-range fit)")

    # --- Step 6: language vs tokenizer effect -------------------------------
    detail, summary, interpretation = analyze_variation(df)

    variation_path = output_dir / "zipf_variation_summary.csv"
    detail.to_csv(variation_path, index=False)
    logger.info("Variation summary written -> %s", variation_path)

    print_table(detail, ["comparison", "held_fixed", "n_groups", "mean_s",
                         "range_s", "std_s"],
                "VARIATION IN THE ZIPF EXPONENT")

    print("Averaged variation measures")
    print("-" * 78)
    print(f"  Across LANGUAGES  (tokenizer held fixed):  "
          f"mean range = {summary['mean_range_across_languages']:.4f},  "
          f"mean SD = {summary['mean_std_across_languages']:.4f}")
    print(f"  Across TOKENIZERS (language held fixed):   "
          f"mean range = {summary['mean_range_across_tokenizers']:.4f},  "
          f"mean SD = {summary['mean_std_across_tokenizers']:.4f}")
    print()
    print("Interpretation")
    print("-" * 78)
    print(interpretation)
    print()

    # --- Step 7: interesting case -------------------------------------------
    print("=" * 78)
    print(describe_interesting_case(df))
    print("=" * 78)
    print()

    # --- Wrap up -------------------------------------------------------------
    if failures:
        print("The following files could not be processed:")
        for path, msg in failures:
            print(f"  - {path}: {msg}")
        print()

    logger.info("Done. %d/%d datasets analysed successfully.",
                len(records), len(datasets))
    logger.info("Plots: %s", plots_dir)
    logger.info("CSVs : %s, %s", results_path.name, variation_path.name)

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
