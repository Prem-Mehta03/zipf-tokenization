"""
vocab_tokenizer_analysis.py  --  Person 3: Tokenizer & vocabulary comparison
============================================================================

WHAT THIS DOES
--------------
Part A  Compares the three tokenizers used by Person 1 (LLaMA / Qwen / Kimi):
        vocab size, class, backend, tokenization algorithm, special tokens.
Part B  Measures how the number of DISTINCT token types observed grows as more
        of the raw corpus is processed (without retraining any tokenizer).
Part C  Plots the vocabulary growth curves for all three tokenizers.
Part D  Computes marginal vocabulary growth between consecutive checkpoints.
Part E  Estimates a "plateau" / sweet spot with a transparent heuristic.
Part F  Prints an automatic interpretation of the numbers.

EXPECTED REPOSITORY STRUCTURE
-----------------------------
The script auto-detects the repo root by walking upwards looking for a
`data/raw_text` (or `Corpus+Tokenization/data/raw_text`) directory.

NOTE ON THE ACTUAL REPO LAYOUT (verified):
The README documents the data under `Corpus+Tokenization/data/...`, but in the
committed repository it actually lives at the REPO ROOT:

    zipf-tokenization/
    |-- Corpus+Tokenization/
    |   |-- corpus_download.py
    |   `-- tokenize_text.py
    |-- data/
    |   |-- raw_text/            en_wiki.txt, hi_wiki.txt, ar_wiki.txt
    |   `-- token_frequency/     {en,hi,ar}_{llama,qwen,kimi}_freq.json
    |-- zipf_analysis.py
    `-- vocab_tokenizer_analysis.py   <-- this file

Both layouts are searched.

MODEL IDs
---------
Taken verbatim from Corpus+Tokenization/tokenize_text.py:
    llama -> NousResearch/Meta-Llama-3.1-8B
    qwen  -> Qwen/Qwen2.5-7B
    kimi  -> moonshotai/Kimi-K2-Instruct

CONSISTENCY WITH PERSON 1
-------------------------
Person 1 counted frequencies over `tokenizer.tokenize(...)` output, i.e.
decoded subword STRINGS, and batched 2000 lines at a time. This script uses the
same call and the same batching so the counts are directly comparable. Because
a tokenizer's string<->id mapping is one-to-one, the number of distinct token
strings equals the number of distinct token IDs; the final distinct set is
converted to IDs so the reported quantity really is "unique token IDs observed".

DEPENDENCIES
------------
    pip install transformers sentencepiece numpy pandas matplotlib
    (optional: pip install tabulate)

Requires network access on first run to download the tokenizer files from the
Hugging Face Hub (tokenizers only, not the full models). They are cached
afterwards.

HOW TO RUN
----------
    python vocab_tokenizer_analysis.py                 # default: English
    python vocab_tokenizer_analysis.py --language hi   # Hindi
    python vocab_tokenizer_analysis.py --language ar --max-lines 1000

RUNTIME WARNING
---------------
The full English corpus is ~70 MB and is tokenized once per tokenizer, which
takes on the order of several minutes each. Use --max-lines to subsample while
developing (e.g. --max-lines 500).

OUTPUTS
-------
    VocabAnalysis/output/tokenizer_comparison.csv
    VocabAnalysis/output/vocab_growth_results.csv
    VocabAnalysis/output/sweet_spot_estimates.csv
    VocabAnalysis/output/vocab_growth.png
    VocabAnalysis/output/marginal_vocab_growth.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model IDs copied from Corpus+Tokenization/tokenize_text.py -- do not change
# these without also re-running Person 1's tokenization step.
TOKENIZERS: Dict[str, str] = {
    "llama": "NousResearch/Meta-Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
    "kimi": "moonshotai/Kimi-K2-Instruct",
}

TOKENIZER_DISPLAY: Dict[str, str] = {
    "llama": "LLaMA",
    "qwen": "Qwen",
    "kimi": "Kimi",
}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ar": "Arabic",
}

# Part B default language (override with --language).
DEFAULT_LANGUAGE = "en"

# Raw corpus filename pattern used by Person 1's corpus_download.py.
RAW_TEXT_PATTERN = "{lang}_wiki.txt"

RAW_TEXT_DIR_CANDIDATES: List[str] = [
    "data/raw_text",
    "Corpus+Tokenization/data/raw_text",
]
TOKEN_FREQ_DIR_CANDIDATES: List[str] = [
    "data/token_frequency",
    "Corpus+Tokenization/data/token_frequency",
]

OUTPUT_DIR_NAME = "VocabAnalysis/output"

# Corpus checkpoints, as fractions of the corpus (by line/article count).
# Must be ascending and end at 1.0 to use the whole corpus.
CORPUS_FRACTIONS: Sequence[float] = [
    0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
]

# Lines joined per tokenizer call. Matches Person 1's chunk_size.
TOKENIZE_CHUNK_LINES = 2000

# Part E: plateau is declared when marginal growth drops below this fraction of
# the initial marginal growth...
PLATEAU_THRESHOLD = 0.05
# ...and stays below it for this many consecutive checkpoints (guards against a
# single noisy checkpoint triggering a false plateau).
PLATEAU_CONSECUTIVE = 2

PLOT_DPI = 150
PLOT_FIGSIZE = (8.0, 5.5)
TOKENIZER_COLORS = {"llama": "#1f77b4", "qwen": "#d62728", "kimi": "#2ca02c"}
TOKENIZER_MARKERS = {"llama": "o", "qwen": "s", "kimi": "^"}

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logger = logging.getLogger("vocab_tokenizer_analysis")

UNKNOWN = "Unknown — manual verification required"


# =============================================================================
# REPOSITORY DISCOVERY
# =============================================================================

def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        for rel in RAW_TEXT_DIR_CANDIDATES + TOKEN_FREQ_DIR_CANDIDATES:
            if (candidate / rel).is_dir():
                return candidate
        if (candidate / "Corpus+Tokenization").is_dir():
            return candidate
    return start


def find_dir(repo_root: Path, candidates: Sequence[str], what: str) -> Optional[Path]:
    for rel in candidates:
        path = repo_root / rel
        if path.is_dir():
            logger.info("Using %s directory: %s", what, path)
            return path
    logger.warning("Could not find a %s directory. Tried: %s", what,
                   ", ".join(str(repo_root / r) for r in candidates))
    return None


# =============================================================================
# PART A -- TOKENIZER COMPARISON
# =============================================================================

def load_tokenizer(name: str, model_id: str):
    """
    Load one tokenizer from the Hugging Face Hub. Returns None on failure and
    logs the reason (network, gating, missing dependency) rather than crashing.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.error("transformers is not installed (%s). Run: pip install transformers "
                     "sentencepiece", exc)
        return None

    logger.info("Loading tokenizer %-6s (%s) ...", name, model_id)
    try:
        # trust_remote_code=True matches Person 1's call; Kimi ships custom code.
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # broad on purpose: hub errors are many and varied
        logger.error("FAILED to load %s (%s): %s: %s",
                     name, model_id, type(exc).__name__, exc)
        logger.error("  Check network access to huggingface.co, whether the repo is "
                     "gated, and that `sentencepiece` is installed.")
        return None

    logger.info("  loaded. vocab_size=%s", getattr(tok, "vocab_size", "?"))
    return tok


def detect_algorithm(tok) -> Tuple[str, str]:
    """
    Determine the tokenization algorithm, preferring hard evidence over guesses.

    Order of evidence:
      1. The fast tokenizer's backend model type (from tokenizer.json) -- the
         authoritative source: "BPE", "Unigram", "WordPiece", "WordLevel".
      2. The serialized backend JSON, if the attribute route fails.
      3. The tokenizer class name, as a weak fallback.
    Returns (algorithm, evidence). Never invents an answer: if nothing is
    conclusive, returns UNKNOWN.
    """
    # 1. Backend model class (fast/Rust tokenizers only).
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        model = getattr(backend, "model", None)
        if model is not None:
            cls_name = type(model).__name__
            mapping = {
                "BPE": "BPE (byte-level / byte-pair encoding)",
                "Unigram": "Unigram (SentencePiece-style)",
                "WordPiece": "WordPiece",
                "WordLevel": "WordLevel",
            }
            if cls_name in mapping:
                return mapping[cls_name], f"backend_tokenizer.model class = {cls_name}"

        # 2. Serialized backend JSON.
        try:
            spec = json.loads(backend.to_str())
            model_type = spec.get("model", {}).get("type")
            if model_type:
                return str(model_type), "tokenizer.json -> model.type"
        except Exception:  # serialization is best-effort only
            pass

    # 3. Class-name fallback -- weak evidence, labelled as such.
    cls = type(tok).__name__
    if "SentencePiece" in cls or "Llama" in cls:
        return (UNKNOWN,
                f"slow tokenizer class '{cls}'; backend unavailable, so the "
                f"algorithm could not be confirmed")
    return UNKNOWN, f"no backend model available; class = {cls}"


def summarize_special_tokens(tok) -> str:
    """Compact, readable summary of the special tokens, or an explicit 'none'."""
    parts: List[str] = []
    for attr in ("bos_token", "eos_token", "unk_token", "pad_token",
                 "sep_token", "cls_token", "mask_token"):
        value = getattr(tok, attr, None)
        if value:
            parts.append(f"{attr}={value}")
    extra = getattr(tok, "additional_special_tokens", None) or []
    if extra:
        shown = ", ".join(map(str, extra[:5]))
        suffix = f", +{len(extra) - 5} more" if len(extra) > 5 else ""
        parts.append(f"additional[{len(extra)}]=({shown}{suffix})")
    return "; ".join(parts) if parts else "none declared"


def describe_tokenizer(name: str, model_id: str, tok) -> Dict[str, object]:
    """Build one row of the tokenizer comparison table."""
    if tok is None:
        return {
            "tokenizer": TOKENIZER_DISPLAY.get(name, name),
            "model_id": model_id,
            "vocab_size": np.nan,
            "vocab_size_with_added": np.nan,
            "tokenizer_class": UNKNOWN,
            "backend": UNKNOWN,
            "algorithm": UNKNOWN,
            "special_tokens": UNKNOWN,
            "notes": "Tokenizer could not be loaded — see the error log above.",
        }

    is_fast = bool(getattr(tok, "is_fast", False))
    backend = "Rust (HF `tokenizers`, fast)" if is_fast else "Python (slow tokenizer)"

    base_vocab = getattr(tok, "vocab_size", None)
    try:
        full_vocab = len(tok)  # includes added/special tokens
    except Exception:
        full_vocab = None

    algorithm, evidence = detect_algorithm(tok)

    added = getattr(tok, "get_added_vocab", lambda: {})() or {}
    notes = [f"algorithm evidence: {evidence}"]
    if base_vocab is not None and full_vocab is not None and full_vocab != base_vocab:
        notes.append(f"{full_vocab - base_vocab} added/special token(s) beyond the "
                     f"base vocab ({len(added)} in added_vocab)")
    if getattr(tok, "model_max_length", None) and tok.model_max_length < 10 ** 6:
        notes.append(f"model_max_length={tok.model_max_length}")

    return {
        "tokenizer": TOKENIZER_DISPLAY.get(name, name),
        "model_id": model_id,
        "vocab_size": base_vocab if base_vocab is not None else np.nan,
        "vocab_size_with_added": full_vocab if full_vocab is not None else np.nan,
        "tokenizer_class": type(tok).__name__,
        "backend": backend,
        "algorithm": algorithm,
        "special_tokens": summarize_special_tokens(tok),
        "notes": "; ".join(notes),
    }


# =============================================================================
# TABLE PRINTING
# =============================================================================

def print_table(df: pd.DataFrame, columns: Sequence[str], title: str,
                floatfmt: str = ".4f") -> None:
    sub = df[list(columns)]
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    try:
        from tabulate import tabulate
        print(tabulate(sub, headers="keys", tablefmt="github",
                       showindex=False, floatfmt=floatfmt))
    except ImportError:
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.max_colwidth", 60):
            print(sub.to_string(index=False))
    print()


# =============================================================================
# PART B -- VOCABULARY GROWTH
# =============================================================================

def load_corpus_lines(raw_dir: Path, lang: str,
                      max_lines: Optional[int] = None) -> List[str]:
    """Load the cleaned corpus for `lang`. One article per line (Person 1's format)."""
    path = raw_dir / RAW_TEXT_PATTERN.format(lang=lang)
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            f"Expected a file named '{RAW_TEXT_PATTERN.format(lang=lang)}' produced by "
            f"Corpus+Tokenization/corpus_download.py."
        )

    logger.info("Reading corpus: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln for ln in (l.strip() for l in fh) if ln]

    if max_lines is not None:
        lines = lines[:max_lines]
        logger.info("  limited to the first %d lines (--max-lines)", len(lines))

    if not lines:
        raise ValueError(f"{path} contains no non-empty lines.")

    logger.info("  %d articles/lines loaded.", len(lines))
    return lines


def fraction_boundaries(n_lines: int, fractions: Sequence[float]) -> List[Tuple[float, int]]:
    """
    Convert fractions to strictly increasing line-count checkpoints.

    Duplicates (which happen when the corpus is small and two fractions round to
    the same line count) are dropped so no checkpoint adds zero new text.
    """
    boundaries: List[Tuple[float, int]] = []
    seen: set = set()
    for frac in sorted(fractions):
        if not 0 < frac <= 1.0:
            logger.warning("Ignoring out-of-range corpus fraction: %s", frac)
            continue
        n = max(1, int(round(frac * n_lines)))
        if n in seen:
            logger.info("  fraction %.2f maps to the same %d lines as an earlier "
                        "checkpoint — skipping", frac, n)
            continue
        seen.add(n)
        boundaries.append((frac, n))
    return boundaries


def measure_vocab_growth(lines: List[str], tok, tok_name: str,
                         fractions: Sequence[float]) -> List[Dict[str, object]]:
    """
    Walk the corpus once, accumulating the set of distinct token strings, and
    record the state at each checkpoint.

    This is incremental: text between checkpoint k-1 and k is tokenized exactly
    once, never re-tokenized. Uses tokenizer.tokenize() with the same 2000-line
    batching Person 1 used, so results are comparable with the frequency files.
    """
    boundaries = fraction_boundaries(len(lines), fractions)
    if not boundaries:
        raise ValueError("No usable corpus fractions — check CORPUS_FRACTIONS.")

    seen_tokens: set = set()
    total_tokens = 0
    total_chars = 0
    cursor = 0
    records: List[Dict[str, object]] = []

    for frac, end_line in boundaries:
        # Tokenize only the newly added slice of the corpus.
        for start in range(cursor, end_line, TOKENIZE_CHUNK_LINES):
            batch = lines[start:min(start + TOKENIZE_CHUNK_LINES, end_line)]
            joined = " ".join(batch)
            if not joined.strip():
                continue
            total_chars += len(joined)
            tokens = tok.tokenize(joined)
            total_tokens += len(tokens)
            seen_tokens.update(tokens)

        cursor = end_line

        records.append({
            "tokenizer_code": tok_name,
            "tokenizer": TOKENIZER_DISPLAY.get(tok_name, tok_name),
            "corpus_fraction": frac,
            "lines_processed": end_line,
            "characters_processed": total_chars,
            "tokens_processed": total_tokens,
            "unique_tokens_observed": len(seen_tokens),
        })

        logger.info("  %-5s frac=%.2f  lines=%6d  tokens=%12d  unique=%7d",
                    tok_name, frac, end_line, total_tokens, len(seen_tokens))

    # Sanity check: distinct strings should map 1:1 onto distinct IDs.
    try:
        ids = tok.convert_tokens_to_ids(list(seen_tokens))
        n_ids = len({i for i in ids if i is not None})
        if n_ids != len(seen_tokens):
            logger.warning("  %s: %d distinct token strings mapped to %d distinct IDs "
                           "(expected equal). Reporting distinct strings.",
                           tok_name, len(seen_tokens), n_ids)
    except Exception as exc:
        logger.warning("  %s: could not verify string->id mapping (%s).", tok_name, exc)

    return records


def add_vocab_metadata(df: pd.DataFrame, comparison: pd.DataFrame,
                       language: str) -> pd.DataFrame:
    """Attach language, total tokenizer vocab, and coverage percentage."""
    vocab_lookup = dict(zip(comparison["tokenizer"], comparison["vocab_size_with_added"]))
    base_lookup = dict(zip(comparison["tokenizer"], comparison["vocab_size"]))

    df = df.copy()
    df.insert(0, "language", language)
    df["total_tokenizer_vocab"] = df["tokenizer"].map(vocab_lookup)
    # Fall back to the base vocab if the "with added" figure is unavailable.
    df["total_tokenizer_vocab"] = df["total_tokenizer_vocab"].fillna(
        df["tokenizer"].map(base_lookup)
    )
    df["vocab_coverage_percent"] = (
        100.0 * df["unique_tokens_observed"] / df["total_tokenizer_vocab"]
    )
    return df


# =============================================================================
# PART D -- MARGINAL GROWTH
# =============================================================================

def add_marginal_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per tokenizer, compute for consecutive checkpoints:

        new_unique_tokens        = U(k) - U(k-1)
        additional_tokens        = T(k) - T(k-1)
        marginal_growth          = new_unique / additional_tokens
                                   (new token types discovered per token read)
        marginal_growth_relative = marginal_growth / (first marginal_growth)

    The first checkpoint has no predecessor, so it is seeded from the origin
    (0 tokens, 0 unique types), which is the natural baseline for "how fast did
    discovery start".
    """
    out = []
    for tok, grp in df.groupby("tokenizer", sort=False):
        grp = grp.sort_values("corpus_fraction").copy()

        prev_unique = grp["unique_tokens_observed"].shift(1).fillna(0.0)
        prev_tokens = grp["tokens_processed"].shift(1).fillna(0.0)

        new_unique = grp["unique_tokens_observed"] - prev_unique
        additional = grp["tokens_processed"] - prev_tokens

        grp["new_unique_tokens"] = new_unique
        grp["additional_tokens_processed"] = additional
        grp["marginal_growth"] = np.where(additional > 0, new_unique / additional, np.nan)

        first_valid = grp["marginal_growth"].dropna()
        baseline = float(first_valid.iloc[0]) if not first_valid.empty else np.nan
        grp["initial_marginal_growth"] = baseline
        grp["marginal_growth_relative"] = (
            grp["marginal_growth"] / baseline if baseline and baseline > 0 else np.nan
        )

        out.append(grp)

    return pd.concat(out, ignore_index=True)


# =============================================================================
# PART E -- PLATEAU / SWEET SPOT
# =============================================================================

def estimate_plateau(df: pd.DataFrame, threshold: float = PLATEAU_THRESHOLD,
                     consecutive: int = PLATEAU_CONSECUTIVE) -> pd.DataFrame:
    """
    Heuristic: the plateau is the first checkpoint at which relative marginal
    growth has been below `threshold` for `consecutive` checkpoints in a row.

    Deliberately simple and fully transparent -- it is a description of where
    returns diminish in THIS corpus, not a proof of an optimum.
    """
    rows: List[Dict[str, object]] = []

    for tok, grp in df.groupby("tokenizer", sort=False):
        grp = grp.sort_values("corpus_fraction").reset_index(drop=True)
        rel = grp["marginal_growth_relative"].to_numpy(dtype=float)

        criterion = (f"relative marginal growth < {threshold:.2f} of initial, "
                     f"for {consecutive} consecutive checkpoints")

        hit_index: Optional[int] = None
        run = 0
        for i, value in enumerate(rel):
            if np.isfinite(value) and value < threshold:
                run += 1
                if run >= consecutive:
                    hit_index = i
                    break
            else:
                run = 0

        if hit_index is None:
            rows.append({
                "tokenizer": tok,
                "plateau_detected": False,
                "plateau_corpus_fraction": np.nan,
                "tokens_processed_at_plateau": np.nan,
                "observed_vocab_at_plateau": np.nan,
                "tokenizer_total_vocab": grp["total_tokenizer_vocab"].iloc[-1],
                "vocab_coverage_percent": np.nan,
                "criterion_used": criterion,
                "note": "No plateau detected within the sampled corpus.",
            })
            continue

        row = grp.iloc[hit_index]
        rows.append({
            "tokenizer": tok,
            "plateau_detected": True,
            "plateau_corpus_fraction": float(row["corpus_fraction"]),
            "tokens_processed_at_plateau": int(row["tokens_processed"]),
            "observed_vocab_at_plateau": int(row["unique_tokens_observed"]),
            "tokenizer_total_vocab": row["total_tokenizer_vocab"],
            "vocab_coverage_percent": float(row["vocab_coverage_percent"]),
            "criterion_used": criterion,
            "note": "",
        })

    return pd.DataFrame(rows)


# =============================================================================
# PART C -- PLOTS
# =============================================================================

def plot_vocab_growth(df: pd.DataFrame, language: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for tok, grp in df.groupby("tokenizer", sort=False):
        grp = grp.sort_values("tokens_processed")
        code = str(grp["tokenizer_code"].iloc[0])
        ax.plot(grp["tokens_processed"], grp["unique_tokens_observed"],
                marker=TOKENIZER_MARKERS.get(code, "o"),
                color=TOKENIZER_COLORS.get(code),
                linewidth=1.8, markersize=5, label=str(tok))

    ax.set_xlabel("Corpus size (tokens processed)")
    ax.set_ylabel("Unique token types observed")
    ax.set_title(f"Vocabulary growth vs corpus size — {language} Wikipedia\n"
                 f"(coverage of pre-trained tokenizers; no tokenizer was retrained)",
                 fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(title="Tokenizer", fontsize=9)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()

    path = output_dir / "vocab_growth.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


def plot_marginal_growth(df: pd.DataFrame, language: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for tok, grp in df.groupby("tokenizer", sort=False):
        grp = grp.sort_values("tokens_processed")
        code = str(grp["tokenizer_code"].iloc[0])
        ax.plot(grp["tokens_processed"], grp["marginal_growth"],
                marker=TOKENIZER_MARKERS.get(code, "o"),
                color=TOKENIZER_COLORS.get(code),
                linewidth=1.8, markersize=5, label=str(tok))

    ax.set_xlabel("Corpus size (tokens processed)")
    ax.set_ylabel("Marginal growth\n(new token types per additional token read)")
    ax.set_title(f"Marginal vocabulary growth — {language} Wikipedia", fontsize=11)
    ax.set_yscale("log")  # marginal growth falls by orders of magnitude
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(title="Tokenizer", fontsize=9)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()

    path = output_dir / "marginal_vocab_growth.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


# =============================================================================
# PART F -- AUTOMATIC INTERPRETATION
# =============================================================================

def interpret(growth: pd.DataFrame, plateau: pd.DataFrame, language: str) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append(f"AUTOMATIC INTERPRETATION — {language}")
    lines.append("=" * 100)

    # 1. Fastest initial discovery.
    firsts = (growth.sort_values("corpus_fraction")
                    .groupby("tokenizer", sort=False)
                    .first()
                    .reset_index())
    if firsts["marginal_growth"].notna().any():
        fastest = firsts.loc[firsts["marginal_growth"].idxmax()]
        lines.append(
            f"1. Fastest initial vocabulary discovery: {fastest['tokenizer']} "
            f"({fastest['marginal_growth']:.3e} new token types per token read over the "
            f"first {int(fastest['corpus_fraction'] * 100)}% of the corpus)."
        )
        ranked = firsts.sort_values("marginal_growth", ascending=False)
        order = ", ".join(f"{r['tokenizer']} ({r['marginal_growth']:.2e})"
                          for _, r in ranked.iterrows())
        lines.append(f"   Ranking at the first checkpoint: {order}")
    else:
        lines.append("1. Initial marginal growth could not be computed.")

    # 2. Earliest plateau.
    detected = plateau[plateau["plateau_detected"]]
    if detected.empty:
        lines.append(
            "2. No tokenizer reached the plateau criterion within the sampled corpus, "
            "so vocabulary discovery was still ongoing at 100% of the corpus."
        )
    else:
        earliest = detected.loc[detected["plateau_corpus_fraction"].idxmin()]
        lines.append(
            f"2. Earliest plateau: {earliest['tokenizer']} at "
            f"{earliest['plateau_corpus_fraction'] * 100:.0f}% of the corpus "
            f"(~{int(earliest['tokens_processed_at_plateau']):,} tokens, "
            f"{int(earliest['observed_vocab_at_plateau']):,} distinct types observed)."
        )
        not_detected = plateau[~plateau["plateau_detected"]]
        if not not_detected.empty:
            names = ", ".join(not_detected["tokenizer"].astype(str))
            lines.append(f"   No plateau detected within the sampled corpus for: {names}.")

    # 3. Coverage at full corpus.
    finals = (growth.sort_values("corpus_fraction")
                    .groupby("tokenizer", sort=False)
                    .last()
                    .reset_index())
    lines.append("3. Share of each tokenizer's total vocabulary actually used by this "
                 "corpus:")
    for _, row in finals.iterrows():
        total = row["total_tokenizer_vocab"]
        total_str = f"{int(total):,}" if pd.notna(total) else "unknown"
        cov = (f"{row['vocab_coverage_percent']:.1f}%"
               if pd.notna(row["vocab_coverage_percent"]) else "unknown")
        lines.append(
            f"   {str(row['tokenizer']):<6} {int(row['unique_tokens_observed']):>8,} of "
            f"{total_str:>10} types  ->  {cov} coverage"
        )
    if finals["vocab_coverage_percent"].notna().any():
        lo = finals.loc[finals["vocab_coverage_percent"].idxmin()]
        hi = finals.loc[finals["vocab_coverage_percent"].idxmax()]
        max_cov = float(hi["vocab_coverage_percent"])
        lines.append(
            f"   Highest coverage: {hi['tokenizer']} ({max_cov:.1f}%); "
            f"lowest: {lo['tokenizer']} ({lo['vocab_coverage_percent']:.1f}%)."
        )
        # Commentary follows the numbers rather than assuming an outcome.
        if max_cov < 50.0:
            lines.append(
                "   Every tokenizer leaves over half of its vocabulary untouched by this "
                "corpus, which is expected: these are multilingual vocabularies being "
                "exercised by a single language."
            )
        elif max_cov < 90.0:
            lines.append(
                "   A substantial share of each vocabulary is exercised, but none is "
                "close to fully covered, so corpus size — not vocabulary size — is still "
                "the binding constraint here."
            )
        else:
            lines.append(
                "   At least one tokenizer is close to full vocabulary coverage, so the "
                "corpus is large enough to exercise nearly its entire vocabulary and the "
                "growth curve is necessarily near its ceiling."
            )

    # 4. Sweet spot.
    if detected.empty:
        lines.append(
            "4. Suggested sweet spot: none can be supported by this data. Discovery of "
            "new token types had not slowed to the threshold anywhere in the sampled "
            "corpus, so a larger corpus would be needed before commenting on "
            "diminishing returns."
        )
    else:
        med = float(detected["tokens_processed_at_plateau"].median())
        lines.append(
            f"4. Suggested sweet spot: across the tokenizers that plateaued, vocabulary "
            f"discovery slows substantially after roughly {med:,.0f} tokens of {language} "
            f"text. Beyond that point, additional corpus yields comparatively few "
            f"previously-unseen token types."
        )

    lines.append("")
    lines.append("LIMITATION — please carry this into the report:")
    lines.append(
        "  This experiment measures how much of a FIXED, PRE-TRAINED tokenizer's "
        "vocabulary\n"
        "  a corpus actually exercises. No tokenizer was retrained at different "
        "vocabulary\n"
        "  sizes. The plateau therefore shows diminishing returns in observed vocabulary\n"
        "  COVERAGE for this corpus; it does NOT demonstrate an optimal tokenizer "
        "vocabulary\n"
        "  size, and the numbers above should not be read as one. Establishing an "
        "optimum\n"
        "  would require training tokenizers at several vocabulary sizes and comparing "
        "them\n"
        "  on a downstream measure such as compression rate or task performance."
    )
    lines.append("=" * 100)
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person 3: tokenizer comparison and vocabulary growth analysis."
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=sorted(LANGUAGE_NAMES),
                        help="Language code for the vocabulary growth study "
                             "(default: %(default)s).")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Use only the first N articles. Useful for a fast trial "
                             "run; omit to use the whole corpus.")
    parser.add_argument("--skip-growth", action="store_true",
                        help="Run Part A (tokenizer comparison) only and skip the "
                             "slow tokenization in Parts B-F.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
    # Keep the console readable: HF/httpx emit an INFO line per HTTP request.
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    repo_root = find_repo_root(Path(__file__).resolve().parent)
    logger.info("Repository root detected: %s", repo_root)

    output_dir = repo_root / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ---------------- PART A ----------------
    logger.info("--- PART A: tokenizer comparison ---")
    loaded: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []
    for name, model_id in TOKENIZERS.items():
        tok = load_tokenizer(name, model_id)
        loaded[name] = tok
        rows.append(describe_tokenizer(name, model_id, tok))

    comparison = pd.DataFrame(rows)
    comparison_path = output_dir / "tokenizer_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info("Tokenizer comparison written -> %s", comparison_path)

    print_table(comparison,
                ["tokenizer", "model_id", "vocab_size", "vocab_size_with_added",
                 "tokenizer_class", "backend", "algorithm"],
                "PART A — TOKENIZER COMPARISON", floatfmt=".0f")
    print("Special tokens")
    print("-" * 100)
    for _, row in comparison.iterrows():
        print(f"  {row['tokenizer']:<6}: {row['special_tokens']}")
    print()
    print("Notes")
    print("-" * 100)
    for _, row in comparison.iterrows():
        print(f"  {row['tokenizer']:<6}: {row['notes']}")
    print()

    usable = {n: t for n, t in loaded.items() if t is not None}
    if not usable:
        logger.error("No tokenizers could be loaded. Parts B-F cannot run.")
        logger.error("Fix tokenizer loading (network / sentencepiece / gated repos) "
                     "and re-run.")
        return 1
    if len(usable) < len(TOKENIZERS):
        missing = sorted(set(TOKENIZERS) - set(usable))
        logger.warning("Continuing with %d of %d tokenizers. Missing: %s",
                       len(usable), len(TOKENIZERS), ", ".join(missing))

    if args.skip_growth:
        logger.info("--skip-growth set: stopping after Part A.")
        return 0

    # ---------------- PARTS B-F ----------------
    lang_code = args.language
    language = LANGUAGE_NAMES[lang_code]
    logger.info("--- PARTS B-F: vocabulary growth for %s ---", language)

    raw_dir = find_dir(repo_root, RAW_TEXT_DIR_CANDIDATES, "raw text")
    if raw_dir is None:
        logger.error("Cannot run the vocabulary growth study without the raw corpus.")
        return 1

    try:
        lines = load_corpus_lines(raw_dir, lang_code, args.max_lines)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    all_records: List[Dict[str, object]] = []
    for name, tok in usable.items():
        logger.info("Measuring vocabulary growth with %s ...", name)
        try:
            all_records.extend(measure_vocab_growth(lines, tok, name, CORPUS_FRACTIONS))
        except Exception as exc:
            logger.error("FAILED vocabulary growth for %s: %s: %s",
                         name, type(exc).__name__, exc)

    if not all_records:
        logger.error("No vocabulary growth data was produced.")
        return 1

    growth = pd.DataFrame(all_records)
    growth = add_vocab_metadata(growth, comparison, language)
    growth = add_marginal_growth(growth)

    growth_columns = [
        "language", "tokenizer", "corpus_fraction", "lines_processed",
        "characters_processed", "tokens_processed", "unique_tokens_observed",
        "total_tokenizer_vocab", "vocab_coverage_percent",
        "new_unique_tokens", "additional_tokens_processed",
        "marginal_growth", "marginal_growth_relative", "initial_marginal_growth",
    ]
    growth_path = output_dir / "vocab_growth_results.csv"
    growth[growth_columns].to_csv(growth_path, index=False)
    logger.info("Vocabulary growth CSV written -> %s", growth_path)

    print_table(growth,
                ["tokenizer", "corpus_fraction", "tokens_processed",
                 "unique_tokens_observed", "vocab_coverage_percent",
                 "marginal_growth", "marginal_growth_relative"],
                f"PARTS B & D — VOCABULARY GROWTH ({language})", floatfmt=".6g")

    plot_vocab_growth(growth, language, output_dir)
    plot_marginal_growth(growth, language, output_dir)

    plateau = estimate_plateau(growth)
    plateau_path = output_dir / "sweet_spot_estimates.csv"
    plateau.to_csv(plateau_path, index=False)
    logger.info("Sweet spot estimates written -> %s", plateau_path)

    print_table(plateau,
                ["tokenizer", "plateau_detected", "plateau_corpus_fraction",
                 "tokens_processed_at_plateau", "observed_vocab_at_plateau",
                 "tokenizer_total_vocab", "vocab_coverage_percent"],
                "PART E — PLATEAU / SWEET SPOT ESTIMATES", floatfmt=".6g")
    print(f"  Criterion: {plateau['criterion_used'].iloc[0]}")
    for _, row in plateau.iterrows():
        if row["note"]:
            print(f"  {row['tokenizer']}: {row['note']}")
    print()

    print(interpret(growth, plateau, language))
    print()

    logger.info("Done. Outputs in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
