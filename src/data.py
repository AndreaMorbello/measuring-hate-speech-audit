"""Data loading and inspection utilities for the Measuring Hate Speech audit.

Wraps the HuggingFace dataset ucberkeley-dlab/measuring-hate-speech with
local parquet caching and structured column-group discovery.
"""

from __future__ import annotations

# Standard library
from pathlib import Path

# Third-party
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HF_DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
HF_CONFIG = "default"

# The 10 ordinal annotation labels defined in Kennedy et al. (2020).
# These do not share a common column prefix, so they are enumerated explicitly.
_ANNOTATION_LABEL_COLS: list[str] = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hatespeech",
]

_DIRECT_IDENTIFIER_COLS: list[str] = ["comment_id", "annotator_id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the DataFrame columns into semantically meaningful groups.

    Groups are derived using prefix matching where columns share a naming
    convention (``annotator_*``, ``target_*``), and by explicit enumeration
    for identifiers and annotation labels that have no shared prefix.

    Args:
        df: The raw dataset DataFrame as returned by ``load_raw_dataset``.

    Returns:
        A dictionary with the following keys:

        - ``direct_identifiers``: row-level and annotator-level IDs.
        - ``annotator_quasi_identifiers``: demographic attributes of annotators
          (all ``annotator_*`` columns except ``annotator_id``).
        - ``target_groups``: attributes of the targeted social group in the text
          (all ``target_*`` columns).
        - ``annotation_labels``: the 10 ordinal labels from the codebook.
        - ``outcome``: the continuous IRT-aggregated hate-speech score.
        - ``text``: the raw comment text column.
        - ``other``: any columns not matched by the above rules (defensive catch-all).

    Example:
        >>> groups = get_column_groups(df)
        >>> groups["outcome"]
        ['hate_speech_score']
    """
    cols = set(df.columns)

    direct_identifiers = [c for c in _DIRECT_IDENTIFIER_COLS if c in cols]

    annotator_quasi_identifiers = [
        c for c in df.columns if c.startswith("annotator_") and c != "annotator_id"
    ]

    target_groups = [c for c in df.columns if c.startswith("target_")]

    annotation_labels = [c for c in _ANNOTATION_LABEL_COLS if c in cols]

    outcome = ["hate_speech_score"] if "hate_speech_score" in cols else []

    text = ["text"] if "text" in cols else []

    classified = (
        set(direct_identifiers)
        | set(annotator_quasi_identifiers)
        | set(target_groups)
        | set(annotation_labels)
        | set(outcome)
        | set(text)
    )
    other = [c for c in df.columns if c not in classified]

    return {
        "direct_identifiers": direct_identifiers,
        "annotator_quasi_identifiers": annotator_quasi_identifiers,
        "target_groups": target_groups,
        "annotation_labels": annotation_labels,
        "outcome": outcome,
        "text": text,
        "other": other,
    }


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Compute basic descriptive statistics for the raw dataset.

    Args:
        df: The raw dataset DataFrame as returned by ``load_raw_dataset``.

    Returns:
        A dictionary containing:

        - ``n_rows`` (int): total number of annotation rows.
        - ``n_columns`` (int): total number of columns.
        - ``n_unique_comments`` (int): distinct comments in the corpus.
        - ``n_unique_annotators`` (int): distinct annotators.
        - ``annotations_per_comment_mean`` (float): mean annotations per comment.
        - ``annotations_per_comment_median`` (float): median annotations per comment.
        - ``hate_speech_score_stats`` (dict): descriptive stats for the outcome
          variable: ``mean``, ``std``, ``min``, ``max``, ``q25``, ``q75``.
        - ``missing_values`` (dict[str, float]): columns with at least one missing
          value mapped to their missing-value percentage (0–100).

    Example:
        >>> summary = summarize_dataset(df)
        >>> summary["n_unique_comments"]
        39565
    """
    annotations_per_comment = df.groupby("comment_id").size()

    hs = df["hate_speech_score"]
    hs_stats = {
        "mean": float(hs.mean()),
        "std": float(hs.std()),
        "min": float(hs.min()),
        "max": float(hs.max()),
        "q25": float(hs.quantile(0.25)),
        "q75": float(hs.quantile(0.75)),
    }

    missing_pct = (df.isnull().mean() * 100).round(4)
    missing_values = {col: float(pct) for col, pct in missing_pct.items() if pct > 0}

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_unique_comments": int(df["comment_id"].nunique()),
        "n_unique_annotators": int(df["annotator_id"].nunique()),
        "annotations_per_comment_mean": float(annotations_per_comment.mean()),
        "annotations_per_comment_median": float(annotations_per_comment.median()),
        "hate_speech_score_stats": hs_stats,
        "missing_values": missing_values,
    }
