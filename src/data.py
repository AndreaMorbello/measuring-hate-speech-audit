"""Data loading and inspection utilities for the Measuring Hate Speech audit.

Wraps the HuggingFace dataset ucberkeley-dlab/measuring-hate-speech with
local parquet caching and structured column-group discovery.
"""

from __future__ import annotations

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HF_DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
HF_CONFIG = "default"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Raggruppa le colonne del dataset Berkeley Measuring Hate Speech in
    categorie etiche, utili per l'audit di privacy, bias e fairness.

    La categorizzazione segue Kennedy et al. (2020) e isola nove gruppi:
    identificatori diretti, testo, metadata, etichette di outcome, metriche
    psicometriche del modello Rasch, gruppi target nel testo, quasi-
    identificatori degli annotatori (summary e relativi dummy), piu' un
    catchall difensivo ``other`` per colonne future non classificabili.

    Args:
        df: DataFrame del dataset Measuring Hate Speech.

    Returns:
        Dizionario con 9 chiavi, ciascuna mappata a una lista di nomi colonna.
        Tutte le entry hardcoded sono filtrate per esistenza nel df, quindi
        un gruppo puo' risultare vuoto se le colonne attese sono assenti.

    Example:
        >>> groups = get_column_groups(df)
        >>> len(groups['annotator_qi_summary'])
        6
        >>> groups['other']  # vuoto se la categorizzazione e' completa
        []
    """
    cols: set = set(df.columns)

    # Filtro difensivo: ogni hardcoded entry viene inclusa solo se presente
    # nel df. Se HF rinomina/rimuove una colonna, il gruppo non contiene
    # nomi fantasma e il downstream code non si rompe con KeyError.
    direct_identifiers: list = [c for c in ["comment_id", "annotator_id"] if c in cols]

    text: list = [c for c in ["text"] if c in cols]

    metadata: list = [c for c in ["platform"] if c in cols]

    outcomes: list = [
        c
        for c in [
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
            "hate_speech_score",
        ]
        if c in cols
    ]

    annotation_quality = [
        c
        for c in [
            "infitms",
            "outfitms",
            "std_err",
            "hypothesis",
            "annotator_severity",
            "annotator_infitms",
            "annotator_outfitms",
        ]
        if c in cols
    ]

    target_groups = [c for c in df.columns if c.startswith("target_")]

    annotator_qi_summary = [
        c
        for c in [
            "annotator_gender",
            "annotator_trans",
            "annotator_educ",
            "annotator_income",
            "annotator_ideology",
            "annotator_age",
        ]
        if c in cols
    ]

    classified = set(
        direct_identifiers
        + text
        + metadata
        + outcomes
        + annotation_quality
        + target_groups
        + annotator_qi_summary
    )
    annotator_qi_dummies = [
        c for c in df.columns if c.startswith("annotator_") and c not in classified
    ]

    # Catchall difensivo: cattura eventuali colonne aggiunte in futuro da HF
    # che non rientrano in nessuna delle regole sopra. Idealmente vuoto.
    classified |= set(annotator_qi_dummies)
    other = [c for c in df.columns if c not in classified]

    return {
        "direct_identifiers": direct_identifiers,
        "text": text,
        "metadata": metadata,
        "outcomes": outcomes,
        "annotation_quality": annotation_quality,
        "target_groups": target_groups,
        "annotator_qi_summary": annotator_qi_summary,
        "annotator_qi_dummies": annotator_qi_dummies,
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


# ---------------------------------------------------------------------------
# EDA utilities (estensione §2.4-bis, §2.6-bis, §2.7-bis del notebook)
# ---------------------------------------------------------------------------

# Mappa documentata dei codici numerici di `platform` ai nomi leggibili.
# Riferimento: Kennedy et al. (2020), supplementary material.
PLATFORM_NAMES: dict[int, str] = {0: "YouTube", 1: "Reddit", 2: "Twitter"}


def platform_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Conta annotazioni per piattaforma con percentuale.

    Args:
        df: dataset Measuring Hate Speech.

    Returns:
        DataFrame indicizzato per nome piattaforma, con colonne ``count``
        e ``pct`` (percentuale sul totale annotazioni, due decimali).
    """
    counts = df["platform"].value_counts().sort_index()
    counts.index = [PLATFORM_NAMES.get(int(i), f"unknown ({i})") for i in counts.index]
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct}).sort_values("count", ascending=False)


def label_correlation_matrix(
    df: pd.DataFrame,
    labels: list[str] | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Matrice di correlazione tra le etichette ordinali.

    Spearman e' la scelta naturale per dati ordinali (Likert 0-4): non
    assume linearita' ne' normalita', solo monotonicita' tra ranghi.

    Args:
        df: dataset Measuring Hate Speech.
        labels: lista delle colonne ordinali. Se ``None``, usa le 10
            etichette standard escludendo ``hate_speech_score``.
        method: passato a ``pd.DataFrame.corr`` (default ``"spearman"``).

    Returns:
        Matrice di correlazione quadrata di dimensione ``len(labels)``.
    """
    if labels is None:
        labels = [
            "sentiment", "respect", "insult", "humiliate", "status",
            "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
        ]
    return df[labels].corr(method=method)


def gini_coefficient(values: np.ndarray | pd.Series | list) -> float:
    """Indice di Gini su una distribuzione non-negativa.

    Implementazione manuale via formula chiusa sulla serie ordinata:

        G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))

    Restituisce 0.0 per distribuzioni perfettamente uniformi, valori
    prossimi a 1.0 per concentrazioni estreme. Definito 0.0 anche per
    serie vuote o tutte nulle (caso degenere).

    Args:
        values: array di valori non-negativi.

    Returns:
        Indice di Gini in [0, 1].
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    total = arr.sum()
    if n == 0 or total == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2 * np.sum(indices * arr) - (n + 1) * total) / (n * total))


def annotator_productivity(df: pd.DataFrame) -> dict:
    """Statistiche sulla concentrazione della produttivita' degli annotatori.

    Args:
        df: dataset Measuring Hate Speech.

    Returns:
        Dizionario con:
        - ``counts`` (pd.Series): annotazioni per ``annotator_id``.
        - ``mean``, ``median``, ``max`` (float/int): statistiche aggregate.
        - ``gini`` (float): indice di Gini della distribuzione.
        - ``top1pct_share`` (float): quota di annotazioni prodotte
          dall'1% piu' produttivo del pool.
    """
    counts = df.groupby("annotator_id").size().sort_values(ascending=False)
    n = len(counts)
    top1pct_n = max(1, int(np.ceil(n * 0.01)))
    top1pct_share = float(counts.iloc[:top1pct_n].sum() / counts.sum())
    return {
        "counts": counts,
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "max": int(counts.max()),
        "gini": gini_coefficient(counts.values),
        "top1pct_share": top1pct_share,
    }


def comment_score_dispersion(df: pd.DataFrame) -> pd.Series:
    """Deviazione standard di ``hate_speech_score`` per commento.

    Misura il disaccordo tra annotatori sullo stesso commento. Std
    bassa = consenso (commento "pacificamente" hateful o non-hateful);
    std alta = commento controverso. Commenti con un solo annotatore
    producono NaN (per definizione: la std di un elemento e' indefinita).

    Args:
        df: dataset Measuring Hate Speech.

    Returns:
        Serie indicizzata per ``comment_id`` con la std dei punteggi.
        Le entry NaN (commenti con un solo annotatore) sono rimosse.
    """
    return df.groupby("comment_id")["hate_speech_score"].std().dropna()
