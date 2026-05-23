import hashlib
import pandas as pd
from typing import Callable


def compute_k_anonymity(df: pd.DataFrame, qi_columns: list[str]) -> dict:
    """Calcola k-anonymity sul df rispetto alle QI specificate.

    Returns dict con: k_effective (min), distribution (Series con conteggi
    per gruppo QI), n_unique_combinations, share_singletons.
    """
    group_sizes = df.groupby(qi_columns, dropna=False).size()
    return {
        "k_effective": int(group_sizes.min()),
        "distribution": group_sizes,
        "n_unique_combinations": len(group_sizes),
        "share_singletons": float((group_sizes == 1).mean()),
    }


def compute_l_diversity(
    df: pd.DataFrame,
    qi_columns: list[str],
    sensitive_column: str,
) -> dict:
    """Calcola l-diversity: # valori distinti dell'attributo sensibile per ogni gruppo QI."""
    l_per_group = df.groupby(qi_columns, dropna=False)[sensitive_column].nunique(dropna=False)
    return {
        "l_effective": int(l_per_group.min()),
        "distribution": l_per_group,
        "n_groups": len(l_per_group),
        "share_l_geq_2": float((l_per_group >= 2).mean()),
        "share_l_geq_3": float((l_per_group >= 3).mean()),
    }


def generalize_quasi_identifiers(
    df: pd.DataFrame,
    strategy: dict[str, Callable | dict],
) -> pd.DataFrame:
    """Applica generalizzazione alle colonne specificate.

    strategy: dict {colonna: mapping}. Mapping può essere un dict {valore_nativo: valore_generalizzato}
    o una funzione callable(Series) -> Series.
    """
    df_out = df.copy()
    for col, mapping in strategy.items():
        if callable(mapping):
            df_out[col] = mapping(df_out[col])
        else:
            df_out[col] = df_out[col].map(mapping)
    return df_out


def generalize_multilabel_to_3way(
    df: pd.DataFrame,
    prefix: str,
    primary_categories: list[str],
    other_label: str = "other_or_mixed",
) -> pd.Series:
    """Aggrega dummy multi-label in (len(primary_categories)+1) categorie esclusive.

    Logica: la riga riceve la categoria primary se ESCLUSIVAMENTE quella dummy è attiva
    (no multi-label). Tutto il resto va in other_label.

    Usata per race, sexuality, religion, e in generale per ogni gruppo di dummy
    binarie multi-label che vogliamo collassare in colonna categorica singola.
    """
    all_dummies = [c for c in df.columns if c.startswith(prefix)]
    if not all_dummies:
        raise ValueError(f"Nessuna colonna trovata con prefisso '{prefix}'")
    
    n_active = df[all_dummies].sum(axis=1)
    result = pd.Series(other_label, index=df.index, dtype=object)
    
    for cat in primary_categories:
        col = f"{prefix}{cat}"
        if col not in df.columns:
            available = [c.replace(prefix, "") for c in all_dummies]
            raise ValueError(
                f"Colonna '{col}' non trovata. Disponibili: {available}"
            )
        mask = (df[col] == 1) & (n_active == 1)
        result[mask] = cat
    
    return result


def collapse_binary(
    series: pd.Series,
    majority_value: str,
    majority_label: str = None,
    minority_label: str = "non_" + "{majority}",
) -> pd.Series:
    """Collassa una colonna categorica in 2 valori: majority_value vs tutto-il-resto."""
    maj = majority_label or majority_value
    minor = minority_label.format(majority=majority_value) if "{majority}" in minority_label else minority_label
    return series.apply(lambda x: maj if x == majority_value else minor)


def suppress_singletons(
    df: pd.DataFrame,
    qi_columns: list[str],
    k_threshold: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Sopprime righe in cui la combinazione QI compare meno di k_threshold volte.

    Returns (df_suppressed, stats) con metriche sul costo della suppression.
    """
    group_sizes = df.groupby(qi_columns, dropna=False).size()
    valid_groups = group_sizes[group_sizes >= k_threshold].index
    mask = df.set_index(qi_columns).index.isin(valid_groups)
    df_suppressed = df[mask].copy().reset_index(drop=True)
    
    stats = {
        "n_removed": int((~mask).sum()),
        "share_removed": float((~mask).mean()),
        "n_groups_removed": int((group_sizes < k_threshold).sum()),
        "k_min_post": int(df_suppressed.groupby(qi_columns, dropna=False).size().min())
                      if len(df_suppressed) > 0 else 0,
    }
    return df_suppressed, stats



GENDER_BAND = {
    "male": "male",
    "female": "female",
}

TRANS_BAND = {
    "yes": "yes",
    "no": "no_or_unknown",
    "prefer_not_to_say": "no_or_unknown",
}

def _age_band(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[-1, 29.999, 49.999, 200],
        labels=["under-30", "30-50", "over-50"],
    ).astype(object)

EDUC_BAND = {
    "some_high_school":     "high_school_or_lower",
    "high_school_grad":     "high_school_or_lower",
    "some_college":         "some_college",
    "college_grad_aa":      "some_college",
    "college_grad_ba":      "some_college",
    "masters":              "masters_or_higher",
    "phd":                  "masters_or_higher",
    "professional_degree":  "masters_or_higher",
}

INCOME_BAND = {
    "<10k":      "low",
    "10k-50k":   "low",
    "50k-100k":  "mid",
    "100k-200k": "high",
    ">200k":     "high",
}

IDEOLOGY_BAND = {
    "extremely_conservative": "conservative",
    "conservative":           "conservative",
    "slightly_conservative":  "conservative",
    "neutral":                "moderate_or_no_opinion",
    "no_opinion":             "moderate_or_no_opinion",
    "slightly_liberal":       "liberal",
    "liberal":                "liberal",
    "extremely_liberal":      "liberal",
}

strategy = {
    "annotator_gender":   lambda s: s.map(GENDER_BAND).fillna("other_or_missing"),
    "annotator_trans":    TRANS_BAND,
    "annotator_age":      _age_band,
    "annotator_educ":     EDUC_BAND,
    "annotator_income":   INCOME_BAND,
    "annotator_ideology": IDEOLOGY_BAND,
}

