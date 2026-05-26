"""
Funzioni per l'audit algoritmico del Cap. 4 e mitigazione del bias.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


# ---------------------------------------------------------------------------
# §4.1 — Asimmetria di giudizio annotatore × target
# ---------------------------------------------------------------------------

def compute_annotator_target_asymmetry(
    df_annotation: pd.DataFrame,
    qi_target_pairs: list[tuple],
    score_col: str = "hate_speech_score",
) -> pd.DataFrame:
    """Per ogni coppia (QI annotatore, gruppo target), confronta lo score medio
    assegnato dagli annotatori in-group vs out-group sui commenti che bersagliano
    quel gruppo. Restituisce un df con effect size Cohen's d.

    Parameters
    ----------
    df_annotation : DataFrame a livello di annotazione (135k righe), con QI
        annotatori e dummies target del commento.
    qi_target_pairs : lista di tuple (qi_col, target_dim, mappings) dove
        mappings è un dict {qi_value: specific_target_col_or_None}. Se
        specific_target_col è None, usa target_dim come colonna binaria.
    score_col : nome della colonna con lo score da confrontare.

    Returns
    -------
    DataFrame con colonne: qi, qi_value, target, score_in_group, score_out_group,
    delta, cohen_d, n_in, n_out.
    """
    results = []
    for qi_col, target_dim, mappings in qi_target_pairs:
        for qi_value, specific_target in mappings.items():
            target_col = specific_target if specific_target else target_dim
            if target_col not in df_annotation.columns:
                continue
            targeted_mask = df_annotation[target_col] == True
            in_group = df_annotation[qi_col] == qi_value

            score_in = df_annotation[targeted_mask & in_group][score_col].mean()
            score_out = df_annotation[targeted_mask & ~in_group][score_col].mean()
            n_in = (targeted_mask & in_group).sum()
            n_out = (targeted_mask & ~in_group).sum()

            std_in = df_annotation[targeted_mask & in_group][score_col].std()
            std_out = df_annotation[targeted_mask & ~in_group][score_col].std()
            pooled_std = np.sqrt(
                ((n_in - 1) * std_in ** 2 + (n_out - 1) * std_out ** 2)
                / (n_in + n_out - 2)
            ) if (n_in + n_out - 2) > 0 else np.nan
            cohen_d = (score_in - score_out) / pooled_std if pooled_std and pooled_std > 0 else np.nan

            results.append({
                "qi": qi_col.replace("annotator_", ""),
                "qi_value": qi_value,
                "target": target_col.replace("target_", ""),
                "score_in_group": score_in,
                "score_out_group": score_out,
                "delta": score_in - score_out,
                "cohen_d": cohen_d,
                "n_in": int(n_in),
                "n_out": int(n_out),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# §4.2.1 — Rappresentatività dei gruppi target
# ---------------------------------------------------------------------------

def compute_target_representation(
    df: pd.DataFrame,
    target_dims: list[str],
    label_col: str = "hate_binary",
    score_col: str = "hate_score_mean",
    target_any_col: str = "target_any",
) -> pd.DataFrame:
    """Per ciascuna macro-dimensione di gruppo target, calcola copertura nel
    dataset, tasso di hate, media e std del giudizio. Include una riga finale
    per i commenti non-targeted come baseline di riferimento.
    """
    stats = []
    for dim in target_dims:
        targeted = df[df[dim]]
        n_total = int(df[dim].sum())
        n_hate = int(targeted[label_col].sum())
        stats.append({
            "dimension": dim.replace("target_", ""),
            "n_comments": n_total,
            "share_dataset": n_total / len(df),
            "hate_rate": n_hate / n_total if n_total > 0 else 0,
            "score_mean": targeted[score_col].mean(),
            "score_std": targeted[score_col].std(),
        })

    non_targeted = df[~df[target_any_col]]
    stats.append({
        "dimension": "(non-targeted)",
        "n_comments": len(non_targeted),
        "share_dataset": len(non_targeted) / len(df),
        "hate_rate": non_targeted[label_col].mean(),
        "score_mean": non_targeted[score_col].mean(),
        "score_std": non_targeted[score_col].std(),
    })

    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# §4.2.3 — Reweighing pre-processing
# ---------------------------------------------------------------------------

def primary_target(row: pd.Series, target_dims: list[str]) -> str:
    """Restituisce la prima macro-dimensione target attiva nel commento,
    o 'none' se il commento non bersaglia nessun gruppo protetto.
    """
    for dim in target_dims:
        if row[dim]:
            return dim.replace("target_", "")
    return "none"


# ---------------------------------------------------------------------------
# §4.4.1 — Performance disaggregate per gruppo
# ---------------------------------------------------------------------------

def per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    groups: np.ndarray,
    group_values: list = None,
    min_support: int = 10,
) -> pd.DataFrame:
    """Per ciascun gruppo, calcola precision/recall/F1/AUC e supporto."""
    if group_values is None:
        group_values = np.unique(groups)

    results = []
    for g in group_values:
        mask = groups == g
        if mask.sum() < min_support:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true[mask], y_proba[mask])
        except ValueError:
            auc = np.nan
        results.append({
            "group": g,
            "n_test": int(mask.sum()),
            "n_positive": int(y_true[mask].sum()),
            "precision": p,
            "recall": r,
            "f1": f,
            "roc_auc": auc,
            "predicted_positive_rate": y_pred[mask].mean(),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# §4.4.2 — Metriche di fairness
# ---------------------------------------------------------------------------

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    reference_group: str = "none",
    min_support: int = 10,
) -> pd.DataFrame:
    """Demographic parity, equal opportunity, disparate impact rispetto al
    gruppo di riferimento.

    Returns
    -------
    DataFrame con colonne: group, demographic_parity, equal_opportunity_diff,
    disparate_impact, passes_80pct_rule.
    """
    ref_mask = groups == reference_group
    if ref_mask.sum() == 0:
        raise ValueError(f"Reference group '{reference_group}' non trovato")

    p_pos_ref = y_pred[ref_mask].mean()
    tpr_ref = (
        y_pred[ref_mask & (y_true == 1)].mean()
        if (ref_mask & (y_true == 1)).sum() > 0 else 0
    )

    results = []
    for g in np.unique(groups):
        if g == reference_group:
            continue
        mask = groups == g
        if mask.sum() < min_support:
            continue
        p_pos = y_pred[mask].mean()
        tpr = (
            y_pred[mask & (y_true == 1)].mean()
            if (mask & (y_true == 1)).sum() > 0 else 0
        )
        di = p_pos / p_pos_ref if p_pos_ref > 0 else np.nan

        results.append({
            "group": g,
            "demographic_parity": di,
            "equal_opportunity_diff": tpr - tpr_ref,
            "disparate_impact": di,
            "passes_80pct_rule": 0.8 <= di <= 1.25 if not np.isnan(di) else False,
        })

    return pd.DataFrame(results)