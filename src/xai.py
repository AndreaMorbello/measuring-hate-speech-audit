"""Funzioni per interpretabilita' e mitigazione avanzata del Cap. 5."""

import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt

def top_predictive_features(
    model,
    vectorizer,
    n: int = 30,
    direction: str = "both",
) -> pd.DataFrame:
    """Restituisce le n feature con coefficiente piu' alto (e/o piu' basso).
    
    Parameters
    ----------
    model : LogisticRegression addestrato.
    vectorizer : TfidfVectorizer addestrato (per recuperare i nomi delle feature).
    n : numero di feature per direzione.
    direction : "positive" (verso hate), "negative" (verso non-hate), o "both".
    
    Returns
    -------
    DataFrame con colonne: feature, coefficient, direction.
    """
    coefs = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    
    rows = []
    if direction in ("positive", "both"):
        top_pos_idx = np.argsort(coefs)[-n:][::-1]
        for idx in top_pos_idx:
            rows.append({
                "feature": feature_names[idx],
                "coefficient": coefs[idx],
                "direction": "→ hate",
            })
    
    if direction in ("negative", "both"):
        top_neg_idx = np.argsort(coefs)[:n]
        for idx in top_neg_idx:
            rows.append({
                "feature": feature_names[idx],
                "coefficient": coefs[idx],
                "direction": "→ non-hate",
            })
    
    return pd.DataFrame(rows)


def compare_feature_importance(
    model_a, model_b, vectorizer, n: int = 30,
) -> pd.DataFrame:
    """Confronta i coefficienti di due modelli. Restituisce un df con le
    feature top-n di entrambi i modelli, con il delta coefficient.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs_a = model_a.coef_[0]
    coefs_b = model_b.coef_[0]
    
    top_features_set = set()
    top_features_set.update(feature_names[np.argsort(coefs_a)[-n:]])
    top_features_set.update(feature_names[np.argsort(coefs_b)[-n:]])
    
    rows = []
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    for feat in top_features_set:
        idx = feature_to_idx[feat]
        rows.append({
            "feature": feat,
            "coef_baseline": coefs_a[idx],
            "coef_smote": coefs_b[idx],
            "delta": coefs_b[idx] - coefs_a[idx],
        })
    
    return pd.DataFrame(rows).sort_values("delta", ascending=False)

def select_erasure_cases(
    X_test_texts: np.ndarray,
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_proba_baseline: np.ndarray,
    y_pred_smote: np.ndarray,
    y_proba_smote: np.ndarray,
    target_test: np.ndarray,
    underrepresented_groups: list = None,
) -> dict:
    """Seleziona algoritmicamente 6 casi emblematici per analisi Erasure.
    Restituisce dict {case_name: idx_in_test_set}.
    """
    if underrepresented_groups is None:
        underrepresented_groups = ["disability", "religion", "age"]
    
    cases = {}
    
    # 1. Vero positivo ad alta confidenza (smote)
    tp_mask = (y_true == 1) & (y_pred_smote == 1)
    if tp_mask.any():
        candidates = np.where(tp_mask)[0]
        cases["TP_high_conf"] = candidates[np.argmax(y_proba_smote[candidates])]
    
    # 2. Vero negativo ad alta confidenza (smote)
    tn_mask = (y_true == 0) & (y_pred_smote == 0)
    if tn_mask.any():
        candidates = np.where(tn_mask)[0]
        cases["TN_high_conf"] = candidates[np.argmin(y_proba_smote[candidates])]
    
    # 3. Falso positivo su gruppo sotto-rappresentato (smote)
    underrep_mask = np.isin(target_test, underrepresented_groups)
    fp_underrep_mask = (y_true == 0) & (y_pred_smote == 1) & underrep_mask
    if fp_underrep_mask.any():
        candidates = np.where(fp_underrep_mask)[0]
        cases["FP_underrepresented"] = candidates[np.argmax(y_proba_smote[candidates])]
    
    # 4. Falso positivo amplificato dal SMOTE (baseline corretto, smote sbaglia)
    fp_amplified_mask = (y_true == 0) & (y_pred_baseline == 0) & (y_pred_smote == 1)
    if fp_amplified_mask.any():
        candidates = np.where(fp_amplified_mask)[0]
        delta = y_proba_smote[candidates] - y_proba_baseline[candidates]
        cases["FP_smote_amplified"] = candidates[np.argmax(delta)]
    
    # 5. Falso negativo (smote)
    fn_mask = (y_true == 1) & (y_pred_smote == 0)
    if fn_mask.any():
        candidates = np.where(fn_mask)[0]
        cases["FN_high_uncertainty"] = candidates[np.argmin(np.abs(y_proba_smote[candidates] - 0.5))]
    
    # 6. Disagreement tra modelli
    disagreement_mask = np.abs(y_proba_baseline - y_proba_smote) > 0.3
    if disagreement_mask.any():
        candidates = np.where(disagreement_mask)[0]
        cases["DISAGREEMENT"] = candidates[np.argmax(
            np.abs(y_proba_baseline[candidates] - y_proba_smote[candidates])
        )]
    
    return cases


def erasure_attribution(
    text: str,
    model,
    vectorizer,
) -> pd.DataFrame:
    """Calcola Erasure-based attribution per ciascun token di un testo.
    
    Per ogni token, lo rimuove dal testo e misura il delta di probabilita'
    predetta. Token con delta positivo grande stavano sostenendo la classe
    positiva (hate).
    """
    tokens = text.split()
    
    # Predizione originale
    X_orig = vectorizer.transform([text])
    p_orig = model.predict_proba(X_orig)[0, 1]
    
    rows = []
    for i, token in enumerate(tokens):
        text_ablated = " ".join(tokens[:i] + tokens[i+1:])
        X_ablated = vectorizer.transform([text_ablated])
        p_ablated = model.predict_proba(X_ablated)[0, 1]
        
        rows.append({
            "position": i,
            "token": token,
            "p_original": p_orig,
            "p_ablated": p_ablated,
            "delta": p_orig - p_ablated,
        })
    
    return pd.DataFrame(rows)


def visualize_erasure(
    df_attribution: pd.DataFrame,
    title: str = "",
    figsize: tuple = (12, 2),
):
    """Visualizza Erasure attribution come heatmap inline sul testo."""
    fig, ax = plt.subplots(figsize=figsize)
    
    tokens = df_attribution["token"].values
    deltas = df_attribution["delta"].values
    
    max_abs = np.abs(deltas).max() if len(deltas) > 0 else 1
    colors = plt.cm.RdBu_r(0.5 + 0.5 * deltas / max_abs)
    
    ax.bar(range(len(tokens)), deltas, color=colors, edgecolor="white")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("delta (P originale − P senza token)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def summarize_models(metrics_dfs: dict, fairness_dfs: dict) -> pd.DataFrame:
    """Aggrega metriche aggregate dei modelli per confronto sintetico."""
    rows = []
    for name, m_df in metrics_dfs.items():
        f_df = fairness_dfs[name]
        rows.append({
            "modello": name,
            "f1_medio_per_gruppo": m_df["f1"].mean(),
            "recall_medio_per_gruppo": m_df["recall"].mean(),
            "n_gruppi_passa_80pct": int(f_df["passes_80pct_rule"].sum()),
            "EO_diff_assoluta_media": f_df["equal_opportunity_diff"].abs().mean(),
            "DI_distanza_media_da_1": (f_df["disparate_impact"] - 1).abs().mean(),
        })
    return pd.DataFrame(rows)