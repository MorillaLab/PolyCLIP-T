import numpy as np
import pandas as pd


def compute_centroid(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        raise ValueError("Empty group: cannot compute centroid.")
    return X.mean(axis=0)


def build_numeric_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    benign_labels=(0, 1),
    low_labels=(2, 3),
    high_labels=(4, 5),
):
    """
    Build prototypes from numeric labels.

    Parameters
    ----------
    X : np.ndarray
        Embeddings of shape (N, D)
    y : np.ndarray
        Labels of shape (N,)
    benign_labels : tuple
        Labels used for benign prototype
    low_labels : tuple
        Labels used for low-penetrance prototype
    high_labels : tuple
        Labels used for high-penetrance prototype
    """
    y = np.asarray(y)

    protos = {}

    m_benign = np.isin(y, benign_labels)
    m_low = np.isin(y, low_labels)
    m_high = np.isin(y, high_labels)

    if m_benign.any():
        protos["benign"] = compute_centroid(X[m_benign])

    if m_low.any():
        protos["low"] = compute_centroid(X[m_low])

    if m_high.any():
        protos["high"] = compute_centroid(X[m_high])

    return protos


def mixed_norm_score(X: np.ndarray, ref: np.ndarray, alpha: float = 0.7, beta: float = 0.3):
    """
    Mixed norm score:
        alpha * ||x||_2 + beta * d(x, ref)

    Here d is approximated by Euclidean distance.
    """
    X = np.asarray(X, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    l2_term = np.linalg.norm(X, axis=1)
    dist_term = np.linalg.norm(X - ref[None, :], axis=1)

    return alpha * l2_term + beta * dist_term


def compute_variant_scores(
    X: np.ndarray,
    prototypes: dict,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> pd.DataFrame:
    out = {}

    if "benign" in prototypes:
        out["S_benign"] = mixed_norm_score(X, prototypes["benign"], alpha, beta)

    if "low" in prototypes:
        out["S_low"] = mixed_norm_score(X, prototypes["low"], alpha, beta)

    if "high" in prototypes:
        out["S_high"] = mixed_norm_score(X, prototypes["high"], alpha, beta)

    return pd.DataFrame(out)


def rank_low_penetrance_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    meta_query: pd.DataFrame | None = None,
    benign_labels=(0, 1),
    low_labels=(2, 3),
    high_labels=(4, 5),
    alpha: float = 0.7,
    beta: float = 0.3,
    margin: float = 0.0,
):
    """
    Build prototypes from train embeddings and score query embeddings.
    """
    prototypes = build_numeric_prototypes(
        X_train,
        y_train,
        benign_labels=benign_labels,
        low_labels=low_labels,
        high_labels=high_labels,
    )

    df_scores = compute_variant_scores(
        X_query,
        prototypes=prototypes,
        alpha=alpha,
        beta=beta,
    )

    # selection logic
    if {"S_benign", "S_low"}.issubset(df_scores.columns):
        df_scores["delta_benign_low"] = df_scores["S_benign"] - df_scores["S_low"]
    else:
        df_scores["delta_benign_low"] = np.nan

    if {"S_high", "S_low"}.issubset(df_scores.columns):
        df_scores["delta_high_low"] = df_scores["S_high"] - df_scores["S_low"]
    else:
        df_scores["delta_high_low"] = np.nan

    # priority score
    if {"delta_benign_low", "delta_high_low"}.issubset(df_scores.columns):
        df_scores["priority_low_pen"] = (
            df_scores["delta_benign_low"].fillna(0.0) +
            0.5 * df_scores["delta_high_low"].fillna(0.0)
        )
    else:
        df_scores["priority_low_pen"] = np.nan

    # select candidates: closer to low than to benign and high
    selected_mask = np.ones(len(df_scores), dtype=bool)

    if "delta_benign_low" in df_scores.columns:
        selected_mask &= (df_scores["delta_benign_low"] > margin).fillna(False)

    if "delta_high_low" in df_scores.columns:
        selected_mask &= (df_scores["delta_high_low"] > margin).fillna(False)

    df_selected = df_scores.loc[selected_mask].copy()
    df_selected = df_selected.sort_values("priority_low_pen", ascending=False)

    if meta_query is not None:
        meta_query = meta_query.reset_index(drop=True)
        df_scores = pd.concat([meta_query, df_scores.reset_index(drop=True)], axis=1)
        df_selected = pd.concat(
            [
                meta_query.loc[df_selected.index].reset_index(drop=True),
                df_selected.reset_index(drop=True),
            ],
            axis=1,
        )

    return df_scores, df_selected, prototypes