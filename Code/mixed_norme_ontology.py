from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd


# =========================================================
# Utils
# =========================================================
def _is_na_like(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "na", "nan", "none", ".", "null"}


def parse_terms(x, seps=r"[;,|]"):
    if _is_na_like(x):
        return set()
    parts = re.split(seps, str(x).lower())
    out = set()
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.add(p)
    return out


def row_terms_union(row: pd.Series, cols=("GO_BP", "GO_MF", "GO_CC")):
    terms = set()
    for c in cols:
        if c in row.index:
            terms |= parse_terms(row[c])
    return terms


def jaccard(a: set, b: set) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0
    union = a | b
    if len(union) == 0:
        return 0.0
    return len(a & b) / len(union)


# =========================================================
# Ontology anchor prototypes with Jaccard soft labels
# =========================================================
def compute_anchor_weights_from_jaccard(
    df: pd.DataFrame,
    anchor_terms: set,
    ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
):
    weights = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        terms_i = row_terms_union(df.iloc[i], ontology_cols)
        weights[i] = float(jaccard(terms_i, anchor_terms))

    return weights


def normalize_weights(w: np.ndarray, eps: float = 1e-12):
    w = np.asarray(w, dtype=np.float32)
    s = float(w.sum())
    if s <= eps:
        raise ValueError("All ontology weights are zero.")
    return w / s


def weighted_centroid(X: np.ndarray, w: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    w = normalize_weights(w)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if len(X) != len(w):
        raise ValueError(f"X and weights must have same length: {len(X)} vs {len(w)}")

    return np.sum(X * w[:, None], axis=0)


def build_anchor_prototype_from_ontology(
    X: np.ndarray,
    df: pd.DataFrame,
    anchor_terms,
    ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
    min_nonzero: int = 1,
):
    anchor_terms = {str(t).strip().lower() for t in anchor_terms if str(t).strip()}
    if len(anchor_terms) == 0:
        raise ValueError("anchor_terms is empty")

    w = compute_anchor_weights_from_jaccard(
        df=df,
        anchor_terms=anchor_terms,
        ontology_cols=ontology_cols,
    )

    n_nonzero = int((w > 0).sum())
    if n_nonzero < min_nonzero:
        raise ValueError(
            f"Not enough overlap with anchor terms. nonzero={n_nonzero}, required>={min_nonzero}"
        )

    P = weighted_centroid(X, w)
    return P, w


def build_multiple_anchor_prototypes(
    X: np.ndarray,
    df: pd.DataFrame,
    anchor_definitions: dict,
    ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
    min_nonzero: int = 1,
):
    prototypes = {}
    weights = {}

    for name, terms in anchor_definitions.items():
        P, w = build_anchor_prototype_from_ontology(
            X=X,
            df=df,
            anchor_terms=terms,
            ontology_cols=ontology_cols,
            min_nonzero=min_nonzero,
        )
        prototypes[name] = P
        weights[name] = w

    return prototypes, weights


def combine_prototypes(prototypes: dict, coeffs: dict, normalize_coeffs: bool = True):
    if len(coeffs) == 0:
        raise ValueError("coeffs is empty")

    missing = [k for k in coeffs if k not in prototypes]
    if missing:
        raise KeyError(f"Missing prototypes: {missing}")

    coeffs = {k: float(v) for k, v in coeffs.items()}

    if normalize_coeffs:
        s = sum(coeffs.values())
        if s <= 0:
            raise ValueError("Sum of coeffs must be > 0")
        coeffs = {k: v / s for k, v in coeffs.items()}

    first_key = next(iter(coeffs.keys()))
    P = np.zeros_like(prototypes[first_key], dtype=np.float32)

    for k, v in coeffs.items():
        P += np.asarray(prototypes[k], dtype=np.float32) * np.float32(v)

    return P


# =========================================================
# Label-based reference prototypes
# =========================================================
def prototype_from_labels(X: np.ndarray, y: np.ndarray, target_labels):
    y = np.asarray(y)
    mask = np.isin(y, list(target_labels))
    if not np.any(mask):
        raise ValueError(f"No rows found for labels {target_labels}")
    return X[mask].mean(axis=0)


# =========================================================
# Mixed norm
# =========================================================
def mixed_norm_score(X: np.ndarray, ref: np.ndarray, alpha: float = 0.7, beta: float = 0.3):
    X = np.asarray(X, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if ref.ndim != 1:
        raise ValueError(f"ref must be 1D, got {ref.shape}")
    if X.shape[1] != ref.shape[0]:
        raise ValueError(f"Dimension mismatch: X has {X.shape[1]}, ref has {ref.shape[0]}")

    l2_term = np.linalg.norm(X, axis=1)
    dist_term = np.linalg.norm(X - ref[None, :], axis=1)

    return alpha * l2_term + beta * dist_term


# =========================================================
# Main pipeline
# =========================================================
def zero_shot_low_penetrance_from_cfg(
    H: np.ndarray,
    y,
    evaluate_bio_tsv: str,
    out_dir: str = "./results_zero_shot_low_pen",
    ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
    alpha: float = 0.7,
    beta: float = 0.3,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(evaluate_bio_tsv, sep="\t")

    if len(df) != len(H):
        raise ValueError(
            f"Mismatch between TSV rows and embeddings: len(df)={len(df)} vs len(H)={len(H)}"
        )

    # if "label_clinvar" not in df.columns:
    #     raise ValueError("Column 'label_clinvar' not found in TSV")

    # y = df["label_clinvar"].to_numpy()

    print("Loaded TSV:", evaluate_bio_tsv)
    print("Embeddings shape:", H.shape)
    print("Label distribution:")
    # print(df["label_clinvar"].value_counts(dropna=False).sort_index())

    # -----------------------------------------------------
    # Define ontology anchors for low penetrance
    # Adapt these to your biology if needed
    # -----------------------------------------------------
    low_anchor_definitions = {
        "dna_repair_moderate": {
            "dna repair",
            "homologous recombination",
            "double-strand break repair",
            "chromatin binding",
            "genome stability",
        },
        "immune_dysregulation": {
            "immune response",
            "inflammatory response",
            "cytokine signaling",
            "cytokine receptor binding",
            "immune system process",
        },
        "polygenic_susceptibility": {
            "signal transduction",
            "cell communication",
            "regulation of signaling",
            "protein binding",
            "enzyme regulator activity",
        },
    }

    # weights of the anchors in the combined zero-shot prototype
    low_anchor_coeffs = {
        "dna_repair_moderate": 0.4,
        "immune_dysregulation": 0.4,
        "polygenic_susceptibility": 0.2,
    }

    # -----------------------------------------------------
    # Build ontology anchor prototypes using Jaccard soft labels
    # -----------------------------------------------------
    prototypes, anchor_weights = build_multiple_anchor_prototypes(
        X=H,
        df=df,
        anchor_definitions=low_anchor_definitions,
        ontology_cols=ontology_cols,
        min_nonzero=1,
    )

    P_low_zs = combine_prototypes(
        prototypes=prototypes,
        coeffs=low_anchor_coeffs,
        normalize_coeffs=True,
    )

    # -----------------------------------------------------
    # Optional benign/high references from labels
    # Here:
    #   benign = labels 0,1
    #   high   = labels 4,5
    #   low/intermediate are around 2,3
    # Adapt if your coding differs
    # -----------------------------------------------------
    P_benign = prototype_from_labels(H, y, target_labels=(0,3))
    P_high = prototype_from_labels(H, y, target_labels=(4, 5))

    # -----------------------------------------------------
    # Compute scores
    # -----------------------------------------------------
    S_low_zs = mixed_norm_score(H, P_low_zs, alpha=alpha, beta=beta)
    S_benign = mixed_norm_score(H, P_benign, alpha=alpha, beta=beta)
    S_high = mixed_norm_score(H, P_high, alpha=alpha, beta=beta)

    df_out = df.copy()
    df_out["S_low_zs"] = S_low_zs
    df_out["S_benign"] = S_benign
    df_out["S_high"] = S_high
    df_out["delta_benign_low"] = df_out["S_benign"] - df_out["S_low_zs"]
    df_out["delta_high_low"] = df_out["S_high"] - df_out["S_low_zs"]

    # priority: closer to low than to benign and high
    df_out["priority_low_pen"] = (
        df_out["delta_benign_low"] + 0.5 * df_out["delta_high_low"]
    )

    # ranked table
    df_ranked = df_out.sort_values(
        by=["priority_low_pen", "S_low_zs"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # selected candidates
    df_selected = df_ranked[
        (df_ranked["delta_benign_low"] > 0) &
        (df_ranked["delta_high_low"] > 0)
    ].reset_index(drop=True)

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    ranked_csv = os.path.join(out_dir, "ranked_zero_shot_low_penetrance.csv")
    selected_csv = os.path.join(out_dir, "selected_zero_shot_low_penetrance.csv")

    df_ranked.to_csv(ranked_csv, index=False)
    df_selected.to_csv(selected_csv, index=False)

    print("\nSaved ranked table:", ranked_csv)
    print("Saved selected table:", selected_csv)

    print("\nTop 20 ranked candidates:")
    cols_to_show = [c for c in [
        "Gene_Name", "CHROM","POS", "REF", "ALT","Prediction_ACMG_tapes", "KEGG_Gene", "KEGG_Pathway", "label_clinvar",
        "S_low_zs", "S_benign", "S_high",
        "delta_benign_low", "delta_high_low", "priority_low_pen"
    ] if c in df_ranked.columns]
    print(df_ranked[cols_to_show].head(20))
    selected_csv = os.path.join(out_dir, "ontology_mixed_norme_low.csv")
    df_ranked.to_csv(selected_csv, index=False)

    return {
        "df": df,
        "df_ranked": df_ranked,
        "df_selected": df_selected,
        "prototypes": prototypes,
        "P_low_zs": P_low_zs,
        "P_benign": P_benign,
        "P_high": P_high,
        "anchor_weights": anchor_weights,
    }


# =========================================================
# Example call in your script
# =========================================================
# Suppose:
#   H = fused embeddings already computed
#   cfg.evaluate_bio = path to your TSV
#
# Example:
#
# results = zero_shot_low_penetrance_from_cfg(
#     H=H,
#     evaluate_bio_tsv=cfg.evaluate_bio,
#     out_dir="./results_zero_shot_low_pen",
#     ontology_cols=("GO_BP", "GO_MF", "GO_CC"),
#     alpha=0.7,
#     beta=0.3,
# )

# df_ranked = results["df_ranked"]
# df_selected = results["df_selected"]