#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PolyCLIP-T variant model (sequence + numeric annotations).

   - Family holdout WITHOUT leakage removal
   - Shared variants (same CHROM,POS,REF,ALT) are kept in BOTH train and test if they exist in both.

Inputs
------
- TSV of  families 
- FASTA reference genome

Outputs
-------
- SEQ TSV: columns [CHROM, POS, REF, ALT, SEQ_REF, SEQ_ALT]
- NUMERIC TRAIN/TEST TSV: numeric + one-hot + labels + keep cols

Labels
----------------
It is included as numeric features:
  INTEGRATED_SCORE
  (optional: INTEGRATED_CLASS one-hot)
"""

import os
import re
import io
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from pyfaidx import Fasta
from sklearn.preprocessing import OneHotEncoder



NA_LIKE = {".", "NA", "nan", "None", "", "NULL", "NaN"}

def _norm_chr(ch: str) -> str:
    if ch is None or (isinstance(ch, float) and np.isnan(ch)):
        return ""
    s = str(ch).strip()
    s = s.replace("CHR", "").replace("chr", "")
    return s

def _to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def extract_float(x):
    """Ex: 'deleterious(0.02)' -> 0.02 ; '.'/'NA' -> NaN"""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in NA_LIKE:
        return np.nan
    m = re.search(r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", x)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def bin_from_text(x, positives):
    x = "" if pd.isna(x) else str(x).lower()
    return int(any(p in x for p in positives))

def encode_polyphen(x):
    x = "" if pd.isna(x) else str(x).lower()
    if "probably" in x:
        return 2
    if "possibly" in x:
        return 1
    if "benign" in x:
        return 0
    return np.nan



CONSEQUENCE_SCORE = {
    # HIGH
    "transcript_ablation": 5,
    "splice_acceptor_variant": 5,
    "splice_donor_variant": 5,
    "stop_gained": 5,
    "frameshift_variant": 5,
    "stop_lost": 5,
    "start_lost": 5,
    # MODERATE
    "missense_variant": 4,
    "inframe_insertion": 4,
    "inframe_deletion": 4,
    "protein_altering_variant": 4,
    # LOW
    "splice_region_variant": 3,
    "synonymous_variant": 2,
    "stop_retained_variant": 2,
    "start_retained_variant": 2,
    # MODIFIER
    "intron_variant": 1,
    "intergenic_variant": 1,
    "upstream_gene_variant": 1,
    "downstream_gene_variant": 1,
    "regulatory_region_variant": 1,
    "3_prime_utr_variant": 1,
    "5_prime_utr_variant": 1,
    "non_coding_transcript_variant": 1,
}

IMPACT_MAP = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

VARIANT_CLASS_MAP = {
    "SNV": 1, "substitution": 1, "sequence_alteration": 1,
    "insertion": 2, "deletion": 2, "indel": 2,
}

BIOTYPE_MAP = {
    "protein_coding": 3,
    "processed_transcript": 2,
    "lncRNA": 2, "lncrna": 2,
    "miRNA": 2, "snRNA": 2, "snoRNA": 2, "rRNA": 2,
    "pseudogene": 1, "processed_pseudogene": 1,
}

def _split_cons(x: str) -> List[str]:
    if x is None:
        return []
    s = str(x).strip().lower()
    if s in NA_LIKE:
        return []
    parts = re.split(r"[,&|]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def encode_consequence(x):
    """If multiple consequences -> take most severe"""
    parts = _split_cons(x)
    if not parts:
        return 0
    scores = [CONSEQUENCE_SCORE.get(p, 1) for p in parts]
    return int(max(scores)) if scores else 0

def encode_impact(x):
    if pd.isna(x):
        return 0
    return IMPACT_MAP.get(str(x).strip().upper(), 0)

def encode_variant_class(x):
    if pd.isna(x):
        return 0
    return VARIANT_CLASS_MAP.get(str(x).strip(), 0)

def encode_biotype(x):
    if pd.isna(x):
        return 0
    return BIOTYPE_MAP.get(str(x).strip(), 0)

def encode_canonical(x):
    if pd.isna(x):
        return 0
    s = str(x).strip()
    return 1 if s in {"YES", "1", "true", "True"} else 0


def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""



def fetch_ref_alt_window_indel_safe(
    fa: "Fasta",
    chrom: str,
    pos_1based: int,
    ref: str,
    alt: str,
    window: int = 201,
    extra_context: int = 200,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns fixed-length (SEQ_REF, SEQ_ALT) centered at pos_1based.
    For indels: best-effort replace; then trim/pad to keep fixed length.
    If reference mismatch at center -> returns (SEQ_REF, None) => dropped later.
    """
    try:
        chrom = _norm_chr(chrom)
        if not chrom:
            return None, None

        c1 = chrom
        c2 = "chr" + chrom
        if c1 in fa:
            ck = c1
        elif c2 in fa:
            ck = c2
        else:
            return None, None

        pos0 = int(pos_1based) - 1
        half = window // 2

        left0 = max(0, pos0 - half - extra_context)
        right0 = pos0 + half + extra_context + 1

        seq_big = str(fa[ck][left0:right0]).upper()
        if not seq_big:
            return None, None

        center = pos0 - left0
        L = center - half
        R = center + half + 1
        if L < 0 or R > len(seq_big):
            return None, None

        seq_ref = seq_big[L:R]
        if len(seq_ref) != window:
            return None, None

        ref = str(ref).upper()
        alt = str(alt).upper()
        if (ref in NA_LIKE) or (alt in NA_LIKE) or (not ref) or (not alt):
            return None, None

        if len(ref) == 1 and len(alt) == 1:
            seq_alt = seq_ref[:half] + alt + seq_ref[half + 1:]
            return seq_ref, seq_alt

        start = half
        end = min(window, half + len(ref))
        if seq_ref[start:end] != ref[: end - start]:
            return seq_ref, None

        seq_alt = seq_ref[:start] + alt + seq_ref[end:]

        if len(seq_alt) > window:
            seq_alt = seq_alt[:window]
        elif len(seq_alt) < window:
            seq_alt = seq_alt + ("N" * (window - len(seq_alt)))

        if len(seq_alt) != window:
            return None, None
        return seq_ref, seq_alt

    except Exception:
        return None, None



def create_sum_stat(df: pd.DataFrame, out_txt: str):
    summary_lines = []

    buf = io.StringIO()
    df.info(buf=buf)
    summary_lines.append("=== DataFrame Info ===\n")
    summary_lines.append(buf.getvalue() + "\n\n")

    summary_lines.append("=== Descriptive Statistics ===\n")
    summary_lines.append(str(df.describe(include="all")) + "\n\n")

    summary_lines.append("=== NaN and Unique Values per Column ===\n")
    nan_summary = df.isna().sum().to_frame(name="NaN_count")
    nan_summary["dtype"] = df.dtypes
    nan_summary["unique_values"] = df.nunique(dropna=True)
    summary_lines.append(str(nan_summary) + "\n\n")

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("Summary saved to", out_txt)



def build_numeric_ml_table(
    df_in: pd.DataFrame,
    out_dir: str,
    test_family: str,
    label_col_primary: str = "INTEGRATED_SCORE",
    window: int = 201,
):


    df = df_in.copy()

    if "FAMILY_ID" not in df.columns:
        raise ValueError("FAMILY_ID missing: cannot split train/test")


    cons_col = pick_col(df, ["EXOME_Consequence", "ANN_Consequence", "Consequence"])
    impact_col = pick_col(df, ["EXOME_IMPACT", "ANN_IMPACT", "IMPACT"])
    varcls_col = pick_col(df, ["EXOME_VARIANT_CLASS", "ANN_VARIANT_CLASS", "VARIANT_CLASS"])
    biotype_col = pick_col(df, ["EXOME_BIOTYPE", "ANN_BIOTYPE", "BIOTYPE"])
    canonical_col = pick_col(df, ["EXOME_CANONICAL", "ANN_CANONICAL", "CANONICAL"])


    sift_col = pick_col(df, ["ANN_SIFT", "EXOME_ANN_SIFT", "SIFT"])
    if sift_col:
        df["SIFT_score"] = df[sift_col].apply(extract_float)
        df["SIFT_bin"] = df[sift_col].apply(lambda x: bin_from_text(x, ["deleterious"]))
    else:
        df["SIFT_score"] = np.nan
        df["SIFT_bin"] = 0

    poly_col = pick_col(df, ["ANN_PolyPhen", "EXOME_ANN_PolyPhen", "PolyPhen"])
    if poly_col:
        df["PolyPhen_ord"] = df[poly_col].apply(encode_polyphen)
        df["PolyPhen_score"] = df[poly_col].apply(extract_float)
    else:
        df["PolyPhen_ord"] = np.nan
        df["PolyPhen_score"] = np.nan

    for col in ["ANN_MetaLR_pred", "ANN_MetaSVM_pred", "ANN_M-CAP_pred"]:
        outc = col + "_bin"
        if col in df.columns:
            df[outc] = df[col].apply(lambda x: bin_from_text(x, ["d"]))
        else:
            df[outc] = 0

    for col in ["ANN_PROVEAN_score", "ANN_DANN_Score", "ANN_PhyloP", "ANN_PhastCons", "ANN_GERP++_RS", "ANN_FATHMM_CS", "ANN_FATHMM_NCS"]:
        outc = col + "_num"
        if col in df.columns:
            df[outc] = df[col].apply(extract_float)
        else:
            df[outc] = np.nan


    df["Consequence_num"] = df[cons_col].apply(encode_consequence) if cons_col else 0
    df["Impact_num"] = df[impact_col].apply(encode_impact) if impact_col else 0
    df["VariantClass_num"] = df[varcls_col].apply(encode_variant_class) if varcls_col else 0
    df["Biotype_num"] = df[biotype_col].apply(encode_biotype) if biotype_col else 0
    df["Canonical_num"] = df[canonical_col].apply(encode_canonical) if canonical_col else 0


    NUMERIC_COLS = [
        "DP", "QUAL", "VAF", "INFO_AF",

        "ANN_GnomAD_Genome_AF", 
        "ANN_GnomAD_Genome_AF_popmax",
         "ANN_Kaviar_AF", "ANN_EVS_MAF",
        "ANN_gnomAD_AF", "ANN_GnomAD_MNV_AF",


        "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT",
        "N_GERM_ONLY", "N_GERM_AND_SOM",
        "FAMILY_EVIDENCE_SCORE", "SEGREGATION_SCORE",
        "ACMG_INT_GERM", "ACMG_INT_SOM",
        "SOMATIC_REPEAT_COUNT",
        "LOH_PARTIAL_PATHO",
        "PATHWAY_HIT", "PATH_DNA_REPAIR",

        # integrated score
        "INTEGRATED_SCORE",

    
        # derived predictors
        "SIFT_score", "PolyPhen_ord", "PolyPhen_score",
        "ANN_PROVEAN_score_num", "ANN_DANN_Score_num", "ANN_PhyloP_num",
        "ANN_PhastCons_num", "ANN_GERP++_RS_num", "ANN_FATHMM_CS_num", "ANN_FATHMM_NCS_num",

        # encodings
        "Consequence_num", "Impact_num", "VariantClass_num", "Biotype_num", "Canonical_num",
    ]

    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].apply(extract_float)


    BINARY_COLS = [
        "SIFT_bin",
        "ANN_MetaLR_pred_bin", "ANN_MetaSVM_pred_bin", "ANN_M-CAP_pred_bin",
        "FAM_SIRIUS_FLAG", "label_of_interest", "FAM_COUNT_GE_2",
        "PAT_SIRIUS_FLAG",
        "predisposition_flag", "somatic_driver_flag", "two_hit_flag", "literature_flag",
    ]
    for c in BINARY_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ------------------------------
    # Categorical -> one-hot
    # ------------------------------
    CATEGORICAL_COLS = [
        "GT",
        "CNV_LOH_status",
        # "ORIGIN_PROFILE",
        "INTEGRATED_CLASS", 
    ]
    cat_existing = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_existing:
        df[cat_existing] = df[cat_existing].fillna("NA").astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_matrix = encoder.fit_transform(df[cat_existing])
        cat_df = pd.DataFrame(
            cat_matrix,
            columns=encoder.get_feature_names_out(cat_existing),
            index=df.index
        )
    else:
        cat_df = pd.DataFrame(index=df.index)

    # ------------------------------
    # Labels
    # ------------------------------
    def _label_row(r):
        if label_col_primary in r.index:
            v = pd.to_numeric(r[label_col_primary], errors="coerce")
            v = 0 if pd.isna(v) else int(v)
            return 1 if n >=3  else 0
        if "FAM_COUNT_GE_2" in r.index:
            n = pd.to_numeric(r["FAM_COUNT_GE_2"], errors="coerce")
            n = 0 if pd.isna(n) else int(n)
            return 1 if v == 1 else 0
        return 0

    df["labels"] = df.apply(_label_row, axis=1).astype(int)


    keep = []
    for c in ["FAMILY_ID", "CHROM", "POS", "REF", "ALT", "allele_1", "allele_2", "labels"]:
        if c in df.columns:
            keep.append(c)

    df_train = df[df["FAMILY_ID"].astype(str) != str(test_family)].copy()
    df_test  = df[df["FAMILY_ID"].astype(str) == str(test_family)].copy()

    print("Train families:", sorted(df_train["FAMILY_ID"].astype(str).unique().tolist()))
    print("Test families :", sorted(df_test["FAMILY_ID"].astype(str).unique().tolist()))
    print("Train size:", len(df_train), "| pos:", int(df_train["labels"].sum()))
    print("Test size :", len(df_test),  "| pos:", int(df_test["labels"].sum()))

    def make_ml_block(sub_df: pd.DataFrame) -> pd.DataFrame:
        sub_cat = cat_df.loc[sub_df.index]
        out = pd.concat(
            [sub_df[keep + NUMERIC_COLS + BINARY_COLS].copy(), sub_cat],
            axis=1
        )
        out["POS"] = pd.to_numeric(out.get("POS", np.nan), errors="coerce")
        out[BINARY_COLS] = out[BINARY_COLS].fillna(0).astype(int)
        for c in NUMERIC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.fillna(0)
        return out

    train_final = make_ml_block(df_train)
    test_final  = make_ml_block(df_test)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "ML_NUMERIC_TRAIN.tsv")
    test_path  = os.path.join(out_dir, "ML_NUMERIC_TEST.tsv")

    train_final.to_csv(train_path, sep="\t", index=False)
    test_final.to_csv(test_path, sep="\t", index=False)

    print("Saved TRAIN:", train_path, "| shape:", train_final.shape)
    print(train_final["labels"].value_counts(dropna=False))
    print("Saved TEST :", test_path,  "| shape:", test_final.shape)
    print(test_final["labels"].value_counts(dropna=False))

    create_sum_stat(train_final, os.path.join(out_dir, "df_summary_train.txt"))
    create_sum_stat(test_final,  os.path.join(out_dir, "df_summary_test.txt"))

    return train_final, test_final



def build_ml_file_from_tsv(
    tsv_path: str,
    fasta_path: str,
    out_dir: str,
    test_family: str,
    window: int = 201,
    extra_context: int = 200,
):
    """
    1) Read merged TSV
    2) Create SEQ_REF / SEQ_ALT using (REF,ALT) if allele_1/allele_2 missing
    3) Save sequence TSV for CLIP (seq-view)
    4) Save numeric train/test TSV for CLIP (bio-view)
    """

    print("Loading TSV:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    required = ["CHROM", "POS", "REF", "ALT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in TSV: {missing}")

    df["CHROM"] = df["CHROM"].apply(_norm_chr)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    if "allele_1" not in df.columns:
        df["allele_1"] = df["REF"].astype(str)
    if "allele_2" not in df.columns:
        df["allele_2"] = df["ALT"].astype(str)

    fa = Fasta(fasta_path)

    seq_ref_list = []
    seq_alt_list = []

    for _, row in df.iterrows():
        chrom = row["CHROM"]
        pos = row["POS"]

        ref = str(row["allele_1"]).strip()
        alt = str(row["allele_2"]).strip()

        if pd.isna(pos) or ref in NA_LIKE or alt in NA_LIKE or (not ref) or (not alt):
            seq_ref_list.append(None)
            seq_alt_list.append(None)
            continue

        s_ref, s_alt = fetch_ref_alt_window_indel_safe(
            fa=fa,
            chrom=chrom,
            pos_1based=int(pos),
            ref=ref,
            alt=alt,
            window=window,
            extra_context=extra_context,
        )
        seq_ref_list.append(s_ref)
        seq_alt_list.append(s_alt)

    df["SEQ_REF"] = seq_ref_list
    df["SEQ_ALT"] = seq_alt_list

    before = len(df)
    df = df.dropna(subset=["SEQ_REF", "SEQ_ALT"]).copy()
    print(f"Kept {len(df)} / {before} variants with valid sequences")

    os.makedirs(out_dir, exist_ok=True)

    seq_path = os.path.join(out_dir, "ML_SEQUENCES.tsv")
    seq_cols = [c for c in ["FAMILY_ID", "CHROM", "POS", "REF", "ALT", "SEQ_REF", "SEQ_ALT"] if c in df.columns]
    df[seq_cols].to_csv(seq_path, sep="\t", index=False)
    print("Saved ML sequence file:", seq_path)

    build_numeric_ml_table(
        df_in=df,
        out_dir=out_dir,
        test_family=test_family,
        label_col_primary="FAM_COUNT_GE_2",
        window=window,
    )

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Input TSV from family pipeline (e.g., FAMILY_VARIANTS or GLOBAL_UNIQUE)")
    ap.add_argument("--fasta", required=True, help="Reference FASTA path")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--test_family", required=True, help="Family ID to hold out as test (e.g., ROG)")
    ap.add_argument("--window", type=int, default=201, help="Fixed sequence window length (odd number recommended)")
    ap.add_argument("--extra_context", type=int, default=200, help="Extra context to safely build indel windows")
    args = ap.parse_args()

    build_ml_file_from_tsv(
        tsv_path=args.tsv,
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        test_family=args.test_family,
        window=args.window,
        extra_context=args.extra_context,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
