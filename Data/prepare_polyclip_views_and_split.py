#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  1) Load ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv
  2) Build SEQUENCE view (raw DNA): SEQ_REF / SEQ_ALT from reference FASTA
  3) Build BIO/tabular feature matrix X (numeric + encoded categorical)
  4) Split TRAIN/VAL by FAMILY AFTER building both views
  5) Save:
      - train_sequences.tsv / val_sequences.tsv   (raw sequences for DNABERT-2)
      - X_train.parquet / X_val.parquet
      - X_train.csv / X_val.csv
      - y_train.npy / y_val.npy (optional label)
      - train_meta.tsv / val_meta.tsv
      - split_info.json / feature_meta.json

Usage to hold one family:
  python prepare_clip_views_and_split_dnabert2.py \
      --tsv ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv \
      --fasta /path/to/hg38.fa \
      --out_dir clip_ready \
      --half_window 50 \
      --label label_of_interest \
      --val_families OD
"""

import os
import re
import json
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pyfaidx import Fasta



MISSING_STRINGS = {"", ".", "NA", "nan", "None", "NULL", "NaN"}

def _as_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    return "" if s in MISSING_STRINGS else s

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_mkdir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def parse_family_list_cell(cell: str) -> List[str]:
    s = _as_str(cell)
    if not s:
        return []
    parts = re.split(r"[,\|;\s]+", s)
    parts = [p.strip() for p in parts if p.strip() and p.strip() not in MISSING_STRINGS]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def is_per_sample_col(col: str) -> bool:
    if "." not in col:
        return False
    suffix = col.rsplit(".", 1)[-1]
    return suffix in {"STATUT", "AD", "DP", "GQ", "GT", "PID", "Nb_Copy", "BAF"}

def clean_string_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .replace({x: "" for x in MISSING_STRINGS})
         .fillna("")
         .str.strip()
    )


DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1]

def detect_fasta_chr_prefix(fa: Fasta) -> bool:
    for k in fa.keys():
        return str(k).startswith("chr")
    return False

def normalize_chr_for_fasta(chrom: str, fasta_has_chr_prefix: bool) -> str:
    c = str(chrom).strip().replace("CHR", "").replace("chr", "")
    return ("chr" + c) if fasta_has_chr_prefix else c

def fetch_ref_window(
    fa: Fasta,
    chrom: str,
    pos_1based: int,
    half_window: int,
    fasta_has_chr_prefix: bool,
) -> Tuple[str, int, int]:
    chrom_fa = normalize_chr_for_fasta(chrom, fasta_has_chr_prefix)
    left = max(1, int(pos_1based) - half_window)
    right = int(pos_1based) + half_window
    seq = fa[chrom_fa][left - 1:right].seq.upper()
    return seq, left, right

def apply_variant_to_window(
    window_seq: str,
    window_left_1based: int,
    pos_1based: int,
    ref: str,
    alt: str,
) -> Tuple[str, bool]:
    ref = str(ref).upper()
    alt = str(alt).upper()
    offset = int(pos_1based) - int(window_left_1based)  # 0-based
    if offset < 0 or offset >= len(window_seq):
        return window_seq, False

    Lref = len(ref)
    seg = window_seq[offset:offset + Lref]
    ok = (seg == ref)

    alt_seq = window_seq[:offset] + alt + window_seq[offset + Lref:]
    return alt_seq, ok

def build_ref_alt_sequences_fixed_len(
    fa: Fasta,
    chrom: str,
    pos_1based: int,
    ref: str,
    alt: str,
    half_window: int,
    strand: Optional[str],
    fasta_has_chr_prefix: bool,
) -> Tuple[str, str, int]:

    target_len = 2 * half_window + 1

    w_ref, left, _ = fetch_ref_window(fa, chrom, pos_1based, half_window, fasta_has_chr_prefix)
    w_alt, ok = apply_variant_to_window(w_ref, left, pos_1based, ref, alt)


    center = half_window

    def fix_len(seq: str) -> str:
        if len(seq) == target_len:
            return seq
        if len(seq) > target_len:
            start = max(0, center - half_window)
            end = start + target_len
            if end > len(seq):
                end = len(seq)
                start = max(0, end - target_len)
            return seq[start:end]
        pad = target_len - len(seq)
        left_pad = pad // 2
        right_pad = pad - left_pad
        return ("N" * left_pad) + seq + ("N" * right_pad)

    w_ref = fix_len(w_ref)
    w_alt = fix_len(w_alt)

    if strand is not None:
        s = str(strand).strip()
        if s in {"-", "-1"}:
            w_ref = revcomp(w_ref)
            w_alt = revcomp(w_alt)

    return w_ref, w_alt, int(ok)

def add_sequence_columns(
    df: pd.DataFrame,
    fasta_path: str,
    half_window: int,
    use_strand: bool,
) -> pd.DataFrame:
    fa = Fasta(fasta_path, as_raw=False, sequence_always_upper=True)
    fasta_has_chr_prefix = detect_fasta_chr_prefix(fa)

    out = df.copy()
    pos = pd.to_numeric(out["POS"], errors="coerce").fillna(-1).astype(int)
    strand_col = "STRAND" if (use_strand and "STRAND" in out.columns) else None

    seq_ref, seq_alt, ok_list = [], [], []

    for i in range(len(out)):
        chrom = out.iloc[i]["CHROM"]
        p = int(pos.iat[i])
        ref = out.iloc[i]["REF"]
        alt = out.iloc[i]["ALT"]
        strand = out.iloc[i][strand_col] if strand_col else None

        if p <= 0 or pd.isna(chrom) or pd.isna(ref) or pd.isna(alt):
            seq_ref.append("")
            seq_alt.append("")
            ok_list.append(0)
            continue

        try:
            sref, salt, ok = build_ref_alt_sequences_fixed_len(
                fa=fa,
                chrom=str(chrom),
                pos_1based=p,
                ref=str(ref),
                alt=str(alt),
                half_window=half_window,
                strand=strand,
                fasta_has_chr_prefix=fasta_has_chr_prefix
            )
        except Exception:
            sref, salt, ok = "", "", 0

        seq_ref.append(sref)
        seq_alt.append(salt)
        ok_list.append(ok)

    out["SEQ_REF"] = seq_ref
    out["SEQ_ALT"] = seq_alt
    out["REF_MATCH"] = ok_list
    return out


def multihot_from_consequence(cons: pd.Series, keep_terms: Optional[List[str]] = None) -> pd.DataFrame:
    cons = clean_string_series(cons).str.lower()
    if keep_terms is None:
        keep_terms = [
            "missense_variant",
            "stop_gained",
            "frameshift_variant",
            "splice_acceptor_variant",
            "splice_donor_variant",
            "splice_region_variant",
            "start_lost",
            "stop_lost",
            "inframe_insertion",
            "inframe_deletion",
            "protein_altering_variant",
            "synonymous_variant",
            "utr_variant",
            "intron_variant",
            "intergenic_variant",
            "regulatory_region_variant",
        ]
    out = {}
    for t in keep_terms:
        out[f"CONS__{t}"] = cons.str.contains(re.escape(t), na=False).astype(int)
    return pd.DataFrame(out, index=cons.index)

def onehot_small_vocab(s: pd.Series, prefix: str, top_k: int = 30) -> pd.DataFrame:
    s = clean_string_series(s)
    vc = s.value_counts(dropna=False)
    keep = vc.index[:top_k].tolist()
    s2 = s.where(s.isin(keep), other="OTHER")
    return pd.get_dummies(s2, prefix=prefix, dummy_na=False)

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = df.copy()

    per_sample_cols = [c for c in df.columns if is_per_sample_col(c)]
    df = df.drop(columns=per_sample_cols, errors="ignore")

    num_cols = [
        "MAX_POP_AF",
        "DP_GERM", "VAF_GERM",
        "DP_SOM", "VAF_SOM",
        "ACMG_INT_GERM", "ACMG_INT_SOM",
        "DANN_Score", "PhyloP", "PhastCons",
        "SIFT", "PolyPhen", "MutationTaster",
        "FATHMM_Coding_Score", "Distance_Grantham",
        "MaxEntScan_diff", "ada_score", "rf_score",
        "RegulomeDB_score", "Gene_damage_index",
        "SOMATIC_REPEAT_COUNT",
        "LOH_PARTIAL_PATHO_SOM",
        "two_hit_flag",
        "N_PATIENTS_WITH_VARIANT",
        "N_CANCER_WITH_VARIANT",
        "N_NONCANCER_WITH_VARIANT",
        "N_GERM_ONLY",
        "N_GERM_AND_SOM",
        "FAM_SIRIUS_FLAG",
        "FAM_COUNT_GE_2",
        "SEGREGATION_SCORE",
        "FAMILY_EVIDENCE_SCORE",
        "PATHWAY_HIT",
        "PATH_DNA_REPAIR",
        "predisposition_flag",
        "somatic_driver_flag",
        "literature_flag",
        "patient_cancer_evidence_score",
        "patient_cancer_evidence_class",
        "SIRIUS_PASS_POPAF",
        "SIRIUS_PASS_NONSYN",
        "SIRIUS_PASS_SNV_GERMLINE",
        "SIRIUS_PASS_SNV_SOMATIC",
        "SIRIUS_PASS_CNV",
        "IN_ROH",
        "HET_HIGH_HOM",
        "CNV_pathogenic",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    X_num = pd.DataFrame(index=df.index)
    for c in num_cols:
        X_num[c] = to_num(df[c])

    cat_blocks = []
    if "Consequence" in df.columns:
        cat_blocks.append(multihot_from_consequence(df["Consequence"]))
    if "IMPACT" in df.columns:
        cat_blocks.append(onehot_small_vocab(df["IMPACT"], prefix="IMPACT", top_k=10))
    if "VARIANT_CLASS" in df.columns:
        cat_blocks.append(onehot_small_vocab(df["VARIANT_CLASS"], prefix="VC", top_k=20))
    if "BIOTYPE" in df.columns:
        cat_blocks.append(onehot_small_vocab(df["BIOTYPE"], prefix="BIOTYPE", top_k=20))
    if "CANONICAL" in df.columns:
        can = clean_string_series(df["CANONICAL"]).str.upper()
        cat_blocks.append(pd.DataFrame({"CANONICAL__1": can.isin(["YES", "Y", "1", "TRUE"]).astype(int)}, index=df.index))

    X_cat = pd.concat(cat_blocks, axis=1) if cat_blocks else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)

    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            med = X[c].median(skipna=True)
            if np.isnan(med):
                med = 0.0
            X[c] = X[c].fillna(med)

    X = X.astype(np.float32)

    meta = {
        "numeric_cols_used": X_num.columns.tolist(),
        "categorical_cols_added": X_cat.columns.tolist(),
        "dropped_per_sample_cols": per_sample_cols,
    }
    return X, meta

def choose_label(df: pd.DataFrame, label_name: str) -> pd.Series:
    if not label_name:
        return pd.Series([0]*len(df), index=df.index)
    if label_name not in df.columns:
        raise ValueError(f"Label '{label_name}' not found in dataframe.")

    if label_name == "INTEGRATED_CLASS":
        m = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "TOP": 3}
        return clean_string_series(df[label_name]).str.upper().map(m).fillna(-1).astype(int)

    if label_name in {"label_of_interest", "FAM_COUNT_GE_2", "FAM_SIRIUS_FLAG"}:
        return to_num(df[label_name]).fillna(0).astype(int)

    return to_num(df[label_name])


def split_by_family(
    df: pd.DataFrame,
    family_col: str,
    val_families: Optional[List[str]],
    val_family_regex: Optional[str],
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    fam_lists = df[family_col].apply(parse_family_list_cell)
    all_fams = sorted({f for lst in fam_lists for f in lst})
    if not all_fams:
        raise ValueError(f"No families detected in {family_col}.")

    if val_families and len(val_families) > 0:
        val_set = set(map(str, val_families))
    elif val_family_regex:
        rx = re.compile(val_family_regex)
        val_set = {f for f in all_fams if rx.search(f)}
        if not val_set:
            raise ValueError(f"Regex '{val_family_regex}' matched 0 families among {len(all_fams)}.")
    else:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(val_frac * len(all_fams))))
        val_set = set(rng.choice(all_fams, size=n_val, replace=False).tolist())

    in_val = fam_lists.apply(lambda lst: any(f in val_set for f in lst)).astype(bool)
    df_val = df.loc[in_val].copy()
    df_train = df.loc[~in_val].copy()
    return df_train, df_val, sorted(val_set)




def detailed_column_statistics(df: pd.DataFrame, top_k: int = 5):


    print("=" * 120)
    print(f"DATASET SHAPE: {df.shape}")
    print(f"TOTAL MEMORY USAGE: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("=" * 120)

    stats = []

    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        n = len(s)
        n_missing = s.isna().sum()
        n_unique = s.nunique(dropna=True)

        base_info = {
            "column": col,
            "dtype": str(dtype),
            "n": n,
            "missing": int(n_missing),
            "missing_%": round(100 * n_missing / n, 2),
            "unique": int(n_unique),
        }


        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            base_info.update({
                "mean": desc.get("mean"),
                "std": desc.get("std"),
                "min": desc.get("min"),
                "p1": desc.get("1%"),
                "p5": desc.get("5%"),
                "median": desc.get("50%"),
                "p95": desc.get("95%"),
                "p99": desc.get("99%"),
                "max": desc.get("max"),
            })


        else:
            vc = s.value_counts(dropna=True).head(top_k)
            top_values = ", ".join([f"{k} ({v})" for k, v in vc.items()])
            base_info.update({
                "top_values": top_values
            })

        stats.append(base_info)

    stats_df = pd.DataFrame(stats)

    print("\nSUMMARY TABLE:\n")
    print(stats_df)

    return stats_df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv")
    ap.add_argument("--fasta", required=True, help="Reference genome FASTA (hg19/hg38) with .fai")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--half_window", type=int, default=50, help="Context half-window. 50 => 101bp")
    ap.add_argument("--use_strand", action="store_true", help="If set and STRAND exists: reverse-complement on '-'")
    ap.add_argument("--label", default="label_of_interest",
                    help="Label column to export: label_of_interest | FAM_COUNT_GE_2 | INTEGRATED_SCORE | INTEGRATED_CLASS ...")
    ap.add_argument("--val_families", default="",
                    help="Comma-separated families for validation holdout (e.g., 'OD'). If empty uses val_frac or regex.")
    ap.add_argument("--val_regex", default="", help="Regex selecting validation families (e.g., '^OD').")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction of families for validation if none specified.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.tsv, sep="\t", low_memory=False)

    for c in ["CHROM", "POS", "REF", "ALT"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column for sequence: {c}")

    family_col = "FAMILY_ID_LIST_GLOBAL"
    if family_col not in df.columns:
        raise ValueError(f"Missing required family column: {family_col}")

    df = add_sequence_columns(
        df=df,
        fasta_path=args.fasta,
        half_window=args.half_window,
        use_strand=args.use_strand,
    )

    X_all, feat_meta = build_feature_matrix(df)

    stats = detailed_column_statistics(X_all)

    val_fams = [x.strip() for x in args.val_families.split(",") if x.strip()] if args.val_families else None
    df_train, df_val, val_set = split_by_family(
        df=df,
        family_col=family_col,
        val_families=val_fams,
        val_family_regex=(args.val_regex if args.val_regex else None),
        val_frac=args.val_frac,
        seed=args.seed
    )

    stats = detailed_column_statistics(df_train)
    print("General Train stats")
    print(stats)
    stats.to_csv("column_statistics_report2.csv", index=False)

    stats = detailed_column_statistics(df_val)
    print("General test stats")
    print(stats)
    stats.to_csv("column_statistics_report3.csv", index=False)

    X_train = X_all.loc[df_train.index].copy()
    X_val   = X_all.loc[df_val.index].copy()

    all_cols = sorted(set(X_train.columns) | set(X_val.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0.0).astype(np.float32)
    X_val   = X_val.reindex(columns=all_cols, fill_value=0.0).astype(np.float32)

    y_train = choose_label(df_train, args.label)
    y_val   = choose_label(df_val, args.label)

    seq_cols = ["CHROM", "POS", "REF", "ALT", "TX", "SEQ_REF", "SEQ_ALT", "REF_MATCH"]
    for c in ["Gene_Name", "HGNC_Name", "FAMILY_ID_LIST_GLOBAL", "PATIENT_ID_LIST_GLOBAL"]:
        if c in df.columns:
            seq_cols.append(c)
    seq_cols = [c for c in seq_cols if c in df.columns]

    train_seq_path = os.path.join(args.out_dir, "train_sequences.tsv")
    val_seq_path   = os.path.join(args.out_dir, "val_sequences.tsv")
    df_train[seq_cols].to_csv(train_seq_path, sep="\t", index=False)
    df_val[seq_cols].to_csv(val_seq_path, sep="\t", index=False)

    X_train_path = os.path.join(args.out_dir, "X_train.parquet")
    X_val_path   = os.path.join(args.out_dir, "X_val.parquet")
    y_train_path = os.path.join(args.out_dir, "y_train.npy")
    y_val_path   = os.path.join(args.out_dir, "y_val.npy")
    X_train_path_tsv = os.path.join(args.out_dir, "X_train.tsv")
    X_val_path_tsv   = os.path.join(args.out_dir, "X_val.tsv")

    X_train.to_parquet(X_train_path, index=False)
    X_val.to_parquet(X_val_path, index=False)

    np.save(y_train_path, np.asarray(y_train))
    np.save(y_val_path, np.asarray(y_val))
    # np.save(X_train_path_npy, np.asarray(X_train))
    # np.save(X_val_path_npy, np.asarray(X_val))

    X_train.to_csv(X_train_path_tsv, sep="\t", index=False)
    X_val.to_csv(X_val_path_tsv, sep="\t", index=False)

    id_cols = [
        "CHROM", "POS", "REF", "ALT", "TX", "Gene_Name", "HGNC_Name",
        "FAMILY_ID_LIST_GLOBAL", "PATIENT_ID_LIST_GLOBAL",
        "INTEGRATED_SCORE", "INTEGRATED_CLASS", "INTEGRATED_EVIDENCE",
        "label_of_interest", "FAM_COUNT_GE_2", "SEGREGATION_SCORE"
    ]
    id_cols = [c for c in id_cols if c in df.columns]

    train_meta_path = os.path.join(args.out_dir, "train_meta.tsv")
    val_meta_path   = os.path.join(args.out_dir, "val_meta.tsv")
    df_train[id_cols].to_csv(train_meta_path, sep="\t", index=False)
    df_val[id_cols].to_csv(val_meta_path, sep="\t", index=False)

    split_info = {
        "label": args.label,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(df_train)),
        "n_rows_val": int(len(df_val)),
        "n_families_val": int(len(val_set)),
        "val_families": val_set,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "val_regex": args.val_regex,
        "val_families_arg": args.val_families,
        "half_window": args.half_window,
        "use_strand": bool(args.use_strand),
        "sequence_tokenization": "raw_dna_for_dnabert2",
        "sequence_length_bp": int(2 * args.half_window + 1),
    }
    with open(os.path.join(args.out_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    feat_meta_out = {
        "feature_meta": feat_meta,
        "final_feature_dim": int(len(all_cols)),
        "final_feature_columns": all_cols,
    }
    with open(os.path.join(args.out_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(feat_meta_out, f, indent=2)

    # 
    ref_match_rate = float(pd.to_numeric(df["REF_MATCH"], errors="coerce").fillna(0).mean())
    empty_seq_rate = float((df["SEQ_REF"].astype(str).str.len() == 0).mean())
    print("[OK] Saved:")
    print(" ", train_seq_path)
    print(" ", val_seq_path)
    print(" ", X_train_path)
    print(" ", X_val_path)
    print(" ", y_train_path)
    print(" ", y_val_path)
    print(" ", train_meta_path)
    print(" ", val_meta_path)
    print(" ", os.path.join(args.out_dir, "split_info.json"))
    print(" ", os.path.join(args.out_dir, "feature_meta.json"))
    print(f"[REF_MATCH mean] {ref_match_rate:.3f} (low => wrong hg build / chr naming / left-align issues)")
    print(f"[Empty SEQ_REF rate] {empty_seq_rate:.3f}")
    print(f"[Seq length bp] {2*args.half_window+1}")

if __name__ == "__main__":
    main()
