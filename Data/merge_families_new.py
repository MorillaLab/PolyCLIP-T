#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Family pipeline (JSON-driven) producing TSV + CSV.

TSV outputs (kept as before, plus a few extra):
  - Per-family:
      * {FAMILY}/{FAMILY}__{PATIENT}__LONG.tsv
      * {FAMILY}/{FAMILY}__ALL_PATIENTS__LONG.tsv
      * {FAMILY}/{FAMILY}__PATIENT_VARIANTS.tsv
      * {FAMILY}/{FAMILY}__FAMILY_VARIANTS.tsv
      * {FAMILY}/{FAMILY}__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv
      * {FAMILY}/{FAMILY}__LABEL_OF_INTEREST_EQ_1.tsv
      * {FAMILY}/{FAMILY}__FAM_COUNT_GE_2.tsv                ( >=2 patients)
  - Global (all families):
      * ALL_FAMILIES__LONG.tsv
      * ALL_FAMILIES__PATIENT_VARIANTS.tsv
      * ALL_FAMILIES__FAMILY_VARIANTS.tsv
      * ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv
      * ALL_FAMILIES__SIRIUS_EQ_1__UNIQUE_VARIANTS.tsv
      * ALL_FAMILIES__PATIENT_EVIDENCE_CLASS_GE_3__UNIQUE_VARIANTS.tsv
      * ALL_FAMILIES__OPTION_B_ACMG_345__UNIQUE_VARIANTS.tsv
      * ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1__FAMILY_VARIANTS.tsv
      * ALL_FAMILIES__FAM_COUNT_GE_2.tsv                      ( >=2 patients)

CSV outputs (minimal “labels view”):
  - Per-family:
      * {FAMILY}/{FAMILY}__FAMILY_VARIANTS__LABELS.csv
      * {FAMILY}/{FAMILY}__SIRIUS_EQ_1__FAMILY_VARIANTS__LABELS.csv
      * {FAMILY}/{FAMILY}__LABEL_OF_INTEREST_EQ_1__LABELS.csv
      * {FAMILY}/{FAMILY}__FAM_COUNT_GE_2__LABELS.csv
  - Global:
      * ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS__LABELS.csv
      * ALL_FAMILIES__SIRIUS_EQ_1__UNIQUE_VARIANTS__LABELS.csv
      * ALL_FAMILIES__PATIENT_EVIDENCE_CLASS_GE_3__UNIQUE_VARIANTS__LABELS.csv
      * ALL_FAMILIES__OPTION_B_ACMG_345__UNIQUE_VARIANTS__LABELS.csv
      * ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1__FAMILY_VARIANTS__LABELS.csv
      * ALL_FAMILIES__FAM_COUNT_GE_2__LABELS.csv

CSV contains ONLY:
  CHROM, POS, REF, ALT, GENE_NAME,
  FAMILY_ID_LIST_GLOBAL, PATIENT_ID_LIST_GLOBAL,
  + label columns you choose.

Key rules implemented:
  - Every somatic row includes FAMILY_ID, PATIENT_ID.
  - For each patient: merge ALL somatic files first (unique per patient-variant), count repeats (SOMATIC_REPEAT_COUNT),
    keep max for numeric/flags and first non-null for text, then concatenate with germline.
  - Sirius flags computed per-family on FAMILY_LONG (origin-aware thresholds).
  - Patient-level internal flags computed on PATIENT_VARIANTS (unique patient-variant):
      predisposition_flag, somatic_driver_flag, two_hit_flag, literature_flag,
      patient_cancer_evidence_score (0..5), patient_cancer_evidence_class (0..5)
  - Family-level label_of_interest computed on FAMILY_VARIANTS:
      N_CANCER_WITH_VARIANT >= 1 AND N_NONCANCER_WITH_VARIANT == 0 AND FAM_SIRIUS_FLAG == 1
  - Additional block: FAM_COUNT_GE_2 (>=2 patients in the family have the variant)
  - Integrated scientific labels added on FAMILY_VARIANTS and GLOBAL_UNIQUE:
      INTEGRATED_SCORE (0..12), INTEGRATED_CLASS (LOW/MODERATE/HIGH/TOP), INTEGRATED_EVIDENCE
  - Uniqueness enforced:
      PATIENT_VARIANTS: (FAMILY_ID,PATIENT_ID,CHROM,POS,REF,ALT)
      FAMILY_VARIANTS:  (FAMILY_ID,CHROM,POS,REF,ALT)
      GLOBAL_UNIQUE:    (CHROM,POS,REF,ALT)

Allele columns:
  allele_1 / allele_2 are derived from GT indices using mapping [REF] + ALT_LIST.
"""

import os
import re
import gc
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from cyvcf2 import VCF


def normalize_chr(x) -> str:
    return str(x).replace("chr", "").replace("CHR", "").strip()

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def _to_num(x):
    return pd.to_numeric(x, errors="coerce")

def list_files_one(dirpath: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    """Return the only file in dirpath matching suffixes. If none -> None. If >1 -> error."""
    if not dirpath or not os.path.isdir(dirpath):
        return None
    hits = []
    for fn in os.listdir(dirpath):
        if fn.endswith(suffixes):
            hits.append(os.path.join(dirpath, fn))
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise ValueError(f"Expected ONE file in {dirpath} with {suffixes}, found {len(hits)}:\n" + "\n".join(hits))
    return hits[0]

def get_variant_key(df: pd.DataFrame) -> List[str]:
    if all(c in df.columns for c in ["CHROM", "POS", "REF", "ALT"]):
        return ["CHROM", "POS", "REF", "ALT"]
    return ["CHROM", "POS"]

def chrom_sort_key(ch: str):
    s = normalize_chr(ch)
    mapping = {str(i): i for i in range(1, 23)}
    mapping.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
    return mapping.get(s, 1000), s

def sort_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df2 = df.copy()
    df2["_chr_rank"] = df2["CHROM"].astype(str).map(lambda x: chrom_sort_key(x)[0])
    df2["_chr_str"] = df2["CHROM"].astype(str).map(lambda x: chrom_sort_key(x)[1])
    df2["POS"] = pd.to_numeric(df2["POS"], errors="coerce")
    df2 = df2.sort_values(by=["_chr_rank", "_chr_str", "POS", "REF", "ALT"], kind="mergesort")
    df2 = df2.drop(columns=["_chr_rank", "_chr_str"], errors="ignore")
    return df2

def safe_save(df: pd.DataFrame, path: str, sep: str):
    ensure_dir(os.path.dirname(path))
    try:
        df.to_csv(path, sep=sep, index=False)
        print(f"[SAVED] {path} | rows={len(df)} cols={df.shape[1]}")
    except Exception as e:
        print(f"[SAVE_ERROR] {path} -> {type(e).__name__}: {e}")


def normalize_acmg_by_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    source: "germline" ou "somatic"
    Crée:
      - GERM_Prediction_ACMG_tapes / SOM_Prediction_ACMG_tapes (pour add_sirius_pass_flags_family_long)
      - ACMG_GERMLINE / ACMG_SOMATIC (si tu veux les garder aussi)
    """
    out = df.copy()

    acmg_col = None
    for c in ["EXOME_Prediction_ACMG_tapes", "Prediction_ACMG_tapes"]:
        if c in out.columns:
            acmg_col = c
            break

    if "ACMG_GERMLINE" not in out.columns:
        out["ACMG_GERMLINE"] = np.nan
    if "ACMG_SOMATIC" not in out.columns:
        out["ACMG_SOMATIC"] = np.nan

    if acmg_col is None:
        return out

    if source == "germline":
        out["ACMG_GERMLINE"] = out[acmg_col]
        out["GERM_Prediction_ACMG_tapes"] = out[acmg_col]  # <- IMPORTANT
    elif source == "somatic":
        out["ACMG_SOMATIC"] = out[acmg_col]
        out["SOM_Prediction_ACMG_tapes"] = out[acmg_col]   # <- IMPORTANT

    return out



def resolve_gene_col(df: pd.DataFrame) -> str:
    candidates = [
        "GENE_NAME", "Gene_Name", "Gene", "GENE", "SYMBOL", "HGNC_Name", "HGNC",
        "EXOME_GENE_NAME", "EXOME_Gene_Name", "EXOME_Gene", "EXOME_SYMBOL", "EXOME_HGNC_Name",
        "ANN_Gene_Name", "EXOME_ANN_Gene_Name",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def attach_global_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure these columns always exist:
      - FAMILY_ID_LIST_GLOBAL
      - PATIENT_ID_LIST_GLOBAL
    """
    out = df.copy()

    if "FAMILY_ID_LIST_GLOBAL" not in out.columns:
        if "FAMILY_ID_LIST" in out.columns:
            out["FAMILY_ID_LIST_GLOBAL"] = out["FAMILY_ID_LIST"].astype(str)
        elif "FAMILY_ID" in out.columns:
            out["FAMILY_ID_LIST_GLOBAL"] = out["FAMILY_ID"].astype(str)
        else:
            out["FAMILY_ID_LIST_GLOBAL"] = ""

    if "PATIENT_ID_LIST_GLOBAL" not in out.columns:
        if "PATIENT_ID_LIST" in out.columns:
            out["PATIENT_ID_LIST_GLOBAL"] = out["PATIENT_ID_LIST"].astype(str)
        else:
            out["PATIENT_ID_LIST_GLOBAL"] = ""

    out["FAMILY_ID_LIST_GLOBAL"] = out["FAMILY_ID_LIST_GLOBAL"].replace({"nan": "", "None": ""}).fillna("")
    out["PATIENT_ID_LIST_GLOBAL"] = out["PATIENT_ID_LIST_GLOBAL"].replace({"nan": "", "None": ""}).fillna("")
    return out

def export_minimal_csv(df: pd.DataFrame, out_csv_path: str, label_cols: List[str]):
    """
    Save a minimal CSV view with ONLY:
      CHROM, POS, REF, ALT, GENE_NAME, FAMILY_ID_LIST_GLOBAL, PATIENT_ID_LIST_GLOBAL + label_cols
    """
    base_cols = ["CHROM", "POS", "REF", "ALT", "GENE_NAME", "PATIENT_ID_LIST_GLOBAL", "FAMILY_ID_LIST_GLOBAL"] + label_cols

    if df is None or df.empty:
        safe_save(pd.DataFrame(columns=base_cols), out_csv_path, sep=",")
        return

    tmp = attach_global_lists(df)

    gene_col = resolve_gene_col(tmp)
    if gene_col and gene_col != "GENE_NAME":
        tmp = tmp.rename(columns={gene_col: "GENE_NAME"})
    if "GENE_NAME" not in tmp.columns:
        tmp["GENE_NAME"] = ""

    for c in base_cols:
        if c not in tmp.columns:
            tmp[c] = "" if c in ["GENE_NAME", "PATIENT_ID_LIST_GLOBAL", "FAMILY_ID_LIST_GLOBAL"] else 0

    out = tmp[base_cols].copy()
    out = sort_by_variant(out)
    safe_save(out, out_csv_path, sep=",")





def add_segregation_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["SEGREGATION_SCORE"] = 0

    out.loc[
        (out["N_CANCER_WITH_VARIANT"] >= 2) &
        (out["N_NONCANCER_WITH_VARIANT"] == 0),
        "SEGREGATION_SCORE"
    ] = 4

    out.loc[
        (out["N_CANCER_WITH_VARIANT"] == 1) &
        (out["N_NONCANCER_WITH_VARIANT"] == 0),
        "SEGREGATION_SCORE"
    ] = 3

    out.loc[
        (out["N_CANCER_WITH_VARIANT"] >= 1),
        "SEGREGATION_SCORE"
    ] = 2

    out.loc[
        (out["N_PATIENTS_WITH_VARIANT"] == 1),
        "SEGREGATION_SCORE"
    ] = 1

    return out

def add_family_evidence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    FAMILY_EVIDENCE_SCORE (0..4) = composite family-level plausibility score.
    Uses columns already present in FAMILY_VARIANTS or created by prepare_integrated_inputs_family().

    Intended meaning:
      0-1: little family support
      2  : weak/moderate support (often single case or partial evidence)
      3  : moderate family evidence (used in integrated score as +1)
      4  : strong family evidence (used in integrated score as +2)

    Contradiction rule:
      if N_NONCANCER_WITH_VARIANT > 0 -> cap score at 2 (prevents high family evidence with unaffected carriers).
    """
    out = df.copy()

    def num(col, default=0):
        return pd.to_numeric(out.get(col, default), errors="coerce").fillna(default)

    n_cancer = num("N_CANCER_WITH_VARIANT", 0).astype(int)
    n_non    = num("N_NONCANCER_WITH_VARIANT", 0).astype(int)
    n_both   = num("N_GERM_AND_SOM", 0).astype(int)
    sirius   = num("FAM_SIRIUS_FLAG", 0).astype(int)

    two_hit_strict = num("TWO_HIT_STRICT", 0).astype(int)
    two_hit_ev     = num("TWO_HIT_EVIDENCE", 0).astype(int)
    loh            = num("LOH_PARTIAL_PATHO", 0).astype(int)

    score = pd.Series(0, index=out.index, dtype=int)

    score += (n_cancer == 1).astype(int) * 1
    score += (n_cancer >= 2).astype(int) * 2

    score += (n_both >= 1).astype(int) * 1

    score += (two_hit_strict >= 1).astype(int) * 2
    score += ((two_hit_strict == 0) & ((two_hit_ev >= 1) | (loh >= 1))).astype(int) * 1

    score += (sirius >= 1).astype(int) * 1

    score = score.clip(0, 4)
    score = score.where(n_non == 0, score.clip(0, 2))

    out["FAMILY_EVIDENCE_SCORE"] = score.astype(int)
    return out


def add_pathway_flags_from_annotations_long_origin_aware(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PATHWAY_HIT and PATH_DNA_REPAIR on FAMILY_LONG while ORIGIN is still available.
    For each row, we build a text blob from pathway-related columns with priority:
      EXOME_* columns > non-prefixed columns.
    Works for both germline and somatic rows.
    """
    out = df_long.copy()

    if "ORIGIN" not in out.columns:
        raise ValueError("Expected ORIGIN in LONG table for origin-aware pathway flags.")

    def norm_text(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        s = str(x).strip()
        if s in {"", ".", "NA", "nan", "None"}:
            return ""
        return s.lower()

    col_candidates = [
        ("EXOME_KEGG_Pathway", "KEGG_Pathway"),
        ("EXOME_GO_biological_process", "GO_biological_process"),
        ("EXOME_Function_description", "Function_description"),
        ("EXOME_Disease_description", "Disease_description"),
    ]

    texts = []
    for i in range(len(out)):
        parts = []
        for ex_col, raw_col in col_candidates:
            v = ""
            if ex_col in out.columns:
                v = norm_text(out.iloc[i][ex_col])
            if (not v) and (raw_col in out.columns):
                v = norm_text(out.iloc[i][raw_col])
            if v:
                parts.append(v)
        texts.append(" | ".join(parts))
    text_series = pd.Series(texts, index=out.index)

    import re
    dna_repair_re = re.compile(
        r"(dna repair|dna damage|response to dna damage|double[- ]strand break|homologous recombination|"
        r"mismatch repair|nucleotide excision repair|base excision repair|non[- ]homologous end joining|"
        r"fanconi|genome stability|checkpoint)",
        re.IGNORECASE
    )
    cancer_path_re = re.compile(
        r"(p53|pi3k|akt|mtor|mapk|ras|wnt|tgf[- ]?beta|notch|hippo|jak[- ]stat|cell cycle|apoptosis|"
        r"dna repair|dna damage|homologous recombination|mismatch repair|excisi?on repair|fanconi)",
        re.IGNORECASE
    )

    out["PATH_DNA_REPAIR"] = text_series.str.contains(dna_repair_re, na=False).astype(int)
    out["PATHWAY_HIT"] = text_series.str.contains(cancer_path_re, na=False).astype(int)

    return out



def add_scientific_integrated_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces:
      - INTEGRATED_SCORE (0..12)
      - INTEGRATED_CLASS (LOW/MODERATE/HIGH/TOP)
      - INTEGRATED_EVIDENCE (human-readable reasons)

    Assumes columns may be missing; handles robustly.
    """
    out = df.copy()

    def num(col, default=0):
        return pd.to_numeric(out.get(col, default), errors="coerce").fillna(default)

    germ_pass = (num("GERM_PASS", 0) >= 1).astype(int)

    germ_tier = num("GERM_TIER", np.nan)
    germ_tier_pts = pd.Series(0, index=out.index)
    germ_tier_pts[(germ_tier == 1)] = 2   # Tier 1 strongest
    germ_tier_pts[(germ_tier == 2)] = 1   # Tier 2

    germ_pts = 2 * germ_pass + germ_tier_pts

    two_hit_strict = (num("TWO_HIT_STRICT", 0) >= 1).astype(int)
    two_hit_ev     = (num("TWO_HIT_EVIDENCE", 0) >= 1).astype(int)

    twohit_pts = 3 * two_hit_strict + 1 * ((two_hit_strict == 0) & (two_hit_ev == 1)).astype(int)

    seg = num("SEGREGATION_SCORE", np.nan)
    seg_pts = pd.Series(0, index=out.index)
    seg_pts[seg >= 4] = 3
    seg_pts[seg == 3] = 2
    seg_pts[seg == 2] = 1

    fam_ev = num("FAMILY_EVIDENCE_SCORE", np.nan)
    fam_ev_pts = pd.Series(0, index=out.index)
    fam_ev_pts[fam_ev >= 4] = 2
    fam_ev_pts[fam_ev == 3] = 1

    ge2 = (num("COUNT_GE_2__LABELS", 0) >= 1).astype(int)
    ge2_pts = ((seg_pts == 0) & (ge2 == 1)).astype(int) * 1

    family_pts = seg_pts + fam_ev_pts + ge2_pts

    pathway_hit = (num("PATHWAY_HIT", 0) >= 1).astype(int)
    pathway_pts = 1 * pathway_hit

    if "PATH_DNA_REPAIR" in out.columns:
        pathway_pts += 1 * (num("PATH_DNA_REPAIR", 0) >= 1).astype(int)

    op = out.get("ORIGIN_PROFILE", pd.Series([""] * len(out), index=out.index)).astype(str).str.lower()
    origin_mod = pd.Series(0, index=out.index)
    origin_mod[op.eq("both")] = 1
    origin_mod[op.eq("somatic-only")] = 0

    out["INTEGRATED_SCORE"] = (germ_pts + twohit_pts + family_pts + pathway_pts + origin_mod).clip(0, 12).astype(int)

    out["INTEGRATED_CLASS"] = "LOW"
    out.loc[out["INTEGRATED_SCORE"] >= 4, "INTEGRATED_CLASS"] = "MODERATE"
    out.loc[out["INTEGRATED_SCORE"] >= 7, "INTEGRATED_CLASS"] = "HIGH"
    out.loc[out["INTEGRATED_SCORE"] >= 9, "INTEGRATED_CLASS"] = "TOP"


    reasons = []
    for i in range(len(out)):
        r = []
        if germ_pass.iat[i] == 1:
            r.append("GERM_PASS")
        if int(germ_tier_pts.iat[i]) > 0:
            r.append(f"GERM_TIER+{int(germ_tier_pts.iat[i])}")
        if two_hit_strict.iat[i] == 1:
            r.append("TWO_HIT_STRICT")
        elif two_hit_ev.iat[i] == 1:
            r.append("TWO_HIT_EVIDENCE")
        if int(seg_pts.iat[i]) > 0:
            r.append(f"SEG+{int(seg_pts.iat[i])}")
        if int(fam_ev_pts.iat[i]) > 0:
            r.append(f"FAM_EVD+{int(fam_ev_pts.iat[i])}")
        if int(ge2_pts.iat[i]) > 0:
            r.append("COUNT_GE_2")
        if pathway_hit.iat[i] == 1:
            r.append("PATHWAY_HIT")
        if int(origin_mod.iat[i]) != 0:
            r.append(f"ORIGIN_MOD{int(origin_mod.iat[i]):+d}")
        reasons.append("|".join(r))
    out["INTEGRATED_EVIDENCE"] = reasons

    return out


def prepare_integrated_inputs_family(df_fam_var: pd.DataFrame) -> pd.DataFrame:
    """
    Create reasonable proxy columns expected by add_scientific_integrated_labels()
    using columns that exist in this pipeline. Missing columns are OK.
    """
    out = df_fam_var.copy()

    def numcol(c, default=0):
        return pd.to_numeric(out.get(c, default), errors="coerce").fillna(default)

    if "COUNT_GE_2__LABELS" not in out.columns:
        out["COUNT_GE_2__LABELS"] = (numcol("FAM_COUNT_GE_2", 0) >= 1).astype(int)

    if "GERM_PASS" not in out.columns:
        if "PAT_SIRIUS_GERM_SNV_PASS" in out.columns:
            out["GERM_PASS"] = (numcol("PAT_SIRIUS_GERM_SNV_PASS", 0) >= 1).astype(int)
        else:
            out["GERM_PASS"] = (numcol("ACMG_INT_GERM", 0) >= 4).astype(int)

    if "GERM_TIER" not in out.columns:
        ag = numcol("ACMG_INT_GERM", np.nan)
        out["GERM_TIER"] = np.nan
        out.loc[ag == 5, "GERM_TIER"] = 1
        out.loc[ag == 4, "GERM_TIER"] = 2

    if "TWO_HIT_STRICT" not in out.columns:
        out["TWO_HIT_STRICT"] = (numcol("two_hit_flag", 0) >= 1).astype(int)
    if "TWO_HIT_EVIDENCE" not in out.columns:
        out["TWO_HIT_EVIDENCE"] = ((numcol("LOH_PARTIAL_PATHO", 0) >= 1) | (numcol("PAT_SIRIUS_CNV_PASS", 0) >= 1)).astype(int)

    if "ORIGIN_PROFILE" not in out.columns:
        n_germ_only = numcol("N_GERM_ONLY", 0)
        n_germ_som  = numcol("N_GERM_AND_SOM", 0)
        # If any germ_and_som -> "both"; else if any germ_only -> "germline-only"; else -> "somatic-only" (most likely)
        op = pd.Series(["somatic-only"] * len(out), index=out.index)
        op[(n_germ_only > 0) & (n_germ_som == 0)] = "germline-only"
        op[(n_germ_som > 0)] = "both"
        out["ORIGIN_PROFILE"] = op

    return out


# =========================================================
# ROH (.hom.plink.txt) handling
# =========================================================

def load_hom_plink(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python", dtype=str)
    if "CHR" in df.columns:
        df["CHR"] = df["CHR"].apply(normalize_chr)
    for c in ["POS1", "POS2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["PHOM", "PHET"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def annotate_variants_with_hom(variants: pd.DataFrame, hom: pd.DataFrame) -> pd.DataFrame:
    out = variants.copy()
    out["IN_ROH"] = 0
    out["HET_HIGH_HOM"] = 0
    if variants.empty or hom is None or hom.empty:
        return out

    hom_by_chr = {}
    for chr_, g in hom.groupby("CHR", dropna=False):
        hom_by_chr[str(chr_)] = g.reset_index(drop=True)

    in_roh = np.zeros(len(out), dtype=int)
    high_hom = np.zeros(len(out), dtype=int)

    chrs = out["CHROM"].astype(str).apply(normalize_chr).values
    poss = pd.to_numeric(out["POS"], errors="coerce").values

    for i in range(len(out)):
        c = str(chrs[i])
        p = poss[i]
        if pd.isna(p):
            continue
        segs = hom_by_chr.get(c)
        if segs is None or segs.empty:
            continue
        hit = segs[(segs["POS1"] <= p) & (segs["POS2"] >= p)]
        if hit.empty:
            continue
        in_roh[i] = 1
        hh = ((hit.get("PHOM", 0) >= 0.95) & (hit.get("PHET", 1) <= 0.05)).any()
        high_hom[i] = 1 if hh else 0

    out["IN_ROH"] = in_roh
    out["HET_HIGH_HOM"] = high_hom
    return out


# =========================================================
# CNV handling (AnnotSV TSV)
# =========================================================

def load_cnv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    for c in ["SV_chrom", "CHROM", "chr", "CHR"]:
        if c in df.columns:
            df[c] = df[c].apply(normalize_chr)
    if "SV_chrom" not in df.columns:
        for c in ["CHROM", "chr", "CHR"]:
            if c in df.columns:
                df = df.rename(columns={c: "SV_chrom"})
                break
    for c in ["SV_start", "SV_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def classify_loh_row(cnv_row: dict) -> str:
    maf = safe_float(cnv_row.get("MINOR_ALLELE_FRACTION"))
    cn = safe_float(cnv_row.get("Calculated_Copy_Number"))
    log2 = safe_float(cnv_row.get("MEAN_LOG2_COPY_RATIO"))

    if maf is None or cn is None or log2 is None:
        return "Unknown"

    if cn <= 1.2 and maf < 0.05 and log2 < -0.35:
        return "Deletion_LOH"
    if 1.7 <= cn <= 2.3 and maf < 0.05 and abs(log2) < 0.3:
        return "CopyNeutral_LOH"
    if maf < 0.1 and cn <= 2.3:
        return "Possible_Subclonal_LOH"
    if maf >= 0.1:
        return "No_LOH"
    return "Possible_LOH"

def is_pathogenic_cnv(cnv_row: dict) -> int:
    r = cnv_row.get("AnnotSV_ranking", None)
    if r is None or pd.isna(r):
        return 0
    s = str(r).strip()
    if s in {"", ".", "NA", "nan", "None"}:
        return 0
    try:
        ri = int(s)
    except Exception:
        return 0
    return 1 if ri >= 3 else 0

def match_cnv_to_variants(variants: pd.DataFrame, cnv: pd.DataFrame) -> pd.DataFrame:
    out = variants.copy()
    out["CNV_LOH_status"] = "Unknown"
    out["CNV_pathogenic"] = 0

    if variants.empty or cnv is None or cnv.empty:
        out["LOH_PARTIAL_PATHO"] = 0
        return out

    cnv = cnv.copy()
    if "SV_chrom" not in cnv.columns or "SV_start" not in cnv.columns or "SV_end" not in cnv.columns:
        out["LOH_PARTIAL_PATHO"] = 0
        return out

    cnv_by_chr = {}
    for chr_, g in cnv.groupby("SV_chrom", dropna=False):
        cnv_by_chr[str(chr_)] = g.reset_index(drop=True)

    chrs = out["CHROM"].astype(str).apply(normalize_chr).values
    poss = pd.to_numeric(out["POS"], errors="coerce").values

    loh_status = []
    patho_flag = []

    for i in range(len(out)):
        c = str(chrs[i])
        p = poss[i]
        if pd.isna(p):
            loh_status.append("Unknown")
            patho_flag.append(0)
            continue
        g = cnv_by_chr.get(c)
        if g is None or g.empty:
            loh_status.append("Unknown")
            patho_flag.append(0)
            continue
        hit = g[(g["SV_start"] <= p) & (g["SV_end"] >= p)]
        if hit.empty:
            loh_status.append("Unknown")
            patho_flag.append(0)
            continue
        row = hit.iloc[0].to_dict()
        loh_status.append(classify_loh_row(row))
        patho_flag.append(is_pathogenic_cnv(row))

    out["CNV_LOH_status"] = loh_status
    out["CNV_pathogenic"] = pd.Series(pd.to_numeric(patho_flag, errors="coerce"), index=out.index).fillna(0).astype(int)

    partial_loh = out["CNV_LOH_status"].isin(["Deletion_LOH", "CopyNeutral_LOH", "Possible_Subclonal_LOH", "Possible_LOH"])
    out["LOH_PARTIAL_PATHO"] = (partial_loh & (out["CNV_pathogenic"] == 1)).astype(int)
    return out


# =========================================================
# VCF parsing (single sample)
# =========================================================

def compute_vaf_from_ad(ad, alt_index: int) -> float:
    try:
        if ad is None:
            return np.nan
        ad = np.array(ad).astype(float)
        if ad.ndim == 0:
            return np.nan
        if len(ad) < alt_index + 1:
            return np.nan
        ref = ad[0]
        alt = ad[alt_index]
        tot = ref + alt
        return float(alt / tot) if tot > 0 else np.nan
    except Exception:
        return np.nan

def gt_to_alleles(ref: str, alts: List[str], gt_tuple) -> Tuple[str, str]:
    if gt_tuple is None or len(gt_tuple) < 2:
        return "", ""
    a1, a2 = gt_tuple[0], gt_tuple[1]
    allele_map = [ref] + (alts or [])
    def idx_to_base(idx):
        try:
            if idx is None or idx < 0:
                return ""
            if idx < len(allele_map):
                return str(allele_map[idx])
            return ""
        except Exception:
            return ""
    return idx_to_base(a1), idx_to_base(a2)

def parse_one_vcf(vcf_path: str) -> pd.DataFrame:
    vcf = VCF(vcf_path)
    samples = list(vcf.samples)
    if len(samples) != 1:
        raise ValueError(f"Expected single-sample VCF, got {len(samples)} samples in {vcf_path}")
    sample = samples[0]

    rows = []
    for var in vcf:
        chrom = normalize_chr(var.CHROM)
        pos = int(var.POS)
        ref = var.REF
        alts = list(var.ALT) if var.ALT is not None else []
        alt_list_str = ",".join([str(a) for a in alts]) if alts else ""

        gt = var.genotypes[0] if var.genotypes is not None and len(var.genotypes) else None
        if gt is not None and len(gt) >= 3:
            a1, a2, phased = gt[0], gt[1], bool(gt[2])
            sep = "|" if phased else "/"
            gt_str = f"{a1}{sep}{a2}"
        else:
            gt_str = None

        allele_1, allele_2 = gt_to_alleles(ref, alts, gt)

        dp = np.nan
        try:
            dpv = var.format("DP")
            if dpv is not None:
                dp = float(dpv[0][0]) if np.array(dpv[0]).size else np.nan
        except Exception:
            dp = np.nan

        ad = None
        try:
            adv = var.format("AD")
            if adv is not None:
                ad = adv[0]
        except Exception:
            ad = None

        for k, alt in enumerate(alts, start=1):
            vaf = compute_vaf_from_ad(ad, alt_index=k)
            info_af = var.INFO.get("AF")
            if isinstance(info_af, (list, tuple)) and len(info_af) >= k:
                info_af_val = info_af[k - 1]
            else:
                info_af_val = info_af

            rows.append({
                "CHROM": chrom,
                "POS": pos,
                "REF": ref,
                "ALT": alt,
                "ALT_LIST": alt_list_str,
                "SAMPLE": sample,
                "GT": gt_str,
                "allele_1": allele_1,
                "allele_2": allele_2,
                "DP": dp,
                "VAF": vaf,
                "INFO_AF": info_af_val,
                "QUAL": float(var.QUAL) if var.QUAL is not None else np.nan,
                "FILTER": "PASS" if var.FILTER is None else str(var.FILTER),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["CHROM"] = df["CHROM"].apply(normalize_chr)
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
        df["DP"] = pd.to_numeric(df["DP"], errors="coerce")
        df["VAF"] = pd.to_numeric(df["VAF"], errors="coerce")
        df["INFO_AF"] = pd.to_numeric(df["INFO_AF"], errors="coerce")
    return df


# =========================================================
# Exome merge
# =========================================================

def infer_exome_path_from_vcf(vcf_path: str) -> Optional[str]:
    base = os.path.basename(vcf_path).replace(".vcf.gz", "").replace(".vcf", "")
    exome_base = base.replace("_Genetic_Variants", "_Exome_Genetic_Variants")
    folder = os.path.dirname(vcf_path)
    for ext in [".txt", ".tsv"]:
        p = os.path.join(folder, exome_base + ext)
        if os.path.exists(p):
            return p
    return None

def load_exome_table(exome_path: str) -> pd.DataFrame:
    if not exome_path or (not os.path.exists(exome_path)):
        return pd.DataFrame()
    try:
        df = pd.read_csv(exome_path, sep="\t", dtype=str, low_memory=False)
    except Exception:
        df = pd.read_csv(exome_path, sep=r"\s+", engine="python", dtype=str, low_memory=False)

    if "CHROM" in df.columns:
        df["CHROM"] = df["CHROM"].apply(normalize_chr)
    elif "chr" in df.columns:
        df = df.rename(columns={"chr": "CHROM"})
        df["CHROM"] = df["CHROM"].apply(normalize_chr)

    if "POS" in df.columns:
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    elif "pos" in df.columns:
        df = df.rename(columns={"pos": "POS"})
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    return df

def merge_exome_into_variants(variants_df: pd.DataFrame, exome_df: pd.DataFrame) -> pd.DataFrame:
    if exome_df is None or exome_df.empty:
        return variants_df

    v = variants_df.copy()
    e = exome_df.copy()

    keys_full = ["CHROM", "POS", "REF", "ALT"]
    keys_pos = ["CHROM", "POS"]

    if all(k in v.columns for k in keys_full) and all(k in e.columns for k in keys_full):
        join_keys = keys_full
    elif all(k in v.columns for k in keys_pos) and all(k in e.columns for k in keys_pos):
        join_keys = keys_pos
    else:
        return v

    e = e.drop_duplicates(subset=join_keys, keep="first").copy()

    payload = [c for c in e.columns if c not in join_keys]
    e2 = e[join_keys + payload].copy()
    e2 = e2.rename(columns={c: f"EXOME_{c}" for c in payload})

    out = v.merge(e2, on=join_keys, how="left")
    return out


# =========================================================
# Sirius flags (row-level on FAMILY_LONG)
# =========================================================

def parse_acmg_to_int(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", ".", "NA", "nan", "None"}:
        return np.nan
    if s.isdigit():
        return int(s)
    sl = s.lower()
    if "pathogenic" in sl and "likely" not in sl:
        return 5
    if "likely_pathogenic" in sl or ("likely" in sl and "pathog" in sl):
        return 4
    if "uncertain" in sl or "vus" in sl:
        return 3
    if "likely_benign" in sl or ("likely" in sl and "benign" in sl):
        return 2
    if "benign" in sl:
        return 1
    if "non" in sl and "class" in sl:
        return 0
    return np.nan

def compute_max_pop_af(df: pd.DataFrame) -> pd.Series:
    af_cols = [
        "INFO_AF",
        "IG_AF", "MAX_AF",
        "ANN_gnomAD_AF", "gnomAD_Global_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF",
        "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF",
        "1000G_Global_AF", "1000G_AFR_AF", "1000G_AMR_AF", "1000G_EAS_AF", "1000G_EUR_AF", "1000G_SAS_AF",
        "Kaviar_AF", "EVS_EA_MAF", "EVS_CA",
        "EXOME_IG_AF", "EXOME_MAX_AF", "EXOME_gnomAD_Global_AF", "EXOME_1000G_Global_AF", "EXOME_Kaviar_AF",
    ]
    present = [c for c in af_cols if c in df.columns]
    if not present:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[present].apply(_to_num).max(axis=1, skipna=True)


def consequence_pass_coding_nonsyn(df: pd.DataFrame) -> pd.Series:
    cons_col = "Consequence"
    if cons_col not in df.columns and "EXOME_Consequence" in df.columns:
        cons_col = "EXOME_Consequence"

    if cons_col not in df.columns:
        return pd.Series([1] * len(df), index=df.index)

    cons = df[cons_col].astype(str).str.lower()

    is_syn = cons.str.contains("synonymous_variant", na=False)

    coding_hits = (
        cons.str.contains("missense_variant", na=False) |
        cons.str.contains("stop_gained", na=False) |
        cons.str.contains("frameshift_variant", na=False) |
        cons.str.contains("splice_acceptor_variant", na=False) |
        cons.str.contains("splice_donor_variant", na=False) |
        cons.str.contains("start_lost", na=False) |
        cons.str.contains("stop_lost", na=False) |
        cons.str.contains("inframe_insertion", na=False) |
        cons.str.contains("inframe_deletion", na=False) |
        cons.str.contains("protein_altering_variant", na=False)
    )

    return ((~is_syn) & coding_hits).astype(int)




def add_sirius_pass_flags_family_long(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    if "ORIGIN" not in df.columns:
        raise ValueError("LONG must contain ORIGIN (germline/somatic).")

    base_candidates = [
        "EXOME_Prediction_ACMG_tapes", "EXOME_Prediction_ACMG",
        "Prediction_ACMG_tapes", "Prediction_ACMG",
        "EXOME_Classification_ACMG", "Classification_ACMG",
    ]


    def pick_first_existing(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in df.columns:
                return c
        return None

    germ_candidates = [f"GERM_{c}" for c in base_candidates] + base_candidates
    som_candidates  = [f"SOM_{c}"  for c in base_candidates] + base_candidates

    acmg_col_germ = pick_first_existing(germ_candidates)
    acmg_col_som  = pick_first_existing(som_candidates)

    is_germ = df["ORIGIN"].astype(str).str.lower().eq("germline")
    is_som  = df["ORIGIN"].astype(str).str.lower().eq("somatic")

    df["ACMG_INT_GERM"] = np.nan
    df["ACMG_INT_SOM"]  = np.nan

    if acmg_col_germ is not None:
        df.loc[is_germ, "ACMG_INT_GERM"] = df.loc[is_germ, acmg_col_germ].apply(parse_acmg_to_int)

    if acmg_col_som is not None:
        df.loc[is_som, "ACMG_INT_SOM"] = df.loc[is_som, acmg_col_som].apply(parse_acmg_to_int)

    df["ACMG_INT"] = np.nan
    df.loc[is_germ, "ACMG_INT"] = df.loc[is_germ, "ACMG_INT_GERM"]
    df.loc[is_som,  "ACMG_INT"] = df.loc[is_som,  "ACMG_INT_SOM"]


    df["SIRIUS_PASS_ACMG_SNV"] = 0
    df["SIRIUS_PASS_ACMG_CNV"] = 0

    df.loc[is_germ, "SIRIUS_PASS_ACMG_SNV"] = df.loc[is_germ, "ACMG_INT_GERM"].isin([2, 3, 4, 5]).astype(int)
    df.loc[is_germ, "SIRIUS_PASS_ACMG_CNV"] = df.loc[is_germ, "ACMG_INT_GERM"].isin([4, 5]).astype(int)

    df.loc[is_som, "SIRIUS_PASS_ACMG_SNV"] = df.loc[is_som, "ACMG_INT_SOM"].isin([2, 3, 4, 5]).astype(int)
    df.loc[is_som, "SIRIUS_PASS_ACMG_CNV"] = df.loc[is_som, "ACMG_INT_SOM"].isin([4, 5]).astype(int)


    df["MAX_POP_AF"] = compute_max_pop_af(df)
    df["SIRIUS_PASS_POPAF"] = ((df["MAX_POP_AF"].isna()) | (df["MAX_POP_AF"] < 0.01)).astype(int)
    df["SIRIUS_PASS_CODING_NONSYN"] = consequence_pass_coding_nonsyn(df)

    df["DP"] = pd.to_numeric(df.get("DP", np.nan), errors="coerce")
    df["VAF"] = pd.to_numeric(df.get("VAF", np.nan), errors="coerce")

    pass_dp_g = (df["DP"] > 10).fillna(False)
    pass_vaf_g = (df["VAF"] > 0.25).fillna(False)
    pass_dp_s = (df["DP"] > 20).fillna(False)
    pass_vaf_s = (df["VAF"] > 0.05).fillna(False)

    base_req = ["SIRIUS_PASS_ACMG_SNV", "SIRIUS_PASS_POPAF", "SIRIUS_PASS_CODING_NONSYN"]
    base_ok = (df[base_req].sum(axis=1) == len(base_req))

    df["SIRIUS_PASS_SNV_GERMLINE"] = (is_germ & base_ok & pass_dp_g & pass_vaf_g).astype(int)
    df["SIRIUS_PASS_SNV_SOMATIC"] = (is_som & base_ok & pass_dp_s & pass_vaf_s).astype(int)

    if "LOH_PARTIAL_PATHO" not in df.columns:
        df["LOH_PARTIAL_PATHO"] = 0
    df["LOH_PARTIAL_PATHO"] = pd.to_numeric(df["LOH_PARTIAL_PATHO"], errors="coerce").fillna(0).astype(int)

    df["SIRIUS_PASS_CNV"] = ((df["LOH_PARTIAL_PATHO"] == 1) & (df["SIRIUS_PASS_ACMG_CNV"] == 1)).astype(int)

    df["SIRIUS_FLAG_ROW"] = (
        (df["SIRIUS_PASS_SNV_GERMLINE"] == 1) |
        (df["SIRIUS_PASS_SNV_SOMATIC"] == 1) |
        (df["SIRIUS_PASS_CNV"] == 1)
    ).astype(int)

    return df




def is_positive_value(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip()
    if s in {"", ".", "NA", "nan", "None"}:
        return False
    try:
        if float(s) == 0.0:
            return False
    except Exception:
        pass
    if s == "0":
        return False
    return True

def any_positive(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([0] * len(df), index=df.index)
    out = np.zeros(len(df), dtype=int)
    for c in present:
        out |= df[c].apply(is_positive_value).astype(int).values
    return pd.Series(out, index=df.index)

def literature_keyword_flag(df: pd.DataFrame) -> pd.Series:
    text_cols = [c for c in [
        "DisGeNET", "PUBMED", "Disease_description",
        "ClinVar_CLNDN", "ClinVar_CLNDISDB", "Trait_association(GWAS)",
        "EXOME_DisGeNET", "EXOME_PUBMED", "EXOME_Disease_description",
        "EXOME_ClinVar_CLNDN", "EXOME_ClinVar_CLNDISDB", "EXOME_Trait_association(GWAS)",
    ] if c in df.columns]
    if not text_cols:
        return pd.Series([0] * len(df), index=df.index)

    patt = re.compile(r"(cancer|tumou?r|carcinoma|leukemia|lymphoma|sarcoma|melanoma)", re.IGNORECASE)
    hit = pd.Series([False] * len(df), index=df.index)
    for c in text_cols:
        hit = hit | df[c].astype(str).str.contains(patt, na=False)
    return hit.astype(int)

def compute_patient_internal_flags(df_pat: pd.DataFrame) -> pd.DataFrame:
    df = df_pat.copy()

    for c in ["HAS_GERM", "HAS_SOM", "ACMG_INT_GERM", "ACMG_INT_SOM", "LOH_PARTIAL_PATHO",
              "PAT_SIRIUS_GERM_SNV_PASS", "PAT_SIRIUS_SOM_SNV_PASS", "PAT_SIRIUS_CNV_PASS"]:
        if c not in df.columns:
            df[c] = 0

    df["HAS_GERM"] = pd.to_numeric(df["HAS_GERM"], errors="coerce").fillna(0).astype(int)
    df["HAS_SOM"] = pd.to_numeric(df["HAS_SOM"], errors="coerce").fillna(0).astype(int)
    df["ACMG_INT_GERM"] = pd.to_numeric(df["ACMG_INT_GERM"], errors="coerce")
    df["ACMG_INT_SOM"] = pd.to_numeric(df["ACMG_INT_SOM"], errors="coerce")
    df["LOH_PARTIAL_PATHO"] = pd.to_numeric(df["LOH_PARTIAL_PATHO"], errors="coerce").fillna(0).astype(int)

    cancer_gene_cols = [
        "Cancer_Genes", "Cancer_Genes_Oncokb", "Cancer_Genes_OncoKB", "GENE_IN_ONCOKB",
        "Gene_Cancer_Status",
        "EXOME_Cancer_Genes", "EXOME_Cancer_Genes_Oncokb", "EXOME_GENE_IN_ONCOKB", "EXOME_Gene_Cancer_Status",
    ]
    has_cancer_gene = any_positive(df, cancer_gene_cols)

    df["predisposition_flag"] = (
        (df["HAS_GERM"] == 1) &
        (df["ACMG_INT_GERM"].isin([4, 5])) &
        (has_cancer_gene == 1)
    ).astype(int)

    driver_cols = [
        "ONCOGENIC", "HIGHEST_LEVEL", "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4",
        "LEVEL_R1", "LEVEL_R2", "LEVEL_R3",
        "IS-A-HOTSPOT", "IS-A-3D-HOTSPOT",
        "VARIANT_IN_ONCOKB", "MUTATION_EFFECT",
        "CIViC_Variant_clinical_significance", "CIViC_Region_clinical_significance",
        "COSMIC", "COSMIC_FATHMM", "Hotspots_Kit",
        "EXOME_ONCOGENIC", "EXOME_HIGHEST_LEVEL", "EXOME_LEVEL_1", "EXOME_LEVEL_2", "EXOME_LEVEL_3A",
        "EXOME_LEVEL_3B", "EXOME_LEVEL_4", "EXOME_IS-A-HOTSPOT", "EXOME_IS-A-3D-HOTSPOT",
        "EXOME_CIViC_Variant_clinical_significance", "EXOME_CIViC_Region_clinical_significance",
        "EXOME_COSMIC", "EXOME_COSMIC_FATHMM", "EXOME_Hotspots_Kit",
    ]
    has_driver = any_positive(df, driver_cols)
    df["somatic_driver_flag"] = ((df["HAS_SOM"] == 1) & (has_driver == 1)).astype(int)

    hit_sum = (
        pd.to_numeric(df["PAT_SIRIUS_GERM_SNV_PASS"], errors="coerce").fillna(0).astype(int) +
        pd.to_numeric(df["PAT_SIRIUS_SOM_SNV_PASS"], errors="coerce").fillna(0).astype(int) +
        pd.to_numeric(df["PAT_SIRIUS_CNV_PASS"], errors="coerce").fillna(0).astype(int)
    )
    second_path_snv = (hit_sum >= 2).astype(int)

    df["two_hit_flag"] = (
        (df["HAS_GERM"] == 1) &
        (df["HAS_SOM"] == 1) &
        ((df["LOH_PARTIAL_PATHO"] == 1) | (second_path_snv == 1))
    ).astype(int)

    df["literature_flag"] = literature_keyword_flag(df)

    score = (
        1 * df["predisposition_flag"] +
        2 * df["somatic_driver_flag"] +
        2 * df["two_hit_flag"] +
        1 * df["literature_flag"]
    )
    df["patient_cancer_evidence_score"] = score.clip(lower=0, upper=5).astype(int)
    df["patient_cancer_evidence_class"] = df["patient_cancer_evidence_score"].astype(int)
    return df


def process_source_one_patient(
    family_id: str,
    patient_id: str,
    origin: str,
    vcf_path: str,
    cnv_path: Optional[str],
    hom_path: Optional[str],
    has_somatic_patient: int,
    enable_exome: bool = True,
) -> pd.DataFrame:
    df = parse_one_vcf(vcf_path)

    if enable_exome:
        exome_path = infer_exome_path_from_vcf(vcf_path)
        if exome_path:
            ex = load_exome_table(exome_path)
            df = merge_exome_into_variants(df, ex)
    df = normalize_acmg_by_source(df, origin)
    if cnv_path and os.path.exists(cnv_path):
        cnv = load_cnv(cnv_path)
        df = match_cnv_to_variants(df, cnv)
    else:
        df["CNV_LOH_status"] = "Unknown"
        df["CNV_pathogenic"] = 0
        df["LOH_PARTIAL_PATHO"] = 0

    if hom_path and os.path.exists(hom_path):
        hom = load_hom_plink(hom_path)
        df = annotate_variants_with_hom(df, hom)
    else:
        df["IN_ROH"] = 0
        df["HET_HIGH_HOM"] = 0

    df["FAMILY_ID"] = family_id
    df["PATIENT_ID"] = patient_id
    df["ORIGIN"] = origin
    df["HAS_SOMATIC_PATIENT"] = int(has_somatic_patient)
    return df


# =========================================================
# JSON helpers
# =========================================================

def pick_one_from_dirs(run: dict, key: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    p = run.get(key)
    if not p:
        return None
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        return list_files_one(p, suffixes)
    return None


# =========================================================
# Patient-level build: germline + all somatic runs collapsed
# =========================================================

def collapse_somatic_runs_to_patient_variants(df_som_long: pd.DataFrame) -> pd.DataFrame:
    if df_som_long is None or df_som_long.empty:
        return pd.DataFrame()

    vkey = get_variant_key(df_som_long)
    group_cols = ["FAMILY_ID", "PATIENT_ID"] + vkey

    df = df_som_long.copy()
    df["_ones"] = 1
    other_cols = [c for c in df.columns if c not in group_cols]

    numeric_candidates = set([
        "POS", "DP", "VAF", "QUAL", "INFO_AF",
        "LOH_PARTIAL_PATHO", "CNV_pathogenic", "IN_ROH", "HET_HIGH_HOM",
        "HAS_SOMATIC_PATIENT",
    ])
    flag_candidates = set([
        "LOH_PARTIAL_PATHO", "CNV_pathogenic", "IN_ROH", "HET_HIGH_HOM",
        "HAS_SOMATIC_PATIENT",
    ])

    def first_nonnull(s):
        t = s.dropna()
        return t.iloc[0] if len(t) else np.nan

    agg: Dict[str, object] = {"SOMATIC_REPEAT_COUNT": ("_ones", "sum")}

    for c in other_cols:
        if c == "_ones":
            continue
        if c in flag_candidates:
            agg[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).max())
        elif c in numeric_candidates:
            agg[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").max())
        else:
            agg[c] = (c, first_nonnull)

    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out = out.drop(columns=["_ones"], errors="ignore")
    return out

def build_patient_long(
    family_id: str,
    patient_id: str,
    cfg: dict,
    enable_exome: bool,
) -> pd.DataFrame:
    frames = []
    has_somatic_patient = 1 if (cfg.get("somatic_runs") and len(cfg.get("somatic_runs")) > 0) else 0

    germ = cfg.get("germline_files")
    if germ:
        vcf = germ.get("vcf")
        cnv = germ.get("cnv")
        hom = germ.get("het")
        if not vcf or not os.path.exists(vcf):
            raise FileNotFoundError(f"[{family_id}/{patient_id}] germline vcf missing: {vcf}")

        df_g = process_source_one_patient(
            family_id, patient_id,
            origin="germline",
            vcf_path=vcf,
            cnv_path=cnv if cnv and os.path.exists(cnv) else None,
            hom_path=hom if hom and os.path.exists(hom) else None,
            has_somatic_patient=has_somatic_patient,
            enable_exome=enable_exome
        )
        frames.append(df_g)

    som_frames = []
    for run in cfg.get("somatic_runs", []) or []:
        for vcf_key, cnv_key in [("vcf_genetic_dir", "cnv_genetic_dir"), ("vcf_oncology_dir", "cnv_oncology_dir")]:
            vcf_path = pick_one_from_dirs(run, vcf_key, (".vcf.gz", ".vcf"))
            if vcf_path is None:
                continue
            cnv_path = pick_one_from_dirs(run, cnv_key, (".tsv", ".txt"))
            hom_path = pick_one_from_dirs(run, "het_dir", (".hom.plink.txt", ".plink.txt", ".txt"))

            df_s = process_source_one_patient(
                family_id, patient_id,
                origin="somatic",
                vcf_path=vcf_path,
                cnv_path=cnv_path,
                hom_path=hom_path,
                has_somatic_patient=1,
                enable_exome=enable_exome
            )
            som_frames.append(df_s)

    df_som_long = pd.concat(som_frames, ignore_index=True) if som_frames else pd.DataFrame()
    df_som_pat = collapse_somatic_runs_to_patient_variants(df_som_long)

    if df_som_pat is not None and not df_som_pat.empty:
        df_som_pat["ORIGIN"] = "somatic"
        df_som_pat["HAS_SOMATIC_PATIENT"] = 1
        frames.append(df_som_pat)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# =========================================================
# Collapse to patient-variants (unique per patient-variant)
# =========================================================

def collapse_family_long_to_patient_variants(df_family_long: pd.DataFrame) -> pd.DataFrame:
    if df_family_long is None or df_family_long.empty:
        return pd.DataFrame()

    vkey = get_variant_key(df_family_long)
    group_cols = ["FAMILY_ID", "PATIENT_ID"] + vkey

    df = df_family_long.copy()

    required = [
        "SIRIUS_PASS_SNV_GERMLINE", "SIRIUS_PASS_SNV_SOMATIC", "SIRIUS_PASS_CNV",
        "SIRIUS_FLAG_ROW", "ACMG_INT", "LOH_PARTIAL_PATHO", "PATHWAY_HIT", "PATH_DNA_REPAIR"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = 0

    df["_is_germ"] = (df["ORIGIN"].astype(str) == "germline").astype(int)
    df["_is_som"] = (df["ORIGIN"].astype(str) == "somatic").astype(int)

    df["_acmg_germ"] = np.where(df["_is_germ"] == 1, pd.to_numeric(df["ACMG_INT"], errors="coerce"), np.nan)
    df["_acmg_som"] = np.where(df["_is_som"] == 1, pd.to_numeric(df["ACMG_INT"], errors="coerce"), np.nan)

    def first_nonnull(s):
        t = s.dropna()
        return t.iloc[0] if len(t) else np.nan

    other_cols = [c for c in df.columns if c not in group_cols]

    agg_map: Dict[str, object] = {}
    agg_map["HAS_GERM"] = ("_is_germ", "max")
    agg_map["HAS_SOM"] = ("_is_som", "max")
    agg_map["ACMG_INT_GERM"] = ("_acmg_germ", "max")
    agg_map["ACMG_INT_SOM"] = ("_acmg_som", "max")

    agg_map["PAT_SIRIUS_GERM_SNV_PASS"] = ("SIRIUS_PASS_SNV_GERMLINE", "max")
    agg_map["PAT_SIRIUS_SOM_SNV_PASS"] = ("SIRIUS_PASS_SNV_SOMATIC", "max")
    agg_map["PAT_SIRIUS_CNV_PASS"] = ("SIRIUS_PASS_CNV", "max")
    agg_map["PAT_SIRIUS_FLAG"] = ("SIRIUS_FLAG_ROW", "max")
    

    skip = set(["_is_germ", "_is_som", "_acmg_germ", "_acmg_som"])
    for c in other_cols:
        if c in skip or c in agg_map or c in group_cols:
            continue

        if c in {"LOH_PARTIAL_PATHO", "CNV_pathogenic", "IN_ROH", "HET_HIGH_HOM",
                 "SIRIUS_PASS_SNV_GERMLINE", "SIRIUS_PASS_SNV_SOMATIC", "SIRIUS_PASS_CNV", "SIRIUS_FLAG_ROW", "PATHWAY_HIT", "PATH_DNA_REPAIR"}:
            agg_map[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).max())
        else:
            def _num_max_or_first(x):
                xn = pd.to_numeric(x, errors="coerce")
                if xn.notna().any():
                    return float(xn.max())
                return first_nonnull(x)
            agg_map[c] = (c, _num_max_or_first)

    out = df.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")
    return out


# =========================================================
# Family-variants (unique per family-variant) + labels + INTEGRATED
# =========================================================

def collapse_patient_variants_to_family_variants(df_pat: pd.DataFrame) -> pd.DataFrame:
    if df_pat is None or df_pat.empty:
        return pd.DataFrame()

    vkey = get_variant_key(df_pat)
    key = ["FAMILY_ID"] + vkey
    df = df_pat.copy()

    def uniq_list(series):
        vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
        vals = [v for v in vals if v not in {"", "nan", "None", "."}]
        return ",".join(sorted(vals))

    def agg_block(g):
        g0 = g[["PATIENT_ID", "HAS_SOMATIC_PATIENT", "HAS_GERM", "HAS_SOM", "PAT_SIRIUS_FLAG"]].drop_duplicates()

        n_pat = int(g0["PATIENT_ID"].nunique())
        n_cancer = int(g0.loc[pd.to_numeric(g0["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int) == 1, "PATIENT_ID"].nunique())
        n_noncancer = int(g0.loc[pd.to_numeric(g0["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int) == 0, "PATIENT_ID"].nunique())

        has_germ = pd.to_numeric(g0["HAS_GERM"], errors="coerce").fillna(0).astype(int)
        has_som = pd.to_numeric(g0["HAS_SOM"], errors="coerce").fillna(0).astype(int)

        n_germ_only = int(((has_germ == 1) & (has_som == 0)).sum())
        n_germ_and_som = int(((has_germ == 1) & (has_som == 1)).sum())

        fam_sirius = int(pd.to_numeric(g["PAT_SIRIUS_FLAG"], errors="coerce").fillna(0).max())

        return pd.Series({
            "N_PATIENTS_WITH_VARIANT": n_pat,
            "PATIENT_ID_LIST": uniq_list(g["PATIENT_ID"]),
            "N_CANCER_WITH_VARIANT": n_cancer,
            "N_NONCANCER_WITH_VARIANT": n_noncancer,
            "N_GERM_ONLY": n_germ_only,
            "N_GERM_AND_SOM": n_germ_and_som,
            "FAM_SIRIUS_FLAG": fam_sirius,
        })

    fam_counts = df.groupby(key, dropna=False).apply(agg_block).reset_index()

    def first_nonnull(s):
        t = s.dropna()
        return t.iloc[0] if len(t) else np.nan

    max_cols = [
        "PAT_SIRIUS_GERM_SNV_PASS", "PAT_SIRIUS_SOM_SNV_PASS", "PAT_SIRIUS_CNV_PASS", "PAT_SIRIUS_FLAG",
        "predisposition_flag", "somatic_driver_flag", "two_hit_flag", "literature_flag",
        "patient_cancer_evidence_score", "patient_cancer_evidence_class",
        "LOH_PARTIAL_PATHO", "SOMATIC_REPEAT_COUNT","PATHWAY_HIT", "PATH_DNA_REPAIR",
    ]
    present_max_cols = [c for c in max_cols if c in df.columns]

    payload_cols = [c for c in df.columns if c not in (key + ["PATIENT_ID"])]

    agg_map = {}
    for c in payload_cols:
        if c in present_max_cols:
            agg_map[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).max())
        else:
            def _num_max_or_first(x):
                xn = pd.to_numeric(x, errors="coerce")
                if xn.notna().any():
                    return float(xn.max())
                return first_nonnull(x)
            agg_map[c] = (c, _num_max_or_first)

    fam_payload = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    fam = fam_counts.merge(fam_payload, on=key, how="left")

    fam["label_of_interest"] = (
        (pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 1) &
        (pd.to_numeric(fam["N_NONCANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 0) &
        (pd.to_numeric(fam["FAM_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)

    # >=2 patients block
    fam["FAM_COUNT_GE_2"] = (pd.to_numeric(fam["N_PATIENTS_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 2).astype(int)

    return fam


# =========================================================
# Global unique variants + lists + INTEGRATED
# =========================================================

def build_global_unique_variants(df_fam_all: pd.DataFrame) -> pd.DataFrame:
    if df_fam_all is None or df_fam_all.empty:
        return pd.DataFrame()

    vkey = get_variant_key(df_fam_all)
    key = vkey

    df = df_fam_all.copy()

    def uniq_list(series):
        vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
        vals = [v for v in vals if v not in {"", "nan", "None", "."}]
        return ",".join(sorted(vals))

    if "PATIENT_ID_LIST" not in df.columns:
        df["PATIENT_ID_LIST"] = ""

    lists = df.groupby(key, dropna=False).agg(
        FAMILY_ID_LIST_GLOBAL=("FAMILY_ID", uniq_list),
        PATIENT_ID_LIST_GLOBAL=("PATIENT_ID_LIST", uniq_list),
    ).reset_index()

    def first_nonnull(s):
        t = s.dropna()
        return t.iloc[0] if len(t) else np.nan

    max_cols = [c for c in [
        "FAM_SIRIUS_FLAG", "label_of_interest", "FAM_COUNT_GE_2",
        "INTEGRATED_SCORE",
        "patient_cancer_evidence_class",
        "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT",
        "N_GERM_ONLY", "N_GERM_AND_SOM",
        "SOMATIC_REPEAT_COUNT", "LOH_PARTIAL_PATHO",
        "ACMG_INT_GERM", "ACMG_INT_SOM",
    ] if c in df.columns]

    skip_cols = set(key + ["FAMILY_ID", "PATIENT_ID_LIST", "FAMILY_ID_LIST_GLOBAL", "PATIENT_ID_LIST_GLOBAL"])

    agg_map = {}
    for c in df.columns:
        if c in skip_cols:
            continue
        if c in max_cols:
            agg_map[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).max())
        else:
            def _num_max_or_first(x):
                xn = pd.to_numeric(x, errors="coerce")
                if xn.notna().any():
                    return float(xn.max())
                return first_nonnull(x)
            agg_map[c] = (c, _num_max_or_first)

    payload = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    out = payload.merge(lists, on=key, how="left")
    out = attach_global_lists(out)

    out2 = prepare_integrated_inputs_family(out)  # same proxy logic works here
    out2 = add_scientific_integrated_labels(out2)
    return out2


def export_patient_class_ge_3_unique(df_pat_all: pd.DataFrame, out_tsv_path: str) -> pd.DataFrame:
    if df_pat_all is None or df_pat_all.empty:
        empty = pd.DataFrame()
        safe_save(empty, out_tsv_path, sep="\t")
        return empty

    df = df_pat_all.copy()
    cls = pd.to_numeric(df.get("patient_cancer_evidence_class", 0), errors="coerce").fillna(0).astype(int)
    df = df.loc[cls >= 3].copy()
    if df.empty:
        safe_save(df, out_tsv_path, sep="\t")
        return df

    vkey = get_variant_key(df)
    key = vkey

    def uniq_list(series):
        vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
        vals = [v for v in vals if v not in {"", "nan", "None", "."}]
        return ",".join(sorted(vals))

    df["FAMILY_PATIENT"] = df["FAMILY_ID"].astype(str) + ":" + df["PATIENT_ID"].astype(str)

    def first_nonnull(s):
        t = s.dropna()
        return t.iloc[0] if len(t) else np.nan

    lists = df.groupby(key, dropna=False).agg(
        FAMILY_ID_LIST_GLOBAL=("FAMILY_ID", uniq_list),
        PATIENT_ID_LIST_GLOBAL=("FAMILY_PATIENT", uniq_list),
    ).reset_index()

    max_cols = [c for c in [
        "predisposition_flag", "somatic_driver_flag", "two_hit_flag", "literature_flag",
        "patient_cancer_evidence_score", "patient_cancer_evidence_class",
        "PAT_SIRIUS_FLAG", "PAT_SIRIUS_GERM_SNV_PASS", "PAT_SIRIUS_SOM_SNV_PASS", "PAT_SIRIUS_CNV_PASS",
        "LOH_PARTIAL_PATHO", "SOMATIC_REPEAT_COUNT", "ACMG_INT_GERM", "ACMG_INT_SOM",
    ] if c in df.columns]

    skip = set(key + ["FAMILY_ID", "PATIENT_ID", "FAMILY_PATIENT"])

    agg_map = {}
    for c in df.columns:
        if c in skip:
            continue
        if c in max_cols:
            agg_map[c] = (c, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).max())
        else:
            def _num_max_or_first(x):
                xn = pd.to_numeric(x, errors="coerce")
                if xn.notna().any():
                    return float(xn.max())
                return first_nonnull(x)
            agg_map[c] = (c, _num_max_or_first)

    payload = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    out = payload.merge(lists, on=key, how="left")
    out = attach_global_lists(out)
    out = sort_by_variant(out)

    safe_save(out, out_tsv_path, sep="\t")
    return out


# =========================================================
# Main: process all families
# =========================================================

def process_all_families_from_json(
    json_path: str,
    out_dir: str,
    enable_exome: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        families = json.load(f)
    if not isinstance(families, dict) or not families:
        raise ValueError("JSON must be {family_id: {patient_id: {...}}}")

    fam_long_all = []
    fam_pat_all = []
    fam_var_all = []

    for family_id, fam in families.items():
        if not isinstance(fam, dict):
            continue
        print(f"\n==== FAMILY: {family_id} ====")
        fam_out = os.path.join(out_dir, str(family_id))
        ensure_dir(fam_out)

        patient_longs = []
        for patient_id, cfg in fam.items():
            print(f"  -> patient: {patient_id}")
            try:
                df_p_long = build_patient_long(str(family_id), str(patient_id), cfg, enable_exome=enable_exome)
            except Exception as e:
                print(f"     [ERROR] patient build failed: {type(e).__name__}: {e}")
                continue

            if df_p_long is None or df_p_long.empty:
                print("     [WARN] no rows")
                continue

            p_long_path = os.path.join(fam_out, f"{family_id}__{patient_id}__LONG.tsv")
            safe_save(sort_by_variant(df_p_long), p_long_path, sep="\t")
            patient_longs.append(df_p_long)

        if not patient_longs:
            print("  [WARN] family has no patient rows")
            continue

        df_fam_long = sort_by_variant(pd.concat(patient_longs, ignore_index=True))
        fam_long_path = os.path.join(fam_out, f"{family_id}__ALL_PATIENTS__LONG.tsv")
        safe_save(df_fam_long, fam_long_path, sep="\t")

        df_fam_long = add_sirius_pass_flags_family_long(df_fam_long)

        df_fam_long = add_pathway_flags_from_annotations_long_origin_aware(df_fam_long)

        df_pat = collapse_family_long_to_patient_variants(df_fam_long)

        df_pat["ORIGIN_PROFILE"] = np.select(
            [
                (df_pat["HAS_GERM"] == 1) & (df_pat["HAS_SOM"] == 1),
                (df_pat["HAS_GERM"] == 1) & (df_pat["HAS_SOM"] == 0),
                (df_pat["HAS_GERM"] == 0) & (df_pat["HAS_SOM"] == 1),
            ],
            ["both", "germline-only", "somatic-only"],
            default="unknown"
        )

        df_pat = compute_patient_internal_flags(df_pat)
        df_pat = sort_by_variant(df_pat)

        safe_save(df_pat, os.path.join(fam_out, f"{family_id}__PATIENT_VARIANTS.tsv"), sep="\t")

        df_fam_var = collapse_patient_variants_to_family_variants(df_pat)
        df_fam_var = prepare_integrated_inputs_family(df_fam_var)   # ensures TWO_HIT_* exists
        df_fam_var = add_family_evidence_score(df_fam_var)
        df_fam_var = add_segregation_score(df_fam_var)
        df_fam_var = add_scientific_integrated_labels(df_fam_var)


        df_fam_var = sort_by_variant(df_fam_var)
        safe_save(df_fam_var, os.path.join(fam_out, f"{family_id}__FAMILY_VARIANTS.tsv"), sep="\t")

        sirius_fam = sort_by_variant(df_fam_var.loc[pd.to_numeric(df_fam_var.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
        safe_save(sirius_fam, os.path.join(fam_out, f"{family_id}__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv"), sep="\t")

        loi = sort_by_variant(df_fam_var.loc[pd.to_numeric(df_fam_var.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
        safe_save(loi, os.path.join(fam_out, f"{family_id}__LABEL_OF_INTEREST_EQ_1.tsv"), sep="\t")

        fam_ge2 = sort_by_variant(df_fam_var.loc[pd.to_numeric(df_fam_var.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
        safe_save(fam_ge2, os.path.join(fam_out, f"{family_id}__FAM_COUNT_GE_2.tsv"), sep="\t")

        family_csv_labels = [
            "FAM_SIRIUS_FLAG", "label_of_interest", "FAM_COUNT_GE_2",
            "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT",
            "N_GERM_ONLY", "N_GERM_AND_SOM",
            "patient_cancer_evidence_class",
            "INTEGRATED_SCORE", "INTEGRATED_CLASS", "INTEGRATED_EVIDENCE",
        ]
        export_minimal_csv(df_fam_var, os.path.join(fam_out, f"{family_id}__FAMILY_VARIANTS__LABELS.csv"), family_csv_labels)
        export_minimal_csv(sirius_fam, os.path.join(fam_out, f"{family_id}__SIRIUS_EQ_1__FAMILY_VARIANTS__LABELS.csv"), family_csv_labels)
        export_minimal_csv(loi, os.path.join(fam_out, f"{family_id}__LABEL_OF_INTEREST_EQ_1__LABELS.csv"), family_csv_labels)
        export_minimal_csv(fam_ge2, os.path.join(fam_out, f"{family_id}__FAM_COUNT_GE_2__LABELS.csv"), family_csv_labels)

        fam_long_all.append(df_fam_long)
        fam_pat_all.append(df_pat)
        fam_var_all.append(df_fam_var)

        del df_fam_long
        gc.collect()

    if not fam_pat_all:
        raise RuntimeError("No families produced any data.")

    df_pat_all = pd.concat(fam_pat_all, ignore_index=True)
    df_fam_all = pd.concat(fam_var_all, ignore_index=True)
    df_long_all = sort_by_variant(pd.concat(fam_long_all, ignore_index=True))

    df_global_unique = build_global_unique_variants(df_fam_all)
    df_global_unique = sort_by_variant(df_global_unique)

    return df_pat_all, df_fam_all, df_global_unique, df_long_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON: {family_id: {patient_id: {...}}}")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--no_exome", action="store_true", help="Disable exome merge")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    enable_exome = (not args.no_exome)

    df_pat_all, df_fam_all, df_global_unique, df_long_all = process_all_families_from_json(
        json_path=args.config,
        out_dir=args.out_dir,
        enable_exome=enable_exome
    )

    safe_save(df_long_all, os.path.join(args.out_dir, "ALL_FAMILIES__LONG.tsv"), sep="\t")
    safe_save(sort_by_variant(df_pat_all), os.path.join(args.out_dir, "ALL_FAMILIES__PATIENT_VARIANTS.tsv"), sep="\t")
    safe_save(sort_by_variant(df_fam_all), os.path.join(args.out_dir, "ALL_FAMILIES__FAMILY_VARIANTS.tsv"), sep="\t")
    safe_save(df_global_unique, os.path.join(args.out_dir, "ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv"), sep="\t")

    global_csv_labels = [
        "FAM_SIRIUS_FLAG", "label_of_interest", "FAM_COUNT_GE_2",
        "patient_cancer_evidence_class",
        "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT",
        "N_GERM_ONLY", "N_GERM_AND_SOM",
        "INTEGRATED_SCORE", "INTEGRATED_CLASS", "INTEGRATED_EVIDENCE",
    ]
    export_minimal_csv(df_global_unique, os.path.join(args.out_dir, "ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS__LABELS.csv"), global_csv_labels)

    sirius_uni = sort_by_variant(df_global_unique.loc[pd.to_numeric(df_global_unique.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    sirius_uni_path = os.path.join(args.out_dir, "ALL_FAMILIES__SIRIUS_EQ_1__UNIQUE_VARIANTS.tsv")
    safe_save(sirius_uni, sirius_uni_path, sep="\t")
    export_minimal_csv(sirius_uni, os.path.join(args.out_dir, "ALL_FAMILIES__SIRIUS_EQ_1__UNIQUE_VARIANTS__LABELS.csv"), global_csv_labels)

    df_tmp = df_global_unique.copy()
    ag = pd.to_numeric(df_tmp.get("ACMG_INT_GERM", np.nan), errors="coerce")
    as_ = pd.to_numeric(df_tmp.get("ACMG_INT_SOM", np.nan), errors="coerce")
    df_tmp["ACMG_INT_MAX"] = pd.concat([ag, as_], axis=1).max(axis=1, skipna=True)

    optB = sort_by_variant(df_tmp.loc[pd.to_numeric(df_tmp["ACMG_INT_MAX"], errors="coerce").isin([3, 4, 5])].copy())
    optB_path = os.path.join(args.out_dir, "ALL_FAMILIES__OPTION_B_ACMG_345__UNIQUE_VARIANTS.tsv")
    safe_save(optB, optB_path, sep="\t")
    export_minimal_csv(optB, os.path.join(args.out_dir, "ALL_FAMILIES__OPTION_B_ACMG_345__UNIQUE_VARIANTS__LABELS.csv"), global_csv_labels + ["ACMG_INT_MAX"])

    # Patient evidence class >= 3 unique variants
    pat_cls_path = os.path.join(args.out_dir, "ALL_FAMILIES__PATIENT_EVIDENCE_CLASS_GE_3__UNIQUE_VARIANTS.tsv")
    pat_cls_df = export_patient_class_ge_3_unique(df_pat_all, pat_cls_path)
    export_minimal_csv(pat_cls_df, os.path.join(args.out_dir, "ALL_FAMILIES__PATIENT_EVIDENCE_CLASS_GE_3__UNIQUE_VARIANTS__LABELS.csv"), global_csv_labels)

    # Label of interest across families (family-variant rows)
    loi_all = sort_by_variant(df_fam_all.loc[pd.to_numeric(df_fam_all.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    loi_all_path = os.path.join(args.out_dir, "ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1__FAMILY_VARIANTS.tsv")
    safe_save(loi_all, loi_all_path, sep="\t")
    export_minimal_csv(loi_all, os.path.join(args.out_dir, "ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1__FAMILY_VARIANTS__LABELS.csv"), global_csv_labels)

    # >=2 patients in family (global)
    fam_ge2_all = sort_by_variant(df_fam_all.loc[pd.to_numeric(df_fam_all.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    fam_ge2_all_path = os.path.join(args.out_dir, "ALL_FAMILIES__FAM_COUNT_GE_2.tsv")
    safe_save(fam_ge2_all, fam_ge2_all_path, sep="\t")
    export_minimal_csv(fam_ge2_all, os.path.join(args.out_dir, "ALL_FAMILIES__FAM_COUNT_GE_2__LABELS.csv"), global_csv_labels)

    del df_pat_all, df_fam_all, df_global_unique, df_long_all
    gc.collect()
    print("[DONE]")


if __name__ == "__main__":
    main()
