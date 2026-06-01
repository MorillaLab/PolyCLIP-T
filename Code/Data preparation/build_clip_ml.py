#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_multiview_ml.py

Purpose
-------
Build multi-view ML artifacts from an annotated TSV (germline or somatic) to support:

Model A (no text)
  L = L_seq-bio + λ L_tda
  - seq view: SEQ_REF / SEQ_ALT (fixed window from FASTA)
  - bio view: numeric feature table
  - soft targets (optional): T_sb from ontology (GO/KEGG/HPO) with weighted Jaccard

Model B (with text)
  L = L_seq-bio + β L_seq-text + γ L_bio-text + λ L_tda
  - same seq + bio
  - text view: clinical/functional text built from selected columns
  - text soft targets (optional): T_st and T_bt from TF-IDF similarity
  - BEST text representation for the article:
      (B2) field-wise encoding + weighted pooling (avoids truncation & handles missing fields)
    Baseline:
      (B1) single concatenated CLINICAL_TEXT string

Outputs (in out_dir)
--------------------
Sequences:
  ML_SEQUENCES_ALL.tsv
  ML_SEQUENCES_TRAIN.tsv
  ML_SEQUENCES_TEST.tsv

Bio (numeric features):
  ML_NUMERIC_ALL.tsv
  ML_NUMERIC_TRAIN.tsv
  ML_NUMERIC_TEST.tsv
  df_summary_train.txt
  df_summary_test.txt

Text (optional):
  ML_TEXT_ALL.tsv              (CLINICAL_TEXT + ONTOLOGY_TEXT)
  ML_TEXT_TRAIN.tsv
  ML_TEXT_TEST.tsv

Text fields (optional, for field-wise pooling B2):
  ML_TEXTFIELDS_ALL.tsv        (TXT_* and ONTO_* columns)
  ML_TEXTFIELDS_TRAIN.tsv
  ML_TEXTFIELDS_TEST.tsv

Ontology (optional, for T_sb):
  ML_ONTO_ALL.tsv              (GO_BP, GO_MF, GO_CC, KEGG, HPO)
  ML_ONTO_TRAIN.tsv
  ML_ONTO_TEST.tsv

Optional evaluation subset (not family-based)
---------------------------------------------
If --eval_n > 0, we sample exactly eval_n variants by VARIANT_KEY from ML_NUMERIC_ALL.tsv
and export corresponding subsets:

  ML_SEQUENCES_EVAL500.tsv
  ML_NUMERIC_EVAL500.tsv
  ML_TEXT_EVAL500.tsv (if exists)
  ML_TEXTFIELDS_EVAL500.tsv (if exists)
  ML_ONTO_EVAL500.tsv (if exists)
  EVAL500_KEYS.txt

Optionally you can exclude these eval variants from TRAIN outputs via --eval_exclude_from_train.

Notes
-----
- If FAMILY_ID is missing in the input TSV, we auto-create FAMILY_ID="GLOBAL".
- If allele_1/allele_2 exist they are used as REF/ALT for sequence windows; else REF/ALT.

"""

from __future__ import annotations

import argparse
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.preprocessing import OneHotEncoder

# Optional torch usage (only required if you use Dataset/collate/pooling)
try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object


# =============================================================================
# Constants
# =============================================================================

NA_LIKE = {".", "NA", "nan", "None", "", "NULL", "NaN"}

CONSEQUENCE_SCORE: Dict[str, int] = {
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

IMPACT_MAP: Dict[str, int] = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

VARIANT_CLASS_MAP: Dict[str, int] = {
    "SNV": 1,
    "substitution": 1,
    "sequence_alteration": 1,
    "insertion": 2,
    "deletion": 2,
    "indel": 2,
}

BIOTYPE_MAP: Dict[str, int] = {
    "protein_coding": 3,
    "processed_transcript": 2,
    "lncRNA": 2,
    "lncrna": 2,
    "miRNA": 2,
    "snRNA": 2,
    "snoRNA": 2,
    "rRNA": 2,
    "pseudogene": 1,
    "processed_pseudogene": 1,
}

# Canonical ontology export names (your columns are exactly these in your TSV schema)
ONTO_SOURCE_COLS = {
    "GO_BP": "GO_biological_process",
    "GO_CC": "GO_cellular_component",
    "GO_MF": "GO_molecular_function",
    "KEGG": "KEGG_Pathway",
    "HPO": "HPO",
}

# Canonical text-field names for field-wise pooling (Model B2)
TEXT_FIELD_COLS = {
    "TXT_FUNC": "Function_description",
    "TXT_DISEASE": "Disease_description",
    "TXT_CLNSIG": "ClinVar_CLNSIG",
    "TXT_CLNDN": "ClinVar_CLNDN",
    "TXT_OMIM": "OMIM",
    "TXT_DISGENET": "DisGeNET",
    "TXT_GWAS": "Trait_association(GWAS)",
    # Somatic KB signals (safe: missing => empty)
    "TXT_COSMIC": "COSMIC",
    # CIViC/CGI/OncoKB are spread across many columns => summarized, not direct copy
}


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _norm_chr(ch: Any) -> str:
    if ch is None or (isinstance(ch, float) and np.isnan(ch)):
        return ""
    s = str(ch).strip()
    return s.replace("CHR", "").replace("chr", "")


def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


# =============================================================================
# Numeric parsing helpers
# =============================================================================

def extract_float(x: Any) -> float:
    """Examples:
      'deleterious(0.02)' -> 0.02
      '.'/'NA' -> NaN
      '0.5' -> 0.5
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in NA_LIKE:
        return np.nan
    m = re.search(r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_path5(x: Any) -> float:
    """
    5 ordered classes:
      1 = Benign
      2 = Likely benign
      3 = VUS / Uncertain significance
      4 = Likely pathogenic
      5 = Pathogenic
    Returns NaN for missing/unclassified/conflicting unknown text.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = x[0]

    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    s = str(x).strip()
    if s in NA_LIKE:
        return np.nan

    if s.isdigit():
        v = int(s)
        # keep only strict 1..5 classes
        return float(v) if 1 <= v <= 5 else np.nan

    sl = s.lower()

    if "conflicting" in sl:
        return np.nan
    if "pathogenic" in sl and "likely" not in sl:
        return 5.0
    if "likely_pathogenic" in sl or ("likely" in sl and "pathog" in sl):
        return 4.0
    if "uncertain" in sl or "vus" in sl:
        return 3.0
    if "likely_benign" in sl or ("likely" in sl and "benign" in sl):
        return 2.0
    if "benign" in sl:
        return 1.0

    return np.nan


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def bin_from_text(x: Any, positives: List[str]) -> int:
    s = "" if pd.isna(x) else str(x).lower()
    return int(any(p in s for p in positives))


def encode_polyphen(x: Any) -> float:
    s = "" if pd.isna(x) else str(x).lower()
    if "probably" in s:
        return 2.0
    if "possibly" in s:
        return 1.0
    if "benign" in s:
        return 0.0
    return np.nan


def _split_cons(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip().lower()
    if s in NA_LIKE:
        return []
    parts = re.split(r"[,&|]+", s)
    return [p.strip() for p in parts if p.strip()]


def encode_consequence(x: Any) -> int:
    """If multiple consequences -> take most severe."""
    parts = _split_cons(x)
    if not parts:
        return 0
    scores = [CONSEQUENCE_SCORE.get(p, 1) for p in parts]
    return int(max(scores)) if scores else 0


def encode_impact(x: Any) -> int:
    if pd.isna(x):
        return 0
    return IMPACT_MAP.get(str(x).strip().upper(), 0)


def encode_variant_class(x: Any) -> int:
    if pd.isna(x):
        return 0
    return VARIANT_CLASS_MAP.get(str(x).strip(), 0)


def encode_biotype(x: Any) -> int:
    if pd.isna(x):
        return 0
    return BIOTYPE_MAP.get(str(x).strip(), 0)


def encode_canonical(x: Any) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    return 1 if s in {"YES", "1", "true", "True"} else 0


# =============================================================================
# FASTA fixed window extraction (ref/alt)
# =============================================================================

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
    For indels: replace starting at the center, then trim/pad to fixed length.
    If reference mismatch at the locus -> (SEQ_REF, None).

    Note:
    - May introduce 'N' padding to keep fixed length for indels.
    - For SNVs, replacement is exact at the center base.
    """
    try:
        chrom = _norm_chr(chrom)
        if not chrom:
            return None, None

        c1, c2 = chrom, f"chr{chrom}"
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

        ref = str(ref).upper().strip()
        alt = str(alt).upper().strip()
        if (ref in NA_LIKE) or (alt in NA_LIKE) or (not ref) or (not alt):
            return None, None

        # SNV
        if len(ref) == 1 and len(alt) == 1:
            seq_alt = seq_ref[:half] + alt + seq_ref[half + 1 :]
            return seq_ref, seq_alt

        # Indel: verify ref matches at the locus
        start = half
        end = min(window, half + len(ref))
        if seq_ref[start:end] != ref[: end - start]:
            return seq_ref, None

        seq_alt = seq_ref[:start] + alt + seq_ref[end:]

        # Force fixed length
        if len(seq_alt) > window:
            seq_alt = seq_alt[:window]
        elif len(seq_alt) < window:
            seq_alt = seq_alt + ("N" * (window - len(seq_alt)))

        if len(seq_alt) != window:
            return None, None

        return seq_ref, seq_alt

    except Exception:
        return None, None


# =============================================================================
# Summary stats helper
# =============================================================================

def create_sum_stat(df: pd.DataFrame, out_txt: str) -> None:
    lines: List[str] = []

    buf = io.StringIO()
    df.info(buf=buf)
    lines.append("=== DataFrame Info ===\n")
    lines.append(buf.getvalue() + "\n\n")

    lines.append("=== Descriptive Statistics ===\n")
    lines.append(str(df.describe(include="all")) + "\n\n")

    lines.append("=== NaN and Unique Values per Column ===\n")
    nan_summary = df.isna().sum().to_frame(name="NaN_count")
    nan_summary["dtype"] = df.dtypes
    nan_summary["unique_values"] = df.nunique(dropna=True)
    lines.append(str(nan_summary) + "\n\n")

    ensure_dir(os.path.dirname(out_txt) or ".")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Summary saved to", out_txt)


# =============================================================================
# Labels helper (requested policy)
# =============================================================================

def make_labels(df: pd.DataFrame) -> pd.Series:
    """
    Requested policy:
      label = 1 iff FAM_COUNT_GE_2 == 1, else 0.
    If column missing: all zeros.
    """
    fam = pd.to_numeric(df.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int)
    return (fam == 1).astype(int)

def make_multitask_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - label_clinvar : strict 5-class ClinVar label
      - label_acmg    : strict 5-class ACMG label
    """

    out = pd.DataFrame(index=df.index)

    # ClinVar
    clinvar_col = find_first_existing_col(df, ["ClinVar_CLNSIG", "CLNSIG"])
    if clinvar_col:
        out["label_clinvar"] = pd.to_numeric(
            df[clinvar_col].map(parse_path5), errors="coerce"
        )
    else:
        out["label_clinvar"] = np.nan

    # ACMG
    if "ACMG_INT" in df.columns:
        y = pd.to_numeric(df["ACMG_INT"], errors="coerce")
        y = y.where(y.isin([1, 2, 3, 4, 5]), np.nan)
        out["label_acmg"] = y
    else:
        acmg_col = find_first_existing_col(
            df,
            [
                "Prediction_ACMG_tapes",
                "EXOME_Prediction_ACMG_tapes",
                "Prediction_ACMG",
                "EXOME_Prediction_ACMG",
                "ACMG_RAW",
            ],
        )
        if acmg_col:
            out["label_acmg"] = pd.to_numeric(
                df[acmg_col].map(parse_path5), errors="coerce"
            )
        else:
            out["label_acmg"] = np.nan

    return out
# =============================================================================
# Text and ontology construction (article-friendly)
# =============================================================================

def clean_text(x: Any) -> str:
    """Normalize a cell to a clean one-line string, or '' if NA-like."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s in NA_LIKE:
        return ""
    return re.sub(r"\s+", " ", s)


def split_terms(x: Any) -> List[str]:
    """Split list-like fields. Supports separators: ',', ';', '|'."""
    s = clean_text(x)
    if not s:
        return []
    parts = re.split(r"[;,|]+", s)
    parts = [p.strip() for p in parts if p.strip() and p.strip() not in NA_LIKE]
    # unique keep order
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def join_terms(terms: List[str], max_items: int = 25) -> str:
    if not terms:
        return ""
    return "; ".join(terms[:max_items])


def opt_block(title: str, content: str) -> str:
    """Return 'TITLE: content' only if content is non-empty."""
    if not content:
        return ""
    return f"{title}: {content}"


def field(row: pd.Series, col: str) -> str:
    return clean_text(row[col]) if col in row.index else ""


def oncokb_level_summary(row: pd.Series) -> str:
    """
    Summarize OncoKB levels (somatic-only, but safe if columns missing).
    Uses:
      HIGHEST_LEVEL, ONCOGENIC, MUTATION_EFFECT
      LEVEL_1, LEVEL_2, LEVEL_3A, LEVEL_3B, LEVEL_4, LEVEL_R1, LEVEL_R2, LEVEL_R3
    """
    highest = field(row, "HIGHEST_LEVEL")
    oncogenic = field(row, "ONCOGENIC")
    effect = field(row, "MUTATION_EFFECT")

    level_cols = ["LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4", "LEVEL_R1", "LEVEL_R2", "LEVEL_R3"]
    present: List[str] = []
    for c in level_cols:
        if c in row.index:
            v = clean_text(row[c])
            if v and v not in {"0", "False", "false", "NO", "No"}:
                present.append(c)

    bits: List[str] = []
    if highest:
        bits.append(f"HIGHEST={highest}")
    if present:
        bits.append(f"LEVELS={','.join(present)}")
    if oncogenic:
        bits.append(f"ONCOGENIC={oncogenic}")
    if effect:
        bits.append(f"EFFECT={effect}")

    return "; ".join(bits)


def civic_summary(row: pd.Series, max_len: int = 200) -> str:
    """Compact summary of CIViC columns if present."""
    keys = [
        "CIViC_Variant_clinical_significance",
        "CIViC_Variant_evidence_level",
        "CIViC_Variant_disease",
        "CIViC_Variant_drugs",
        "CIViC_Variant_citation_id",
        "CIViC_Region_variant_type",
        "CIViC_Region_clinical_significance",
        "CIViC_Region_evidence_level",
        "CIViC_Region_disease",
        "CIViC_Region_drugs",
        "CIViC_Region_citation_id",
    ]
    vals: List[str] = []
    for k in keys:
        v = field(row, k)
        if v:
            if len(v) > max_len:
                v = v[:max_len].rstrip() + "..."
            vals.append(f"{k}={v}")
    return " ; ".join(vals)


def cgi_summary(row: pd.Series, max_len: int = 200) -> str:
    """Compact summary of CancerGenomeInterpreter columns if present."""
    keys = [
        "CancerGenomeInterpreter_Association",
        "CancerGenomeInterpreter_Evidence_level",
        "CancerGenomeInterpreter_Drug",
        "CancerGenomeInterpreter_Primary_Tumor_type",
        "CancerGenomeInterpreter_Source",
    ]
    vals: List[str] = []
    for k in keys:
        v = field(row, k)
        if v:
            if len(v) > max_len:
                v = v[:max_len].rstrip() + "..."
            vals.append(f"{k}={v}")
    return " ; ".join(vals)


def build_ontology_fields(row: pd.Series) -> Dict[str, str]:
    """
    Canonical ontology fields (string form; set parsing happens later):
      GO_BP, GO_MF, GO_CC, KEGG, HPO
    """
    go_bp = join_terms(split_terms(row.get("GO_biological_process", "")), max_items=25)
    go_mf = join_terms(split_terms(row.get("GO_molecular_function", "")), max_items=25)
    go_cc = join_terms(split_terms(row.get("GO_cellular_component", "")), max_items=25)
    kegg = join_terms(split_terms(row.get("KEGG_Pathway", "")), max_items=25)
    hpo = join_terms(split_terms(row.get("HPO", "")), max_items=25)
    return {"GO_BP": go_bp, "GO_MF": go_mf, "GO_CC": go_cc, "KEGG": kegg, "HPO": hpo}


def build_ontology_text(onto: Dict[str, str]) -> str:
    parts = [
        opt_block("GO_BP", onto.get("GO_BP", "")),
        opt_block("GO_MF", onto.get("GO_MF", "")),
        opt_block("GO_CC", onto.get("GO_CC", "")),
        opt_block("KEGG", onto.get("KEGG", "")),
        opt_block("HPO", onto.get("HPO", "")),
    ]
    parts = [p for p in parts if p]
    return " | ".join(parts)


def build_clinical_fields(row: pd.Series) -> Dict[str, str]:
    """
    Canonical clinical/functional text fields for Model B2:
      FUNC, DISEASE, CLNSIG, CLNDN, OMIM, DISGENET, GWAS,
      COSMIC, CIVIC, CGI, ONCOKB
    """
    out: Dict[str, str] = {
        "FUNC": field(row, "Function_description"),
        "DISEASE": field(row, "Disease_description"),
        "CLNSIG": field(row, "ClinVar_CLNSIG"),
        "CLNDN": field(row, "ClinVar_CLNDN"),
        "OMIM": field(row, "OMIM"),
        "DISGENET": field(row, "DisGeNET"),
        "GWAS": field(row, "Trait_association(GWAS)"),
        "COSMIC": field(row, "COSMIC"),
        "CIVIC": civic_summary(row),
        "CGI": cgi_summary(row),
        "ONCOKB": oncokb_level_summary(row),
    }
    return out


def build_clinical_text(cf: Dict[str, str]) -> str:
    """
    Model B1: single concatenated string with stable separators and labels.
    """
    parts = [
        opt_block("FUNCTION", cf.get("FUNC", "")),
        opt_block("DISEASE", cf.get("DISEASE", "")),
        opt_block("CLINVAR_CLNSIG", cf.get("CLNSIG", "")),
        opt_block("CLINVAR_CLNDN", cf.get("CLNDN", "")),
        opt_block("OMIM", cf.get("OMIM", "")),
        opt_block("DISGENET", cf.get("DISGENET", "")),
        opt_block("GWAS", cf.get("GWAS", "")),
    ]
    somatic_parts = [
        opt_block("COSMIC", cf.get("COSMIC", "")),
        opt_block("CIVIC", cf.get("CIVIC", "")),
        opt_block("CGI", cf.get("CGI", "")),
        opt_block("ONCOKB", cf.get("ONCOKB", "")),
    ]
    somatic_parts = [p for p in somatic_parts if p]
    if somatic_parts:
        parts += somatic_parts

    parts = [p for p in parts if p]
    return " | ".join(parts)


def add_text_and_ontology_views(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ONTOLOGY_TEXT (concatenated)
      - CLINICAL_TEXT (concatenated)
      - ONTO_* field columns: ONTO_GO_BP, ONTO_GO_MF, ONTO_GO_CC, ONTO_KEGG, ONTO_HPO
      - TXT_* field columns for Model B2 pooling
    """
    out = df.copy()

    onto_rows: List[Dict[str, str]] = []
    clin_rows: List[Dict[str, str]] = []

    for _, r in out.iterrows():
        onto = build_ontology_fields(r)
        clin = build_clinical_fields(r)
        onto_rows.append(onto)
        clin_rows.append(clin)

    onto_df = pd.DataFrame(onto_rows, index=out.index)
    clin_df = pd.DataFrame(clin_rows, index=out.index)

    # Field columns (Model B2)
    out["ONTO_GO_BP"] = onto_df["GO_BP"]
    out["ONTO_GO_MF"] = onto_df["GO_MF"]
    out["ONTO_GO_CC"] = onto_df["GO_CC"]
    out["ONTO_KEGG"] = onto_df["KEGG"]
    out["ONTO_HPO"] = onto_df["HPO"]

    out["TXT_FUNC"] = clin_df["FUNC"]
    out["TXT_DISEASE"] = clin_df["DISEASE"]
    out["TXT_CLNSIG"] = clin_df["CLNSIG"]
    out["TXT_CLNDN"] = clin_df["CLNDN"]
    out["TXT_OMIM"] = clin_df["OMIM"]
    out["TXT_DISGENET"] = clin_df["DISGENET"]
    out["TXT_GWAS"] = clin_df["GWAS"]
    out["TXT_COSMIC"] = clin_df["COSMIC"]
    out["TXT_CIVIC"] = clin_df["CIVIC"]
    out["TXT_CGI"] = clin_df["CGI"]
    out["TXT_ONCOKB"] = clin_df["ONCOKB"]

    # Concatenated strings (Model B1)
    out["ONTOLOGY_TEXT"] = [build_ontology_text(onto_rows[i]) for i in range(len(out))]
    out["CLINICAL_TEXT"] = [build_clinical_text(clin_rows[i]) for i in range(len(out))]

    return out


# =============================================================================
# Soft targets: ontology and text
# =============================================================================

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b)) / float(len(u))


def row_softmax_np(S: np.ndarray, alpha: float = 0.3, zero_diag: bool = False) -> np.ndarray:
    """Row-stochastic targets: softmax(S/alpha)."""
    S = S.astype(np.float64, copy=True)
    if zero_diag:
        np.fill_diagonal(S, -1e9)

    S = S / max(alpha, 1e-8)
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    T = E / (E.sum(axis=1, keepdims=True) + 1e-12)
    return T.astype(np.float32)


def soft_targets_from_ontology_sets(
    onto_sets: List[Dict[str, set]],
    weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.3,
    zero_diag: bool = False,
) -> np.ndarray:
    """
    onto_sets: list length N; each element keys:
      go_bp, go_mf, go_cc, kegg, hpo -> sets of terms
    """
    if weights is None:
        weights = {"go_bp": 1.0, "go_mf": 0.5, "go_cc": 0.5, "kegg": 1.0, "hpo": 1.0}

    keys = list(weights.keys())
    N = len(onto_sets)
    S = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i, N):
            sim = 0.0
            for k in keys:
                wk = float(weights.get(k, 0.0))
                if wk <= 0:
                    continue
                sim += wk * jaccard(onto_sets[i].get(k, set()), onto_sets[j].get(k, set()))
            S[i, j] = sim
            S[j, i] = sim

    return row_softmax_np(S, alpha=alpha, zero_diag=zero_diag)


def soft_targets_from_text_tfidf(
    texts: List[str],
    alpha: float = 0.3,
    zero_diag: bool = False,
) -> np.ndarray:
    """
    Batch-local TF-IDF cosine similarity -> row softmax targets.
    Suitable for T_st and T_bt when you do not want to compute transformer embeddings.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    clean = [clean_text(t) for t in texts]
    if all(len(t) == 0 for t in clean):
        N = len(clean)
        return np.eye(N, dtype=np.float32)

    vec = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(clean)
    S = cosine_similarity(X)
    return row_softmax_np(S, alpha=alpha, zero_diag=zero_diag)


def onto_sets_from_row(row: pd.Series) -> Dict[str, set]:
    """
    Expect canonical columns created by add_text_and_ontology_views:
      ONTO_GO_BP, ONTO_GO_MF, ONTO_GO_CC, ONTO_KEGG, ONTO_HPO
    """
    return {
        "go_bp": set(split_terms(row.get("ONTO_GO_BP", ""))),
        "go_mf": set(split_terms(row.get("ONTO_GO_MF", ""))),
        "go_cc": set(split_terms(row.get("ONTO_GO_CC", ""))),
        "kegg": set(split_terms(row.get("ONTO_KEGG", ""))),
        "hpo": set(split_terms(row.get("ONTO_HPO", ""))),
    }


# =============================================================================
# Eval sampling helper (by VARIANT_KEY)
# =============================================================================

def sample_eval_keys(
    df: pd.DataFrame,
    n: int,
    seed: int = 7,
    mode: str = "random",
) -> List[str]:
    """
    Return a list of VARIANT_KEY to use for eval.
    Assumes df has VARIANT_KEY and optionally labels.
    """
    if n <= 0:
        return []
    if "VARIANT_KEY" not in df.columns:
        raise ValueError("sample_eval_keys requires VARIANT_KEY in df")

    keys = df["VARIANT_KEY"].dropna().astype(str)
    keys = keys[~keys.isin(list(NA_LIKE))].drop_duplicates()

    if len(keys) <= n:
        return keys.tolist()

    rng = np.random.default_rng(seed)

    if mode == "random" or "labels" not in df.columns:
        return rng.choice(keys.values, size=n, replace=False).tolist()

    # stratified by labels (keeps class balance roughly)
    tmp = df[["VARIANT_KEY", "labels"]].dropna().copy()
    tmp["VARIANT_KEY"] = tmp["VARIANT_KEY"].astype(str)
    tmp = tmp.drop_duplicates("VARIANT_KEY")

    k0 = tmp.loc[tmp["labels"] == 0, "VARIANT_KEY"].values
    k1 = tmp.loc[tmp["labels"] == 1, "VARIANT_KEY"].values

    p1 = len(k1) / max(len(k0) + len(k1), 1)
    n1 = int(round(n * p1))
    n0 = n - n1

    n1 = min(n1, len(k1))
    n0 = min(n0, len(k0))

    part1 = rng.choice(k1, size=n1, replace=False).tolist() if n1 > 0 else []
    part0 = rng.choice(k0, size=n0, replace=False).tolist() if n0 > 0 else []

    chosen = set(part0 + part1)
    if len(chosen) < n:
        remaining = np.array([k for k in keys.values if k not in chosen], dtype=object)
        need = n - len(chosen)
        topup = rng.choice(remaining, size=need, replace=False).tolist()
        chosen.update(topup)

    return list(chosen)


# =============================================================================
# Numeric ML table builder
# =============================================================================
# def build_numeric_ml_table(
#     df_in: pd.DataFrame,
#     out_dir: str,
#     test_family: str,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
def build_numeric_ml_table(
    df_in: pd.DataFrame,
    out_dir: str,
    test_family: str,
    train_n: int = 0,
    train_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds:
      ML_NUMERIC_ALL.tsv
      ML_NUMERIC_TRAIN.tsv
      ML_NUMERIC_TEST.tsv

    Labels:
      labels = 1 iff FAM_COUNT_GE_2 == 1 else 0

    Split:
      train: FAMILY_ID != test_family
      test : FAMILY_ID == test_family
    """
    df = df_in.copy()

    if "FAMILY_ID" not in df.columns:
        df["FAMILY_ID"] = "GLOBAL"

    # --------------------------------------------------
    # Prefer generic columns
    # --------------------------------------------------
    cons_col = pick_col(df, ["EXOME_Consequence", "ANN_Consequence", "Consequence"])
    impact_col = pick_col(df, ["EXOME_IMPACT", "ANN_IMPACT", "IMPACT"])
    varcls_col = pick_col(df, ["EXOME_VARIANT_CLASS", "ANN_VARIANT_CLASS", "VARIANT_CLASS"])
    biotype_col = pick_col(df, ["EXOME_BIOTYPE", "ANN_BIOTYPE", "BIOTYPE"])
    canonical_col = pick_col(df, ["EXOME_CANONICAL", "ANN_CANONICAL", "CANONICAL"])

    # --------------------------------------------------
    # Derived predictors
    # --------------------------------------------------
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

    # Encode consequence/impact/etc.
    df["Consequence_num"] = df[cons_col].apply(encode_consequence) if cons_col else 0
    df["Impact_num"] = df[impact_col].apply(encode_impact) if impact_col else 0
    df["VariantClass_num"] = df[varcls_col].apply(encode_variant_class) if varcls_col else 0
    df["Biotype_num"] = df[biotype_col].apply(encode_biotype) if biotype_col else 0
    df["Canonical_num"] = df[canonical_col].apply(encode_canonical) if canonical_col else 0

    # --------------------------------------------------
    # Optional helper encodings for frequent categorical flags
    # --------------------------------------------------
    origin_col = pick_col(df, ["ORIGIN"])
    if origin_col:
        origin_map = {
            "germline": 1,
            "somatic": 2,
            "both": 3,
            "germline+somatic": 3,
            "unknown": 0,
            "na": 0,
        }
        df["Origin_num"] = (
            df[origin_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(origin_map)
            .fillna(0)
            .astype(int)
        )
    else:
        df["Origin_num"] = 0

    cnv_loh_col = pick_col(df, ["CNV_LOH_status"])
    if cnv_loh_col:
        df["CNV_LOH_status_num"] = (
            df[cnv_loh_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "loh": 2,
                "partial_loh": 1,
                "neutral": 0,
                "none": 0,
                "na": 0,
            })
            .fillna(0)
            .astype(int)
        )
    else:
        df["CNV_LOH_status_num"] = 0

    # --------------------------------------------------
    # Numeric columns
    # Low-penetrance oriented:
    # population AF + functional scores + family/context
    # --------------------------------------------------
    NUMERIC_COLS = [
        # Basic quality / depth
        "QUAL",
        "DP_GERM",
        "VAF_GERM",
        "DP_SOM",
        "VAF_SOM",
        "DP_UNI",
        "VAF_UNI",

        # Population AF
        "IG_AF",
        "EVS_EA_MAF",
        "1000G_Global_AF",
        "gnomAD_Global_AF",
        "Kaviar_AF",
        "MAX_AF",
        "MAX_POP_AF",
        "GnomAD_MNV_AF",

        # Population subgroups (optional but useful if available)
        "1000G_AFR_AF",
        "1000G_AMR_AF",
        "1000G_EAS_AF",
        "1000G_EUR_AF",
        "1000G_SAS_AF",
        "1000G_AA_AF",
        "1000G_EA_AF",
        "gnomAD_AFR_AF",
        "gnomAD_AMR_AF",
        "gnomAD_ASJ_AF",
        "gnomAD_EAS_AF",
        "gnomAD_FIN_AF",
        "gnomAD_NFE_AF",
        "gnomAD_OTH_AF",
        "gnomAD_SAS_AF",

        # Functional / conservation
        "PhyloP",
        "PhastCons",
        "Distance_Grantham",
        "MaxEntScan_alt",
        "MaxEntScan_diff",
        "MaxEntScan_ref",
        "ada_score",
        "rf_score",
        "DANN_Score",
        "FATHMM_Non_Coding_Score",
        "FATHMM_Coding_Score",
        "Gene_damage_index",

        # Derived predictors
        "SIFT_score",
        "PolyPhen_ord",
        "PolyPhen_score",
        "Consequence_num",
        "Impact_num",
        "VariantClass_num",
        "Biotype_num",
        "Canonical_num",
        "Origin_num",
        "CNV_LOH_status_num",

        # Family / aggregation / evidence
        "N_PATIENTS_WITH_VARIANT",
        "N_CANCER_WITH_VARIANT",
        "N_NONCANCER_WITH_VARIANT",
        "SOMATIC_REPEAT_COUNT",
        "FAMILY_EVIDENCE_SCORE",
        "SEGREGATION_SCORE",
        "patient_cancer_evidence_score",

        # Optional: test with and without these in ablation
        "ACMG_INT_GERM",
        "ACMG_INT_SOM",
    ]

    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------------------------
    # Binary columns
    # --------------------------------------------------
    BINARY_COLS = [
        "SIFT_bin",
        "FAM_COUNT_GE_2",
        "FAM_SIRIUS_FLAG",
        "label_of_interest",
        "PAT_SIRIUS_FLAG",

        "HAS_GERM",
        "HAS_SOM",
        "N_GERM_ONLY",
        "N_GERM_AND_SOM",
        "PATHWAY_HIT",
        "PATH_DNA_REPAIR",
        "LOH_PARTIAL_PATHO",
        "LOH_PARTIAL_PATHO_SOM",
        "CNV_HIT",
        "CNV_pathogenic",
        "IN_ROH",
        "HET_HIGH_HOM",
        "HAS_SOMATIC_PATIENT",
        "SIRIUS_PASS_POPAF",
        "SIRIUS_PASS_NONSYN",
        "SIRIUS_PASS_SNV_GERMLINE",
        "SIRIUS_PASS_SNV_SOMATIC",
        "SIRIUS_PASS_CNV",
        "predisposition_flag",
        "somatic_driver_flag",
        "two_hit_flag",
        "literature_flag",
        "TWO_HIT_STRICT",
        "GERM_PASS",
    ]

    for c in BINARY_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------
    # Categorical -> one-hot
    # Keep these modest at first
    # --------------------------------------------------
    CATEGORICAL_COLS = [
        "VARIANT_CLASS",
        # "FILTER",
        "Consequence",
        "IMPACT",
        "BIOTYPE",
        "ORIGIN",
    ]
    cat_existing = [c for c in CATEGORICAL_COLS if c in df.columns]

    if cat_existing:
        df[cat_existing] = df[cat_existing].fillna("NA").astype(str)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_matrix = enc.fit_transform(df[cat_existing])
        cat_df = pd.DataFrame(
            cat_matrix,
            columns=enc.get_feature_names_out(cat_existing),
            index=df.index,
        )
    else:
        cat_df = pd.DataFrame(index=df.index)

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    df["labels"] = make_labels(df)
    label_df = make_multitask_labels(df)
    df["label_clinvar"] = label_df["label_clinvar"]
    df["label_acmg"] = label_df["label_acmg"]

    # --------------------------------------------------
    # Keys
    # --------------------------------------------------
    df["CHROM"] = df["CHROM"].apply(_norm_chr) if "CHROM" in df.columns else ""
    df["POS"] = pd.to_numeric(df.get("POS", np.nan), errors="coerce")

    if "TX" not in df.columns:
        df["TX"] = "NO_TX"

    df["VARIANT_KEY"] = (
        df.get("CHROM", "").astype(str)
        + ":"
        + df["POS"].astype("Int64").astype(str)
        + ":"
        + df.get("REF", "").astype(str)
        + ":"
        + df.get("ALT", "").astype(str)
        + ":"
        + df.get("TX", "").astype(str)
    )

    gene_col = pick_col(df, ["Gene_Name", "HGNC_Name"])
    if gene_col and gene_col != "Gene_Name":
        df["Gene_Name"] = df[gene_col].astype(str)
    elif "Gene_Name" not in df.columns:
        df["Gene_Name"] = ""

    hgnc_col = pick_col(df, ["HGNC_Name"])
    if hgnc_col and hgnc_col != "HGNC_Name":
        df["HGNC_Name"] = df[hgnc_col].astype(str)
    elif "HGNC_Name" not in df.columns:
        df["HGNC_Name"] = ""

    feature_refseq_col = pick_col(df, ["Feature_RefSeq"])
    if feature_refseq_col and feature_refseq_col != "Feature_RefSeq":
        df["Feature_RefSeq"] = df[feature_refseq_col].astype(str)
    elif "Feature_RefSeq" not in df.columns:
        df["Feature_RefSeq"] = ""

    keep = [
        c for c in [
            "FAMILY_ID",
            "VARIANT_KEY",
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "TX",
            "Gene_Name",
            "HGNC_Name",
            "Feature_RefSeq",
            "label_clinvar",
            "label_acmg",
            "labels",
        ]
        if c in df.columns
    ]

    low_var_cols = [
        c for c in df.columns
        if df[c].nunique() < 5
    ]
    print("Low variance columns:", low_var_cols)


    low_var_cols = [
        c for c in df.columns
            if df[c].nunique() < 2
    ]
    print("Low variance columns 1:", low_var_cols)


    # cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    # df = df.drop(columns=cols_to_drop)

    # --------------------------------------------------
    # Split
    # --------------------------------------------------
    df_train = df[df["FAMILY_ID"].astype(str) != str(test_family)].copy()
    df_test = df[df["FAMILY_ID"].astype(str) == str(test_family)].copy()


    # --------------------------------------------------
    # Optional train subsampling
    # --------------------------------------------------
    if train_n > 0:
        old_n = len(df_train)
        n = min(train_n, old_n)

        df_train = df_train.sample(n=n, random_state=train_seed).copy()

        print(f"[TRAIN SUBSET] Keeping {n}/{old_n} training variants")

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

        return out.fillna(0)

    all_final = make_ml_block(df)
    train_final = make_ml_block(df_train)
    test_final = make_ml_block(df_test)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    ensure_dir(out_dir)
    all_path = os.path.join(out_dir, "ML_NUMERIC_ALL.tsv")
    train_path = os.path.join(out_dir, "ML_NUMERIC_TRAIN.tsv")
    test_path = os.path.join(out_dir, "ML_NUMERIC_TEST.tsv")

    all_final.to_csv(all_path, sep="\t", index=False)
    train_final.to_csv(train_path, sep="\t", index=False)
    test_final.to_csv(test_path, sep="\t", index=False)

    print("Saved ALL  :", all_path, "| shape:", all_final.shape)
    print("Saved TRAIN:", train_path, "| shape:", train_final.shape)
    print(train_final["labels"].value_counts(dropna=False))
    print("Saved TEST :", test_path, "| shape:", test_final.shape)
    print(test_final["labels"].value_counts(dropna=False))

    create_sum_stat(train_final, os.path.join(out_dir, "df_summary_train.txt"))
    create_sum_stat(test_final, os.path.join(out_dir, "df_summary_test.txt"))

    return all_final, train_final, test_final
# def build_numeric_ml_table(
#     df_in: pd.DataFrame,
#     out_dir: str,
#     test_family: str,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Builds:
#       ML_NUMERIC_ALL.tsv
#       ML_NUMERIC_TRAIN.tsv
#       ML_NUMERIC_TEST.tsv

#     Labels:
#       labels = 1 iff FAM_COUNT_GE_2 == 1 else 0

#     Split:
#       train: FAMILY_ID != test_family
#       test : FAMILY_ID == test_family
#     """
#     df = df_in.copy()

#     if "FAMILY_ID" not in df.columns:
#         df["FAMILY_ID"] = "GLOBAL"

#     # Prefer generic columns
#     cons_col = pick_col(df, ["EXOME_Consequence", "ANN_Consequence", "Consequence"])
#     impact_col = pick_col(df, ["EXOME_IMPACT", "ANN_IMPACT", "IMPACT"])
#     varcls_col = pick_col(df, ["EXOME_VARIANT_CLASS", "ANN_VARIANT_CLASS", "VARIANT_CLASS"])
#     biotype_col = pick_col(df, ["EXOME_BIOTYPE", "ANN_BIOTYPE", "BIOTYPE"])
#     canonical_col = pick_col(df, ["EXOME_CANONICAL", "ANN_CANONICAL", "CANONICAL"])

#     # Derived predictors
#     sift_col = pick_col(df, ["ANN_SIFT", "EXOME_ANN_SIFT", "SIFT"])
#     if sift_col:
#         df["SIFT_score"] = df[sift_col].apply(extract_float)
#         df["SIFT_bin"] = df[sift_col].apply(lambda x: bin_from_text(x, ["deleterious"]))
#     else:
#         df["SIFT_score"] = np.nan
#         df["SIFT_bin"] = 0

#     poly_col = pick_col(df, ["ANN_PolyPhen", "EXOME_ANN_PolyPhen", "PolyPhen"])
#     if poly_col:
#         df["PolyPhen_ord"] = df[poly_col].apply(encode_polyphen)
#         df["PolyPhen_score"] = df[poly_col].apply(extract_float)
#     else:
#         df["PolyPhen_ord"] = np.nan
#         df["PolyPhen_score"] = np.nan

#     # Encode consequence/impact etc.
#     df["Consequence_num"] = df[cons_col].apply(encode_consequence) if cons_col else 0
#     df["Impact_num"] = df[impact_col].apply(encode_impact) if impact_col else 0
#     df["VariantClass_num"] = df[varcls_col].apply(encode_variant_class) if varcls_col else 0
#     df["Biotype_num"] = df[biotype_col].apply(encode_biotype) if biotype_col else 0
#     df["Canonical_num"] = df[canonical_col].apply(encode_canonical) if canonical_col else 0

#     # Numeric columns: robust presence; missing => NaN then filled with 0
#     NUMERIC_COLS = [
#         "QUAL",
#         "IG_AF",
#         "EVS_EA_MAF",
#         "1000G_Global_AF",
#         "gnomAD_Global_AF",
#         "Kaviar_AF",
#         "MAX_AF",
#         "GnomAD_MNV_AF",
#         "PhyloP",
#         "PhastCons",
#         "Distance_Grantham",
#         "MaxEntScan_alt",
#         "MaxEntScan_diff",
#         "MaxEntScan_ref",
#         "ada_score",
#         "rf_score",
#         "DANN_Score",
#         "FATHMM_Non_Coding_Score",
#         "FATHMM_Coding_Score",
#         "Gene_damage_index",
#         "Probability_Path",
#         "SIFT_score",
#         "PolyPhen_ord",
#         "PolyPhen_score",
#         "Consequence_num",
#         "Impact_num",
#         "VariantClass_num",
#         "Biotype_num",
#         "Canonical_num",
#     ]

#     for c in NUMERIC_COLS:
#         if c not in df.columns:
#             df[c] = np.nan
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Binary columns: robust; missing => 0
#     BINARY_COLS = [
#         "SIFT_bin",
#         "FAM_COUNT_GE_2",
#         "FAM_SIRIUS_FLAG",
#         "label_of_interest",
#         "PAT_SIRIUS_FLAG",
#     ]
#     for c in BINARY_COLS:
#         if c not in df.columns:
#             df[c] = 0
#         df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

#     # Categorical -> one-hot (if present)
#     # CATEGORICAL_COLS = ["FILTER", "GT", "H2.GT", "CONSTIT.GT", "TUMOR.GT", "VARIANT_CLASS"]
#     CATEGORICAL_COLS = [ "VARIANT_CLASS"]
#     cat_existing = [c for c in CATEGORICAL_COLS if c in df.columns]
#     if cat_existing:
#         df[cat_existing] = df[cat_existing].fillna("NA").astype(str)
#         enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#         cat_matrix = enc.fit_transform(df[cat_existing])
#         cat_df = pd.DataFrame(cat_matrix, columns=enc.get_feature_names_out(cat_existing), index=df.index)
#     else:
#         cat_df = pd.DataFrame(index=df.index)

#     # Labels
#     df["labels"] = make_labels(df)
#     label_df = make_multitask_labels(df)
#     df["label_clinvar"] = label_df["label_clinvar"]
#     df["label_acmg"] = label_df["label_acmg"]

#     # Keys
#     df["CHROM"] = df["CHROM"].apply(_norm_chr) if "CHROM" in df.columns else ""
#     df["POS"] = pd.to_numeric(df.get("POS", np.nan), errors="coerce")

#     df["VARIANT_KEY"] = (
#         df.get("CHROM", "").astype(str)
#         + ":"
#         + df["POS"].astype("Int64").astype(str)
#         + ":"
#         + df.get("REF", "").astype(str)
#         + ":"
#         + df.get("ALT", "").astype(str)
#         + ":"
#         + df.get("TX", "").astype(str)
#     )

#     # keep = [c for c in ["FAMILY_ID", "VARIANT_KEY", "CHROM", "POS", "REF", "ALT", "labels"] if c in df.columns]
#     keep = [
#         c for c in [
#             "FAMILY_ID",
#             "VARIANT_KEY",
#             "CHROM",
#             "POS",
#             "REF",
#             "ALT",
#             "label_clinvar",
#             "label_acmg",
#             "labels"
#         ]
#         if c in df.columns
#     ]
#     df_train = df[df["FAMILY_ID"].astype(str) != str(test_family)].copy()
#     df_test = df[df["FAMILY_ID"].astype(str) == str(test_family)].copy()

#     def make_ml_block(sub_df: pd.DataFrame) -> pd.DataFrame:
#         sub_cat = cat_df.loc[sub_df.index]
#         out = pd.concat([sub_df[keep + NUMERIC_COLS + BINARY_COLS].copy(), sub_cat], axis=1)

#         out["POS"] = pd.to_numeric(out.get("POS", np.nan), errors="coerce")
#         out[BINARY_COLS] = out[BINARY_COLS].fillna(0).astype(int)
#         for c in NUMERIC_COLS:
#             out[c] = pd.to_numeric(out[c], errors="coerce")

#         return out.fillna(0)

#     all_final = make_ml_block(df)
#     train_final = make_ml_block(df_train)
#     test_final = make_ml_block(df_test)

#     ensure_dir(out_dir)
#     all_path = os.path.join(out_dir, "ML_NUMERIC_ALL.tsv")
#     train_path = os.path.join(out_dir, "ML_NUMERIC_TRAIN.tsv")
#     test_path = os.path.join(out_dir, "ML_NUMERIC_TEST.tsv")

#     all_final.to_csv(all_path, sep="\t", index=False)
#     train_final.to_csv(train_path, sep="\t", index=False)
#     test_final.to_csv(test_path, sep="\t", index=False)

#     print("Saved ALL  :", all_path, "| shape:", all_final.shape)
#     print("Saved TRAIN:", train_path, "| shape:", train_final.shape)
#     print(train_final["labels"].value_counts(dropna=False))
#     print("Saved TEST :", test_path, "| shape:", test_final.shape)
#     print(test_final["labels"].value_counts(dropna=False))

#     create_sum_stat(train_final, os.path.join(out_dir, "df_summary_train.txt"))
#     create_sum_stat(test_final, os.path.join(out_dir, "df_summary_test.txt"))

#     return all_final, train_final, test_final


# =============================================================================
# Main builder: TSV -> sequences + numeric + (optional) text/ontology exports
# =============================================================================

def build_ml_file_from_tsv(
    tsv_path: str,
    fasta_path: str,
    out_dir: str,
    test_family: str,
    train_n: int = 5000,
    train_seed: int = 42,
    window: int = 201,
    extra_context: int = 200,
    snv_only: bool = False,
    export_text: bool = False,
    export_onto: bool = False,
    export_textfields: bool = False,
    eval_n: int = 500,
    eval_seed: int = 7,
    eval_mode: str = "random",  # "random" or "stratified_label"
    eval_exclude_from_train: bool = False,
) -> pd.DataFrame:
    """
    1) Read TSV
    2) Ensure FAMILY_ID exists (else create "GLOBAL")
    3) Choose alleles (allele_1/allele_2 if present else REF/ALT)
    4) Create SEQ_REF/SEQ_ALT from FASTA
    5) Save sequence TSVs
    6) (Optional) add ONTOLOGY_TEXT / CLINICAL_TEXT + field columns and export
    7) Save numeric tables
    8) (Optional) create EVAL subset of exactly eval_n variants (not family-based)
    """
    print("Loading TSV:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    # Ensure required minimal columns for sequences
    for c in ["CHROM", "POS", "REF", "ALT"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "FAMILY_ID" not in df.columns:
        df["FAMILY_ID"] = "GLOBAL"
    df["FAMILY_ID"] = df["FAMILY_ID"].fillna("GLOBAL").astype(str).str.strip()

    df["CHROM"] = df["CHROM"].apply(_norm_chr)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    # Choose alleles
    a1 = df["allele_1"].astype(str).str.strip() if "allele_1" in df.columns else pd.Series([""] * len(df), index=df.index)
    a2 = df["allele_2"].astype(str).str.strip() if "allele_2" in df.columns else pd.Series([""] * len(df), index=df.index)

    ref = df["REF"].astype(str).str.strip()
    alt = df["ALT"].astype(str).str.strip()

    def valid_allele(s: pd.Series) -> pd.Series:
        s2 = s.fillna("").astype(str).str.strip()
        return (~s2.isin(list(NA_LIKE))) & (s2 != "")

    # df["ALLELE_REF_USED"] = np.where(valid_allele(a1), a1, ref)
    # df["ALLELE_ALT_USED"] = np.where(valid_allele(a2), a2, alt)
    df["ALLELE_REF_USED"] = df["allele_1"] if "allele_1" in df.columns else df["REF"]
    df["ALLELE_ALT_USED"] = df["allele_2"] if "allele_2" in df.columns else df["ALT"]

    if snv_only:
        before = len(df)
        rl = df["ALLELE_REF_USED"].astype(str).str.len()
        al = df["ALLELE_ALT_USED"].astype(str).str.len()
        df = df[(rl == 1) & (al == 1)].copy()
        print(f"SNV-only enabled: kept {len(df)} / {before}")

    # Build sequences
    fa = Fasta(fasta_path)

    seq_ref_list: List[Optional[str]] = []
    seq_alt_list: List[Optional[str]] = []

    for _, row in df.iterrows():
        chrom = row["CHROM"]
        pos = row["POS"]
        ref_used = str(row["ALLELE_REF_USED"]).strip()
        alt_used = str(row["ALLELE_ALT_USED"]).strip()

        if pd.isna(pos) or ref_used in NA_LIKE or alt_used in NA_LIKE or (not ref_used) or (not alt_used):
            seq_ref_list.append(None)
            seq_alt_list.append(None)
            continue

        s_ref, s_alt = fetch_ref_alt_window_indel_safe(
            fa=fa,
            chrom=chrom,
            pos_1based=int(pos),
            ref=ref_used,
            alt=alt_used,
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

    df["VARIANT_KEY"] = (
        df["CHROM"].astype(str)
        + ":"
        + df["POS"].astype("Int64").astype(str)
        + ":"
        + df["REF"].astype(str)
        + ":"
        + df["ALT"].astype(str)
        + ":"
        + df["TX"].astype(str)
    )

    ensure_dir(out_dir)

    # Split by family (optional behavior controlled by your args; can be dummy)
    df_train = df[df["FAMILY_ID"].astype(str) != str(test_family)].copy()
    df_test = df[df["FAMILY_ID"].astype(str) == str(test_family)].copy()

    # Save sequences
    seq_cols = [
        c for c in [
            "FAMILY_ID",
            "VARIANT_KEY",
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "allele_1",
            "allele_2",
            "ALLELE_REF_USED",
            "ALLELE_ALT_USED",
            "SEQ_REF",
            "SEQ_ALT",
        ]
        if c in df.columns
    ]

    df[seq_cols].to_csv(os.path.join(out_dir, "ML_SEQUENCES_ALL.tsv"), sep="\t", index=False)
    df_train[seq_cols].to_csv(os.path.join(out_dir, "ML_SEQUENCES_TRAIN.tsv"), sep="\t", index=False)
    df_test[seq_cols].to_csv(os.path.join(out_dir, "ML_SEQUENCES_TEST.tsv"), sep="\t", index=False)

    print(
        "Saved sequences:",
        os.path.join(out_dir, "ML_SEQUENCES_ALL.tsv"),
        os.path.join(out_dir, "ML_SEQUENCES_TRAIN.tsv"),
        os.path.join(out_dir, "ML_SEQUENCES_TEST.tsv"),
    )

    # Optional: add text/ontology views and export
    if export_text or export_onto or export_textfields:
        df = add_text_and_ontology_views(df)
        df_train = df.loc[df_train.index].copy()
        df_test = df.loc[df_test.index].copy()

    if export_text:
        text_cols = ["FAMILY_ID", "VARIANT_KEY", "CLINICAL_TEXT", "ONTOLOGY_TEXT"]
        df[text_cols].to_csv(os.path.join(out_dir, "ML_TEXT_ALL.tsv"), sep="\t", index=False)
        df_train[text_cols].to_csv(os.path.join(out_dir, "ML_TEXT_TRAIN.tsv"), sep="\t", index=False)
        df_test[text_cols].to_csv(os.path.join(out_dir, "ML_TEXT_TEST.tsv"), sep="\t", index=False)
        print(
            "Saved text:",
            os.path.join(out_dir, "ML_TEXT_ALL.tsv"),
            os.path.join(out_dir, "ML_TEXT_TRAIN.tsv"),
            os.path.join(out_dir, "ML_TEXT_TEST.tsv"),
        )

    if export_textfields:
        fields_cols = [
            "FAMILY_ID", "VARIANT_KEY",
            "TXT_FUNC", "TXT_DISEASE", "TXT_CLNSIG", "TXT_CLNDN", "TXT_OMIM", "TXT_DISGENET", "TXT_GWAS",
            "TXT_COSMIC", "TXT_CIVIC", "TXT_CGI", "TXT_ONCOKB",
            "ONTO_GO_BP", "ONTO_GO_MF", "ONTO_GO_CC", "ONTO_KEGG", "ONTO_HPO",
        ]
        fields_cols = [c for c in fields_cols if c in df.columns]
        df[fields_cols].to_csv(os.path.join(out_dir, "ML_TEXTFIELDS_ALL.tsv"), sep="\t", index=False)
        df_train[fields_cols].to_csv(os.path.join(out_dir, "ML_TEXTFIELDS_TRAIN.tsv"), sep="\t", index=False)
        df_test[fields_cols].to_csv(os.path.join(out_dir, "ML_TEXTFIELDS_TEST.tsv"), sep="\t", index=False)
        print(
            "Saved text fields:",
            os.path.join(out_dir, "ML_TEXTFIELDS_ALL.tsv"),
            os.path.join(out_dir, "ML_TEXTFIELDS_TRAIN.tsv"),
            os.path.join(out_dir, "ML_TEXTFIELDS_TEST.tsv"),
        )

    if export_onto:
        onto_df = df[["FAMILY_ID", "VARIANT_KEY"]].copy()
        onto_df["GO_BP"] = df.get("GO_biological_process", "").fillna("").astype(str)
        onto_df["GO_MF"] = df.get("GO_molecular_function", "").fillna("").astype(str)
        onto_df["GO_CC"] = df.get("GO_cellular_component", "").fillna("").astype(str)
        onto_df["KEGG"] = df.get("KEGG_Pathway", "").fillna("").astype(str)
        onto_df["HPO"] = df.get("HPO", "").fillna("").astype(str)

        onto_df.to_csv(os.path.join(out_dir, "ML_ONTO_ALL.tsv"), sep="\t", index=False)
        onto_df.loc[df_train.index].to_csv(os.path.join(out_dir, "ML_ONTO_TRAIN.tsv"), sep="\t", index=False)
        onto_df.loc[df_test.index].to_csv(os.path.join(out_dir, "ML_ONTO_TEST.tsv"), sep="\t", index=False)

        print(
            "Saved ontology:",
            os.path.join(out_dir, "ML_ONTO_ALL.tsv"),
            os.path.join(out_dir, "ML_ONTO_TRAIN.tsv"),
            os.path.join(out_dir, "ML_ONTO_TEST.tsv"),
        )

    # Numeric tables (writes ALL/TRAIN/TEST)
    # build_numeric_ml_table(df_in=df, out_dir=out_dir, test_family=test_family)
    build_numeric_ml_table(
        df_in=df,
        out_dir=out_dir,
        test_family=test_family,
        train_n=train_n,
        train_seed=train_seed,
    )

    # -----------------------
    # Optional EVAL subset
    # -----------------------
    if eval_n and eval_n > 0:
        num_all_path = os.path.join(out_dir, "ML_NUMERIC_ALL.tsv")
        if not os.path.exists(num_all_path):
            raise RuntimeError("ML_NUMERIC_ALL.tsv not found; ensure build_numeric_ml_table writes it.")

        df_num_all = pd.read_csv(num_all_path, sep="\t", dtype=str)
        if "labels" in df_num_all.columns:
            df_num_all["labels"] = pd.to_numeric(df_num_all["labels"], errors="coerce").fillna(0).astype(int)

        eval_keys = sample_eval_keys(df_num_all, n=eval_n, seed=eval_seed, mode=eval_mode)
        eval_keys_set = set(eval_keys)

        def write_eval(src_path: str, out_name: str) -> None:
            if not os.path.exists(src_path):
                return
            d = pd.read_csv(src_path, sep="\t", dtype=str)
            if "VARIANT_KEY" not in d.columns:
                return
            d_eval = d[d["VARIANT_KEY"].astype(str).isin(eval_keys_set)].copy()
            d_eval.to_csv(os.path.join(out_dir, out_name), sep="\t", index=False)
            print("Saved EVAL:", out_name, "| n=", len(d_eval))

        write_eval(os.path.join(out_dir, "ML_SEQUENCES_ALL.tsv"), "ML_SEQUENCES_EVAL500.tsv")
        write_eval(os.path.join(out_dir, "ML_NUMERIC_ALL.tsv"), "ML_NUMERIC_EVAL500.tsv")
        write_eval(os.path.join(out_dir, "ML_TEXT_ALL.tsv"), "ML_TEXT_EVAL500.tsv")
        write_eval(os.path.join(out_dir, "ML_TEXTFIELDS_ALL.tsv"), "ML_TEXTFIELDS_EVAL500.tsv")
        write_eval(os.path.join(out_dir, "ML_ONTO_ALL.tsv"), "ML_ONTO_EVAL500.tsv")

        if eval_exclude_from_train:
            def rewrite_train(src_path: str) -> None:
                if not os.path.exists(src_path):
                    return
                d = pd.read_csv(src_path, sep="\t", dtype=str)
                if "VARIANT_KEY" not in d.columns:
                    return
                d2 = d[~d["VARIANT_KEY"].astype(str).isin(eval_keys_set)].copy()
                d2.to_csv(src_path, sep="\t", index=False)
                print("Rewrote TRAIN (excluded eval):", os.path.basename(src_path), "| n=", len(d2))

            rewrite_train(os.path.join(out_dir, "ML_SEQUENCES_TRAIN.tsv"))
            rewrite_train(os.path.join(out_dir, "ML_NUMERIC_TRAIN.tsv"))
            rewrite_train(os.path.join(out_dir, "ML_TEXT_TRAIN.tsv"))
            rewrite_train(os.path.join(out_dir, "ML_TEXTFIELDS_TRAIN.tsv"))
            rewrite_train(os.path.join(out_dir, "ML_ONTO_TRAIN.tsv"))

        with open(os.path.join(out_dir, "EVAL500_KEYS.txt"), "w", encoding="utf-8") as f:
            for k in eval_keys:
                f.write(k + "\n")
        print("Saved EVAL key list:", os.path.join(out_dir, "EVAL500_KEYS.txt"))

    return df


# =============================================================================
# Optional: PyTorch dataset + collate + field-wise pooling (Model B2)
# =============================================================================

class VariantMultiViewDataset(Dataset):
    """
    Joins:
      - ML_SEQUENCES_*.tsv
      - ML_NUMERIC_*.tsv
      - ML_TEXT_*.tsv (optional, Model B1)
      - ML_TEXTFIELDS_*.tsv (optional, Model B2)
      - ML_ONTO_*.tsv (optional)

    Each sample:
      {
        "seq_ref": str,
        "seq_alt": str,
        "bio": FloatTensor(p,),
        "label": int,
        "text": str (optional, Model B1),
        "text_fields": dict[str,str] (optional, Model B2),
        "onto": dict[str,set] (optional, for T_sb),
        "meta": {...}
      }
    """

    def __init__(
        self,
        seq_tsv: str,
        bio_tsv: str,
        text_tsv: Optional[str] = None,
        textfields_tsv: Optional[str] = None,
        onto_tsv: Optional[str] = None,
        bio_feature_cols: Optional[List[str]] = None,
        return_text: bool = False,
        return_textfields: bool = False,
        return_onto: bool = False,
    ):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to use VariantMultiViewDataset.")

        self.return_text = return_text
        self.return_textfields = return_textfields
        self.return_onto = return_onto

        df_seq = pd.read_csv(seq_tsv, sep="\t", dtype=str)
        df_bio = pd.read_csv(bio_tsv, sep="\t", dtype=str)

        on = ["FAMILY_ID", "VARIANT_KEY"]
        if not all(c in df_seq.columns for c in on) or not all(c in df_bio.columns for c in on):
            raise ValueError(f"Join keys {on} must exist in both seq and bio TSVs.")

        df = df_seq.merge(df_bio, on=on, how="inner", suffixes=("", "_BIO"))

        if return_text:
            if not (text_tsv and os.path.exists(text_tsv)):
                raise ValueError("return_text=True but text_tsv is missing.")
            df_text = pd.read_csv(text_tsv, sep="\t", dtype=str)
            need = ["FAMILY_ID", "VARIANT_KEY", "CLINICAL_TEXT"]
            if not all(c in df_text.columns for c in need):
                raise ValueError(f"text_tsv must have {need}.")
            df = df.merge(df_text[need], on=on, how="left")

        if return_textfields:
            if not (textfields_tsv and os.path.exists(textfields_tsv)):
                raise ValueError("return_textfields=True but textfields_tsv is missing.")
            df_tf = pd.read_csv(textfields_tsv, sep="\t", dtype=str)
            need = ["FAMILY_ID", "VARIANT_KEY", "TXT_FUNC", "TXT_DISEASE", "ONTO_GO_BP"]
            if not all(c in df_tf.columns for c in need):
                raise ValueError(f"textfields_tsv must have at least {need}.")
            df = df.merge(df_tf, on=on, how="left")

        self.has_onto = False
        if return_onto:
            if not (onto_tsv and os.path.exists(onto_tsv)):
                raise ValueError("return_onto=True but onto_tsv is missing.")
            df_onto = pd.read_csv(onto_tsv, sep="\t", dtype=str)
            if not all(c in df_onto.columns for c in ["FAMILY_ID", "VARIANT_KEY"]):
                raise ValueError("onto_tsv must have FAMILY_ID and VARIANT_KEY.")
            keep_cols = ["FAMILY_ID", "VARIANT_KEY", "GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]
            keep_cols = [c for c in keep_cols if c in df_onto.columns]
            df = df.merge(df_onto[keep_cols], on=on, how="left")
            self.has_onto = True

        if bio_feature_cols is None:
            drop = {
                "FAMILY_ID", "VARIANT_KEY", "CHROM", "POS", "REF", "ALT",
                "allele_1", "allele_2", "labels",
                "SEQ_REF", "SEQ_ALT", "ALLELE_REF_USED", "ALLELE_ALT_USED",
                # Text
                "CLINICAL_TEXT", "ONTOLOGY_TEXT",
                "TXT_FUNC", "TXT_DISEASE", "TXT_CLNSIG", "TXT_CLNDN", "TXT_OMIM", "TXT_DISGENET", "TXT_GWAS",
                "TXT_COSMIC", "TXT_CIVIC", "TXT_CGI", "TXT_ONCOKB",
                "ONTO_GO_BP", "ONTO_GO_MF", "ONTO_GO_CC", "ONTO_KEGG", "ONTO_HPO",
                # Ontology export
                "GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO",
            }
            bio_feature_cols = [c for c in df.columns if c not in drop]
        self.bio_feature_cols = bio_feature_cols

        df = df.reset_index(drop=True)
        self.df = df

        X = df[self.bio_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values
        self.X_bio = X

        y = pd.to_numeric(df.get("labels", 0), errors="coerce").fillna(0).astype(int).values
        self.y = y

        if self.has_onto:
            self.onto_list = []
            for _, r in df.iterrows():
                self.onto_list.append({
                    "go_bp": set(split_terms(r.get("GO_BP", ""))),
                    "go_mf": set(split_terms(r.get("GO_MF", ""))),
                    "go_cc": set(split_terms(r.get("GO_CC", ""))),
                    "kegg": set(split_terms(r.get("KEGG", ""))),
                    "hpo": set(split_terms(r.get("HPO", ""))),
                })
        else:
            self.onto_list = [None] * len(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        out: Dict[str, Any] = {
            "seq_ref": str(r.get("SEQ_REF", "")),
            "seq_alt": str(r.get("SEQ_ALT", "")),
            "bio": torch.from_numpy(self.X_bio[idx]),
            "label": int(self.y[idx]),
            "meta": {
                "family_id": str(r.get("FAMILY_ID", "")),
                "variant_key": str(r.get("VARIANT_KEY", "")),
                "chrom": str(r.get("CHROM", "")),
                "pos": str(r.get("POS", "")),
                "ref": str(r.get("REF", "")),
                "alt": str(r.get("ALT", "")),
            },
        }

        if self.return_text:
            out["text"] = str(r.get("CLINICAL_TEXT", ""))

        if self.return_textfields:
            out["text_fields"] = {
                "FUNC": str(r.get("TXT_FUNC", "")),
                "DISEASE": str(r.get("TXT_DISEASE", "")),
                "CLNSIG": str(r.get("TXT_CLNSIG", "")),
                "CLNDN": str(r.get("TXT_CLNDN", "")),
                "OMIM": str(r.get("TXT_OMIM", "")),
                "DISGENET": str(r.get("TXT_DISGENET", "")),
                "GWAS": str(r.get("TXT_GWAS", "")),
                "COSMIC": str(r.get("TXT_COSMIC", "")),
                "CIVIC": str(r.get("TXT_CIVIC", "")),
                "CGI": str(r.get("TXT_CGI", "")),
                "ONCOKB": str(r.get("TXT_ONCOKB", "")),
            }

        if self.return_onto and self.has_onto:
            out["onto"] = self.onto_list[idx]

        return out


def collate_multiview_batch(
    batch: List[Dict[str, Any]],
    tokenizer_seq=None,
    tokenizer_text=None,
    include_text: bool = False,
    include_textfields: bool = False,
    build_T_sb: bool = False,
    build_T_st: bool = False,
    build_T_bt: bool = False,
    T_sb_mode: str = "ontology",   # "ontology" or "identity"
    T_text_mode: str = "tfidf",    # "tfidf" or "identity"
    alpha_onto: float = 0.3,
    alpha_text: float = 0.3,
    zero_diag: bool = False,
    onto_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Collate returns:
      {
        "seq": token dict OR raw strings,
        "bio": (N,p) float tensor,
        "text": token dict OR raw strings (optional, Model B1),
        "text_fields": dict[field]->list[str] (optional, Model B2),
        "T": {"sb","st","bt"} (optional),
        "labels": (N,) long,
        "meta": list
      }
    """
    if torch is None:
        raise RuntimeError("PyTorch not available.")

    seq_ref = [b.get("seq_ref", "") for b in batch]
    seq_alt = [b.get("seq_alt", "") for b in batch]
    x_bio = torch.stack([b["bio"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    labels = torch.tensor([b.get("label", 0) for b in batch], dtype=torch.long)

    if tokenizer_seq is not None:
        seq_tokens = tokenizer_seq(seq_alt, padding=True, truncation=True, return_tensors="pt")
    else:
        seq_tokens = {"seq_alt": seq_alt, "seq_ref": seq_ref}

    out: Dict[str, Any] = {"seq": seq_tokens, "bio": x_bio, "labels": labels, "meta": meta}

    texts: Optional[List[str]] = None
    if include_text:
        texts = [b.get("text", "") for b in batch]
        if tokenizer_text is not None:
            out["text"] = tokenizer_text(texts, padding=True, truncation=True, return_tensors="pt")
        else:
            out["text"] = {"text": texts}

    if include_textfields:
        fields = [b.get("text_fields", {}) for b in batch]
        keys = ["FUNC", "DISEASE", "CLNSIG", "CLNDN", "OMIM", "DISGENET", "GWAS", "COSMIC", "CIVIC", "CGI", "ONCOKB"]
        out["text_fields"] = {k: [str(f.get(k, "")) for f in fields] for k in keys}

    T: Dict[str, Any] = {}
    N = len(batch)

    def eye() -> "torch.Tensor":
        return torch.eye(N, dtype=torch.float32)

    if build_T_sb:
        if T_sb_mode == "ontology":
            onto = [b.get("onto", None) for b in batch]
            if any(o is None for o in onto):
                T["sb"] = eye()
            else:
                T_np = soft_targets_from_ontology_sets(
                    onto_sets=onto,
                    weights=onto_weights,
                    alpha=alpha_onto,
                    zero_diag=zero_diag,
                )
                T["sb"] = torch.from_numpy(T_np)
        else:
            T["sb"] = eye()

    if include_text and build_T_st:
        if T_text_mode == "tfidf":
            assert texts is not None
            T_np = soft_targets_from_text_tfidf(texts, alpha=alpha_text, zero_diag=zero_diag)
            T["st"] = torch.from_numpy(T_np)
        else:
            T["st"] = eye()

    if include_text and build_T_bt:
        if T_text_mode == "tfidf":
            assert texts is not None
            T_np = soft_targets_from_text_tfidf(texts, alpha=alpha_text, zero_diag=zero_diag)
            T["bt"] = torch.from_numpy(T_np)
        else:
            T["bt"] = eye()

    if T:
        out["T"] = T

    return out


def fieldwise_text_pooling(
    field_embeddings: Dict[str, "torch.Tensor"],
    field_masks: Dict[str, "torch.Tensor"],
    weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> "torch.Tensor":
    """
    Reference pooling for Model B2 (article method):

      z = sum_k w_k * m_k * e_k / (sum_k w_k * m_k + eps)
    """
    if torch is None:
        raise RuntimeError("PyTorch not available.")

    if weights is None:
        weights = {
            "FUNC": 1.0,
            "DISEASE": 1.0,
            "CLNSIG": 1.5,
            "CLNDN": 1.2,
            "OMIM": 1.2,
            "DISGENET": 0.6,
            "GWAS": 0.6,
            "COSMIC": 1.0,
            "CIVIC": 1.5,
            "CGI": 1.2,
            "ONCOKB": 2.0,
        }

    keys = [k for k in field_embeddings.keys() if k in field_masks]
    if not keys:
        raise ValueError("No overlapping keys between field_embeddings and field_masks.")

    num = None
    den = None

    for k in keys:
        e = field_embeddings[k]                 # (N, d)
        m = field_masks[k].float().view(-1, 1)  # (N, 1)
        w = float(weights.get(k, 1.0))

        contrib = w * m * e
        if num is None:
            num = contrib
            den = w * m
        else:
            num = num + contrib
            den = den + (w * m)

    assert num is not None and den is not None
    return num / (den + eps)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--tsv", required=True, help="Input annotated TSV (germline or somatic)")
    ap.add_argument("--fasta", required=True, help="Reference FASTA path")
    ap.add_argument("--out_dir", required=True, help="Output folder")

    ap.add_argument(
        "--test_family",
        required=True,
        help="Family ID held out as test (if missing FAMILY_ID -> test will be empty). "
             "If you don't want a family test split, pass a dummy ID not present.",
    )

    ap.add_argument("--window", type=int, default=201, help="Fixed sequence window length (odd recommended)")
    ap.add_argument("--extra_context", type=int, default=200, help="Extra context for indel-safe windows")
    ap.add_argument("--snv_only", action="store_true", help="Keep only SNVs based on ALLELE_REF_USED/ALLELE_ALT_USED")

    ap.add_argument("--export_text", action="store_true", help="Export ML_TEXT_*.tsv with CLINICAL_TEXT + ONTOLOGY_TEXT (Model B1)")
    ap.add_argument("--export_onto", action="store_true", help="Export ML_ONTO_*.tsv (GO_BP/GO_MF/GO_CC/KEGG/HPO) for T_sb")
    ap.add_argument("--export_textfields", action="store_true", help="Export ML_TEXTFIELDS_*.tsv for field-wise pooling (Model B2)")

    
    #train subset
    ap.add_argument(
        "--train_n",
        type=int,
        default=0,
        help="Number of training variants to keep (0 = use full training set)"
    )

    ap.add_argument(
        "--train_seed",
        type=int,
        default=42,
        help="Random seed for training subsampling"
    )


    # Eval subset (not family-based)
    ap.add_argument("--eval_n", type=int, default=500, help="Number of variants for evaluation subset (0 disables)")
    ap.add_argument("--eval_seed", type=int, default=7, help="Random seed for eval subset sampling")
    ap.add_argument("--eval_mode", choices=["random", "stratified_label"], default="random", help="How to pick eval variants")
    ap.add_argument("--eval_exclude_from_train", action="store_true", help="Remove eval variants from TRAIN outputs to avoid leakage")

    args = ap.parse_args()

    build_ml_file_from_tsv(
        tsv_path=args.tsv,
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        test_family=args.test_family,
        train_n=args.train_n,
        train_seed=args.train_seed,
        window=args.window,
        extra_context=args.extra_context,
        snv_only=args.snv_only,
        export_text=args.export_text,
        export_onto=args.export_onto,
        export_textfields=args.export_textfields,
        eval_n=args.eval_n,
        eval_seed=args.eval_seed,
        eval_mode=args.eval_mode,
        eval_exclude_from_train=args.eval_exclude_from_train,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()



# bash
# python build_multiview_ml.py \
#   --tsv your.tsv --fasta hg38.fa --out_dir OUT \
#   --test_family __NO_FAMILY_TEST__ \
#   --eval_n 500 --eval_seed 7 --eval_mode stratified_label \
#   --export_text --export_textfields --export_onto

