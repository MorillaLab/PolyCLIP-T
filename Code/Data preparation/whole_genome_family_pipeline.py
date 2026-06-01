#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcript-aware EXOME Family Pipeline

Design goals
------------
1) For each source block (germline / somatic genetic / somatic oncology):
       VCF + EXOME -> add CNV -> add HET
2) For each somatic run:
       merge somatic genetic + somatic oncology on transcript-aware key
3) For each patient:
       merge all somatic runs together
       merge consolidated somatic block with consolidated germline block
4) Avoid uncontrolled duplicate columns by:
       - preserving source-specific raw payload with prefixes
       - creating a small normalized shared layer used downstream
5) Parse ANN fields from each VCF header independently.
   Do NOT force one universal ANN schema across germline and somatic VCFs.

Key
---
    (CHROM, POS, REF, ALT, TX)
where:
    - TX from VCF is Feature
    - TX from EXOME is Feature_RefSeq

Notes
-----
- allele_1 / allele_2 always come from the VCF genotype.
- all parsed ANN-derived VCF columns are preserved.
- germline and somatic VCFs may have different ANN layouts; this script handles that.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Constants / keys
# =========================================================

EMPTY_STR_SET = {"", ".", "nan", "None", "NA", "na", "N/A", "null", "NULL"}
KEY_COLS = ["CHROM", "POS", "REF", "ALT", "TX"]
PAT_KEY_COLS = ["FAMILY_ID", "PATIENT_ID"] + KEY_COLS
FAM_KEY_COLS = ["FAMILY_ID"] + KEY_COLS
GLOBAL_KEY_COLS = KEY_COLS

# Last-resort fallback only when an ANN header cannot be parsed.
ANN_FIELDS_FALLBACK = [
    "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type", "Feature", "BIOTYPE",
    "EXON", "INTRON", "HGVSc", "HGVSp", "cDNA_position", "CDS_position", "Protein_position",
    "Amino_acids", "Codons", "Existing_variation", "DISTANCE", "STRAND", "FLAGS", "VARIANT_CLASS",
    "SYMBOL_SOURCE", "HGNC_ID", "CANONICAL", "MANE", "TSL", "APPRIS", "CCDS", "ENSP", "SWISSPROT",
    "TREMBL", "UNIPARC", "REFSEQ_MATCH", "REFSEQ_OFFSET", "SOURCE", "GENE_PHENO", "SIFT", "PolyPhen",
    "DOMAINS", "miRNA", "HGVS_OFFSET", "HGVSg", "AF", "AFR_AF", "AMR_AF", "EAS_AF", "EUR_AF", "SAS_AF",
    "AA_AF", "EA_AF", "gnomAD_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF", "gnomAD_EAS_AF",
    "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF", "MAX_AF", "MAX_AF_POPS",
    "CLIN_SIG", "SOMATIC", "PHENO", "PUBMED", "VAR_SYNONYMS", "MOTIF_NAME", "MOTIF_POS", "HIGH_INF_POS",
    "MOTIF_SCORE_CHANGE", "TRANSCRIPTION_FACTORS", "MaxEntScan_alt", "MaxEntScan_diff", "MaxEntScan_ref",
    "ada_score", "rf_score", "GERP_RS", "Interpro_domain", "LRT_pred", "M_CAP_pred", "MetaLR_pred",
    "MetaSVM_pred", "MutationAssessor_pred", "MutationTaster_pred", "MutationTaster_score",
    "PROVEAN_score", "DisGeNET_PMID", "DisGeNET_SCORE", "DisGeNET_disease", "EVS", "EVS_MAF", "EVS_CA",
    "mir_ACC", "clinvar", "clinvar_CLNSIG", "clinvar_CLNDN", "clinvar_ORIGIN", "clinvar_CLNDISDB",
    "IG", "IG_AF", "IG_RNA", "IG_RNA_AF_RNA", "Kaviar", "Kaviar_AF", "cosmic", "cosmic_FATHMM",
    "cosmic_LEGACY_ID", "PhyloP", "PhastCons", "GnomAD_Genome", "GnomAD_Genome_AF",
    "GnomAD_Genome_AF_afr", "GnomAD_Genome_AF_amr", "GnomAD_Genome_AF_asj", "GnomAD_Genome_AF_eas",
    "GnomAD_Genome_AF_fin", "GnomAD_Genome_AF_nfe", "GnomAD_Genome_AF_oth", "GnomAD_Genome_AF_sas",
    "GnomAD_Genome_AF_popmax", "FATHMM", "FATHMM_NCS", "FATHMM_NCG", "FATHMM_CS", "FATHMM_CG",
    "DANN", "DANN_Score", "CIViC_Variant", "CIViC_Variant_Clinical_Information",
    "CIViC_Region_Clinical_Information", "Chasmplus", "Chasmplus_CHASMPLUS_TTYPE",
    "GnomAD_MNV", "GnomAD_MNV_AF",
]

# Rich text-like columns that should be uniq-joined during collapse.
JOIN_COLS_RICH = {
    "TX", "GENE_NAME", "Feature", "Feature_RefSeq", "Consequence", "IMPACT", "VARIANT_CLASS",
    "KEGG_Gene", "KEGG_Pathway", "Trait_association(GWAS)", "DrugDB", "DisGeNET", "HPO", "OMIM", "PUBMED",
    "Function_description", "Disease_description", "GO_biological_process", "GO_cellular_component",
    "GO_molecular_function", "Tissue_specificity(Uniprot)", "DOMAINS", "MOTIF_NAME",
    "Cancer_Genes", "Cancer_Genes_Oncokb", "GENE_IN_ONCOKB", "OncoKB_NM",
    "CIViC_Variant_clinical_significance", "CIViC_Variant_evidence_level", "CIViC_Variant_disease",
    "CIViC_Variant_drugs", "CIViC_Region_clinical_significance", "CanDL_PMIDs", "CanDL_Cancer_Type",
    "CancerGenomeInterpreter_Association", "CancerGenomeInterpreter_Evidence_level", "VARIANT_IN_ONCOKB",
    "MUTATION_EFFECT", "CITATIONS", "Hotspots_Kit", "FILTER", "ClinVar_CLNSIG", "ClinVar_CLNDN",
    "ClinVar_CLNDISDB", "clinvar_CLNSIG", "clinvar_CLNDN", "clinvar_CLNDISDB", "COSMIC", "ONCOGENIC",
    "HIGHEST_LEVEL", "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4", "SOURCE_SET",
}

# Small stable layer shared across sources and used downstream.
SHARED_NORMALIZED_COLS = [
    "GENE_NAME", "Consequence", "IMPACT", "VARIANT_CLASS", "MAX_POP_AF", "CLNSIG_UNI",
    "DP_UNI", "VAF_UNI", "ACMG_INT", "allele_1", "allele_2", "PATHWAY_HIT", "PATH_DNA_REPAIR",
    "CNV_HIT", "CNV_pathogenic", "LOH_PARTIAL_PATHO", "IN_ROH", "HET_HIGH_HOM",
]


# =========================================================
# Generic helpers
# =========================================================


def normalize_chr(x) -> str:
    return str(x).replace("chr", "").replace("CHR", "").strip()


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def safe_save(df: pd.DataFrame, path: str, sep: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, sep=sep, index=False)
    print(f"[SAVED] {path} | rows={len(df)} cols={df.shape[1]}")


def chrom_sort_key(ch: str):
    s = normalize_chr(ch)
    mapping = {str(i): i for i in range(1, 23)}
    mapping.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
    return mapping.get(s, 1000), s


def sort_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out["_chr_rank"] = out["CHROM"].map(lambda x: chrom_sort_key(x)[0])
    out["_chr_str"] = out["CHROM"].map(lambda x: chrom_sort_key(x)[1])
    sort_cols = [c for c in ["_chr_rank", "_chr_str", "POS", "REF", "ALT", "TX"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").drop(columns=["_chr_rank", "_chr_str"], errors="ignore")
    return out


def list_files_one(dirpath: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    if not dirpath or not os.path.isdir(dirpath):
        return None
    hits = [os.path.join(dirpath, fn) for fn in os.listdir(dirpath) if fn.endswith(suffixes)]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(f"Expected one file in {dirpath} with {suffixes}, found {len(hits)}:\n" + "\n".join(hits))
    return hits[0]


def pick_single_txt_from_dir(path_or_dir: Optional[str]) -> Optional[str]:
    if not path_or_dir:
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        hits = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.lower().endswith(".txt")]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(f"Expected one .txt in {path_or_dir}, found {len(hits)}:\n" + "\n".join(hits))
        return hits[0]
    return None


def pick_single_vcf_from_dir(path_or_dir: Optional[str]) -> Optional[str]:
    if not path_or_dir:
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        hits = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith((".vcf", ".vcf.gz"))]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(f"Expected one VCF in {path_or_dir}, found {len(hits)}:\n" + "\n".join(hits))
        return hits[0]
    return None


def first_nonnull(s: pd.Series):
    t = s.dropna()
    if t.empty:
        return np.nan
    t = t[~t.astype(str).isin(list(EMPTY_STR_SET))]
    return t.iloc[0] if len(t) else np.nan


def uniq_join(s: pd.Series) -> str:
    vals = [str(x) for x in s.dropna().astype(str).tolist()]
    vals = [v for v in vals if v not in EMPTY_STR_SET]
    return "|".join(sorted(set(vals)))


def num_max_or_first(x: pd.Series):
    xn = pd.to_numeric(x, errors="coerce")
    if xn.notna().any():
        return float(xn.max())
    return first_nonnull(x)


def build_keep_all_payload_aggs(df: pd.DataFrame, key_cols: List[str], fixed_aggs: Dict[str, tuple],
                                join_cols: Optional[set] = None, drop_cols: Optional[set] = None) -> Dict[str, tuple]:
    join_cols = join_cols or set()
    drop_cols = drop_cols or set()
    agg = dict(fixed_aggs)
    skip = set(key_cols) | set(agg.keys()) | set(drop_cols)
    for c in df.columns:
        if c in skip:
            continue
        agg[c] = (c, uniq_join if c in join_cols else num_max_or_first)
    return agg


# def _coalesce_columns(df: pd.DataFrame, preferred: str, alternatives: List[str]) -> pd.DataFrame:
#     out = df.copy()
#     if preferred not in out.columns:
#         out[preferred] = np.nan
#     mask = out[preferred].isna() | out[preferred].astype(str).isin(EMPTY_STR_SET)
#     for c in alternatives:
#         if c in out.columns:
#             out.loc[mask, preferred] = out.loc[mask, c]
#             mask = out[preferred].isna() | out[preferred].astype(str).isin(EMPTY_STR_SET)
#     return out

def _coalesce_columns(df: pd.DataFrame, preferred: str, alternatives: List[str]) -> pd.DataFrame:
    out = df.copy()

    if preferred not in out.columns:
        out[preferred] = pd.Series([None] * len(out), index=out.index, dtype="object")
    else:
        out[preferred] = out[preferred].astype("object")

    mask = out[preferred].isna() | out[preferred].astype(str).isin(EMPTY_STR_SET)
    for c in alternatives:
        if c in out.columns:
            vals = out[c].astype("object")
            out.loc[mask, preferred] = vals.loc[mask]
            mask = out[preferred].isna() | out[preferred].astype(str).isin(EMPTY_STR_SET)

    return out


# =========================================================
# EXOME loading
# =========================================================


def load_exome_table(exome_path: str) -> pd.DataFrame:
    if not exome_path or not os.path.exists(exome_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(exome_path, sep="\t", dtype=str, low_memory=False)
    except Exception:
        df = pd.read_csv(exome_path, sep=r"\s+", engine="python", dtype=str, low_memory=False)

    if "CHROM" in df.columns:
        df["CHROM"] = df["CHROM"].map(normalize_chr)
    elif "chr" in df.columns:
        df = df.rename(columns={"chr": "CHROM"})
        df["CHROM"] = df["CHROM"].map(normalize_chr)

    if "POS" in df.columns:
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    elif "pos" in df.columns:
        df = df.rename(columns={"pos": "POS"})
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    for c in ["REF", "ALT"]:
        if c not in df.columns:
            df[c] = ""
    return df


def add_tx_from_exome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    src = None
    for c in ["Feature_RefSeq", "Transcript", "NM", "Feature"]:
        if c in out.columns:
            src = c
            break
    if src is None:
        out["TX"] = "NO_TX"
    else:
        out["TX"] = out[src].astype(str).replace({"": np.nan, ".": np.nan, "nan": np.nan, "None": np.nan}).fillna("NO_TX")
    return out


# =========================================================
# VCF loading / ANN parsing
# =========================================================


def parse_ann_fields_from_header(vcf_path: str) -> List[str]:
    """
    Parse the ANN schema from this specific VCF header.
    Germline and somatic VCFs can differ; this function treats each VCF separately.
    """
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##INFO=<ID=ANN"):
                m = re.search(r"Functional annotations:\s*'([^']+)'", line)
                if m:
                    fields = [x.strip() for x in m.group(1).split("|")]
                    if fields:
                        return fields
                m = re.search(r"Annotation(?:s)?[:=]\s*'([^']+)'", line)
                if m:
                    fields = [x.strip() for x in m.group(1).split("|")]
                    if fields:
                        return fields
                break
            if line.startswith("#CHROM"):
                break
    print(f"[WARN] ANN header schema not found in {vcf_path}; using fallback ANN fields")
    return ANN_FIELDS_FALLBACK


def parse_vcf_header(vcf_path: str):
    info_ids, format_ids, header_cols = [], [], None
    info_pat = re.compile(r'##INFO=<ID=([^,>]+)')
    format_pat = re.compile(r'##FORMAT=<ID=([^,>]+)')
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("##INFO="):
                m = info_pat.match(line)
                if m:
                    info_ids.append(m.group(1))
            elif line.startswith("##FORMAT="):
                m = format_pat.match(line)
                if m:
                    format_ids.append(m.group(1))
            elif line.startswith("#CHROM"):
                header_cols = line.split("\t")
                break
    if header_cols is None:
        raise ValueError(f"VCF header not found in {vcf_path}")
    return info_ids, format_ids, header_cols


def load_vcf_records(vcf_path: str) -> pd.DataFrame:
    rows, header_cols = [], None
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_cols = line.rstrip("\n").split("\t")
                continue
            rows.append(line.rstrip("\n").split("\t"))
    if header_cols is None:
        raise ValueError(f"VCF #CHROM header not found in {vcf_path}")
    df = pd.DataFrame(rows, columns=header_cols).rename(columns={"#CHROM": "CHROM"})
    df["CHROM"] = df["CHROM"].astype(str).map(normalize_chr)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df["REF"] = df["REF"].astype(str)
    df["ALT"] = df["ALT"].astype(str)
    df["ALT_RAW"] = df["ALT"]
    return df


def split_info_column(df: pd.DataFrame, info_ids: List[str]) -> pd.DataFrame:
    out = df.copy()

    def parse_info(info_str: str):
        d = {}
        if pd.isna(info_str) or str(info_str).strip() in {"", "."}:
            return d
        for item in str(info_str).split(";"):
            if not item:
                continue
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
            else:
                d[item] = True
        return d

    parsed = out["INFO"].map(parse_info)
    for key in info_ids:
        out[key] = parsed.map(lambda x: x.get(key, np.nan))
    return out


def detect_sample_prefix(df: pd.DataFrame) -> Optional[str]:
    dp_cols = [c for c in df.columns if c.endswith(".DP")]
    return dp_cols[0].rsplit(".", 1)[0] if dp_cols else None


def split_sample_format(df: pd.DataFrame, sample_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    fixed = {"CHROM", "POS", "ID", "REF", "ALT", "ALT_RAW", "QUAL", "FILTER", "INFO", "FORMAT"}
    sample_candidates = [c for c in out.columns if c not in fixed]
    if sample_col is None:
        sample_col = sample_candidates[0] if sample_candidates else None
    if sample_col is None or "FORMAT" not in out.columns:
        return out

    format_keys = out["FORMAT"].fillna("").astype(str).str.split(":")
    sample_vals = out[sample_col].fillna("").astype(str).str.split(":")
    all_keys = sorted(set(k for row in format_keys for k in row if k))
    for key in all_keys:
        vals = []
        for keys, vals_row in zip(format_keys, sample_vals):
            try:
                idx = keys.index(key)
                vals.append(vals_row[idx] if idx < len(vals_row) else np.nan)
            except ValueError:
                vals.append(np.nan)
        out[f"{sample_col}.{key}"] = vals
    return out


def parse_ann_entry(entry: str, ann_fields: List[str]) -> Dict[str, str]:
    parts = str(entry).split("|")
    if len(parts) < len(ann_fields):
        parts += [""] * (len(ann_fields) - len(parts))
    elif len(parts) > len(ann_fields):
        parts = parts[:len(ann_fields)]
    return dict(zip(ann_fields, parts))


def build_alleles_from_gt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prefix = detect_sample_prefix(out)
    gt_col = "GT" if "GT" in out.columns else (f"{prefix}.GT" if prefix and f"{prefix}.GT" in out.columns else None)

    out["GT_ONLY"] = np.nan
    out["ALLELE_1_INDEX"] = np.nan
    out["ALLELE_2_INDEX"] = np.nan
    out["allele_1"] = np.nan
    out["allele_2"] = np.nan

    if gt_col is None:
        return out

    def extract_gt(x):
        return "" if pd.isna(x) else str(x).split(":")[0]

    def allele_from_index(idx, ref, alt_raw):
        if idx in {"", "."}:
            return np.nan
        try:
            i = int(idx)
        except Exception:
            return np.nan
        alts = str(alt_raw).split(",")
        if i == 0:
            return ref
        if 1 <= i <= len(alts):
            return alts[i - 1]
        return np.nan

    gt_only = out[gt_col].map(extract_gt)
    a1_idx, a2_idx, a1_base, a2_base = [], [], [], []
    for gt, ref, alt_raw in zip(gt_only, out["REF"], out["ALT_RAW"]):
        if gt in {"", ".", "./.", ".|."}:
            a1_idx.append(np.nan); a2_idx.append(np.nan); a1_base.append(np.nan); a2_base.append(np.nan)
            continue
        sep = "/" if "/" in gt else "|" if "|" in gt else None
        if sep is None:
            a1_idx.append(np.nan); a2_idx.append(np.nan); a1_base.append(np.nan); a2_base.append(np.nan)
            continue
        parts = gt.split(sep)
        i1 = parts[0] if len(parts) > 0 else np.nan
        i2 = parts[1] if len(parts) > 1 else np.nan
        a1_idx.append(i1)
        a2_idx.append(i2)
        a1_base.append(allele_from_index(i1, ref, alt_raw))
        a2_base.append(allele_from_index(i2, ref, alt_raw))

    out["GT_ONLY"] = gt_only
    out["ALLELE_1_INDEX"] = a1_idx
    out["ALLELE_2_INDEX"] = a2_idx
    out["allele_1"] = a1_base
    out["allele_2"] = a2_base
    return out


def explode_alt_and_link_ann(df: pd.DataFrame, ann_fields: List[str], ann_col: str = "ANN") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        alt_list = str(row["ALT_RAW"]).split(",") if pd.notna(row["ALT_RAW"]) else [row["ALT"]]
        ann_entries = []
        ann_value = row.get(ann_col, np.nan)
        if pd.notna(ann_value) and str(ann_value) not in {"", "."}:
            ann_entries = [parse_ann_entry(x, ann_fields) for x in str(ann_value).split(",") if x != ""]
        for alt_idx, alt in enumerate(alt_list, start=1):
            alt_ann = [a for a in ann_entries if str(a.get("Allele", "")) == str(alt)] or [None]
            for ann_dict in alt_ann:
                rec = row.to_dict()
                rec["ALT"] = alt
                rec["ALT_INDEX"] = alt_idx
                if ann_dict is not None:
                    rec.update(ann_dict)
                rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out["REF"] = out["REF"].astype(str)
    out["ALT"] = out["ALT"].astype(str)
    return out


def add_common_ann_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize only a small shared subset after source-specific ANN parsing.
    This avoids assuming germline and somatic ANN payloads are identical.
    """
    out = df.copy()

    if "Feature" in out.columns:
        out["TX"] = out["Feature"].astype(str).replace({"": np.nan, ".": np.nan, "nan": np.nan, "None": np.nan}).fillna("NO_TX")
    elif "TX" not in out.columns:
        out["TX"] = "NO_TX"

    # out["GENE_NAME"] = np.nan
    out["GENE_NAME"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    for c in ["SYMBOL", "Gene_Name", "Gene", "HGNC_Name"]:
        if c in out.columns:
            mask = out["GENE_NAME"].isna() | out["GENE_NAME"].astype(str).isin(EMPTY_STR_SET)
            out.loc[mask, "GENE_NAME"] = out.loc[mask, c]

    for c in ["Consequence", "IMPACT", "VARIANT_CLASS"]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def load_full_vcf_table(vcf_path: str) -> pd.DataFrame:
    if not vcf_path or not os.path.exists(vcf_path):
        return pd.DataFrame()
    ann_fields = parse_ann_fields_from_header(vcf_path)
    info_ids, _, _ = parse_vcf_header(vcf_path)
    df = load_vcf_records(vcf_path)
    df = split_info_column(df, info_ids)
    df = split_sample_format(df)
    df = build_alleles_from_gt(df)
    df = explode_alt_and_link_ann(df, ann_fields=ann_fields, ann_col="ANN")
    df = add_common_ann_normalization(df)
    df = df.drop(columns=[c for c in ["INFO", "FILTER", "ANN"] if c in df.columns])
    return df


# =========================================================
# Common normalization / filters
# =========================================================


# def add_unified_gene(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     if "GENE_NAME" not in out.columns:
#         out["GENE_NAME"] = np.nan
#     for c in ["SYMBOL", "Gene_Name", "Gene", "HGNC_Name", "EXOME_GENE_NAME"]:
#         if c in out.columns:
#             mask = out["GENE_NAME"].isna() | out["GENE_NAME"].astype(str).isin(EMPTY_STR_SET)
#             out.loc[mask, "GENE_NAME"] = out.loc[mask, c]
#     return out

def add_unified_gene(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "GENE_NAME" not in out.columns:
        out["GENE_NAME"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    else:
        out["GENE_NAME"] = out["GENE_NAME"].astype("object")

    for c in ["SYMBOL", "Gene_Name", "Gene", "HGNC_Name", "EXOME_GENE_NAME"]:
        if c in out.columns:
            mask = out["GENE_NAME"].isna() | out["GENE_NAME"].astype(str).isin(EMPTY_STR_SET)
            out.loc[mask, "GENE_NAME"] = out.loc[mask, c].astype("object")
    return out


def add_unified_dp_vaf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prefix = detect_sample_prefix(out)
    dp_col = f"{prefix}.DP" if prefix and f"{prefix}.DP" in out.columns else None
    baf_col = f"{prefix}.BAF" if prefix and f"{prefix}.BAF" in out.columns else None
    ad_col = f"{prefix}.AD" if prefix and f"{prefix}.AD" in out.columns else None

    if "DP" in out.columns:
        out["DP_UNI"] = pd.to_numeric(out["DP"], errors="coerce")
    elif dp_col:
        out["DP_UNI"] = pd.to_numeric(out[dp_col], errors="coerce")
    else:
        out["DP_UNI"] = np.nan

    if "VAF" in out.columns:
        out["VAF_UNI"] = pd.to_numeric(out["VAF"], errors="coerce")
    elif baf_col:
        out["VAF_UNI"] = pd.to_numeric(out[baf_col], errors="coerce")
    elif ad_col:
        def vaf_from_ad(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan
            parts = re.split(r"[,\s]+", str(x).strip())
            if len(parts) < 2:
                return np.nan
            try:
                ref = float(parts[0]); alt = float(parts[1]); tot = ref + alt
                return alt / tot if tot > 0 else np.nan
            except Exception:
                return np.nan
        out["VAF_UNI"] = out[ad_col].map(vaf_from_ad)
    else:
        out["VAF_UNI"] = np.nan
    if "allele_1" not in out.columns:
        out["allele_1"] = np.nan
    if "allele_2" not in out.columns:
        out["allele_2"] = np.nan
    return out


def parse_acmg_to_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in EMPTY_STR_SET:
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


def add_acmg_int(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    acmg_col = None
    for c in ["Prediction_ACMG_tapes", "EXOME_Prediction_ACMG_tapes", "Prediction_ACMG", "EXOME_Prediction_ACMG",
              "Classification_ACMG", "EXOME_Classification_ACMG"]:
        if c in out.columns:
            acmg_col = c
            break
    out["ACMG_RAW"] = out[acmg_col] if acmg_col else np.nan
    out["ACMG_INT"] = pd.to_numeric(out["ACMG_RAW"].map(parse_acmg_to_int), errors="coerce")
    return out


def compute_max_pop_af(df: pd.DataFrame) -> pd.Series:
    # af_cols = [
    #     "MAX_AF", "gnomAD_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF", "gnomAD_EAS_AF",
    #     "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF", "AF", "AFR_AF", "AMR_AF",
    #     "EAS_AF", "EUR_AF", "SAS_AF", "AA_AF", "EA_AF", "IG_AF", "Kaviar_AF", "EVS_MAF", "EVS_CA",
    #     "GnomAD_Genome_AF", "GnomAD_Genome_AF_afr", "GnomAD_Genome_AF_amr", "GnomAD_Genome_AF_asj",
    #     "GnomAD_Genome_AF_eas", "GnomAD_Genome_AF_fin", "GnomAD_Genome_AF_nfe", "GnomAD_Genome_AF_oth",
    #     "GnomAD_Genome_AF_sas", "GnomAD_Genome_AF_popmax", "EXOME_MAX_AF"
    # ]

    af_cols = [
        "gnomAD_AF",              # gnomAD exome (global)
        "GnomAD_Genome_AF",       # gnomAD genome (global)
        "Kaviar_AF",              # Kaviar global AF
        "EVS_MAF",                # EVS global MAF
        "AF",                     # VEP / 1000G aggregated AF
        "MAX_AF",                 # already a global max across pops
        "EXOME_MAX_AF"            # exome-based max (if present)
    ]
    present = [c for c in af_cols if c in df.columns]
    if not present:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[present].apply(lambda x: pd.to_numeric(x, errors="coerce")).max(axis=1, skipna=True)


def add_unified_clinsig(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CLNSIG_UNI"] = np.nan
    for c in ["clinvar_CLNSIG", "ClinVar_CLNSIG", "CLIN_SIG"]:
        if c in out.columns:
            out = _coalesce_columns(out, "CLNSIG_UNI", [c])
    return out


def add_pathway_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidates = ["KEGG_Pathway", "GO_biological_process", "Function_description", "Disease_description", "DisGeNET_disease"]
    present = [c for c in candidates if c in out.columns]
    if not present:
        out["PATH_DNA_REPAIR"] = 0
        out["PATHWAY_HIT"] = 0
        return out
    text = out[present[0]].astype(str).fillna("").str.lower()
    for c in present[1:]:
        text = text + " | " + out[c].astype(str).fillna("").str.lower()
    dna_repair_re = re.compile(r"dna repair|dna damage|double[- ]strand break|homologous recombination|mismatch repair|fanconi", re.I)
    cancer_path_re = re.compile(r"p53|pi3k|akt|mtor|mapk|ras|wnt|notch|cell cycle|apoptosis|dna repair|fanconi", re.I)
    out["PATH_DNA_REPAIR"] = text.str.contains(dna_repair_re, na=False).astype(int)
    out["PATHWAY_HIT"] = text.str.contains(cancer_path_re, na=False).astype(int)
    return out


def prep_base_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out["REF"] = out["REF"].astype(str)
    out["ALT"] = out["ALT"].astype(str)
    if "TX" not in out.columns:
        out["TX"] = "NO_TX"
    out = add_unified_gene(out)
    out = add_unified_dp_vaf(out)
    out = add_acmg_int(out)
    out = add_unified_clinsig(out)
    out["MAX_POP_AF"] = compute_max_pop_af(out)
    out = add_pathway_flags(out)
    return out


# =========================================================
# Merge VCF + EXOME (source level)
# =========================================================


def merge_vcf_with_exome(vcf_path: str, exome_path: str, source_label: str) -> pd.DataFrame:
    if not vcf_path or not os.path.exists(vcf_path):
        return pd.DataFrame()
    df_vcf = load_full_vcf_table(vcf_path)
    if df_vcf.empty:
        return df_vcf
    df_vcf["TX"] = df_vcf["Feature"]

    if exome_path and os.path.exists(exome_path):
        df_exome = load_exome_table(exome_path)
        df_exome = add_tx_from_exome(df_exome)
    else:
        df_exome = pd.DataFrame(columns=KEY_COLS)

    for c in ["CHROM", "POS", "REF", "ALT", "TX"]:
        if c not in df_exome.columns:
            df_exome[c] = np.nan if c == "POS" else ("NO_TX" if c == "TX" else "")
    df_exome["CHROM"] = df_exome["CHROM"].astype(str).map(normalize_chr)
    df_exome["POS"] = pd.to_numeric(df_exome["POS"], errors="coerce")
    df_exome["REF"] = df_exome["REF"].astype(str)
    df_exome["ALT"] = df_exome["ALT"].astype(str)
    df_exome["TX"] = df_exome["TX"].astype(str).replace({"nan": "NO_TX", "None": "NO_TX"}).fillna("NO_TX")

    # overlap_nonkey = [c for c in df_vcf.columns if c in df_exome.columns and c not in KEY_COLS]
    # if overlap_nonkey:
    #     df_exome = df_exome.rename(columns={c: f"EXOME_{c}" for c in overlap_nonkey})
    # out = df_vcf.merge(df_exome, on=KEY_COLS, how="outer")
    exome_keep = [c for c in df_exome.columns if c in KEY_COLS or c not in df_vcf.columns]
    df_exome_small = df_exome[exome_keep].copy()
    out = df_vcf.merge(df_exome_small, on=KEY_COLS, how="outer")
    # out = _coalesce_columns(out, "Consequence", ["EXOME_Consequence"])
    # out = _coalesce_columns(out, "IMPACT", ["EXOME_IMPACT"])
    # out = _coalesce_columns(out, "VARIANT_CLASS", ["EXOME_VARIANT_CLASS"])

    # out = _coalesce_columns(out, "FILTER", ["EXOME_FILTER"])
    out = prep_base_table(out)
    out["SOURCE_LABEL"] = source_label
    return sort_by_variant(out)


# =========================================================
# CNV / HET enrichment
# =========================================================


def load_cnv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if os.path.isdir(path):
        hits = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".txt")]
        path = hits[0] if hits else ""
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", dtype=str)
    for c in ["SV_chrom", "CHROM", "chr", "CHR"]:
        if c in df.columns:
            df[c] = df[c].map(normalize_chr)
    if "SV_chrom" not in df.columns:
        for c in ["CHROM", "chr", "CHR"]:
            if c in df.columns:
                df = df.rename(columns={c: "SV_chrom"})
                break
    for c in ["SV_start", "SV_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def match_cnv_to_variants(variants: pd.DataFrame, cnv: pd.DataFrame) -> pd.DataFrame:
    out = variants.copy()
    for c in ["CNV_HIT", "CNV_pathogenic", "LOH_PARTIAL_PATHO"]:
        out[c] = 0
    if cnv is None or cnv.empty or not {"SV_chrom", "SV_start", "SV_end"}.issubset(cnv.columns):
        return out
    c = cnv.copy()
    c["SV_chrom"] = c["SV_chrom"].astype(str).map(normalize_chr)
    c["SV_start"] = pd.to_numeric(c["SV_start"], errors="coerce")
    c["SV_end"] = pd.to_numeric(c["SV_end"], errors="coerce")
    c = c.dropna(subset=["SV_chrom", "SV_start", "SV_end"])
    if c.empty:
        return out
    c["_START"] = c[["SV_start", "SV_end"]].min(axis=1)
    c["_END"] = c[["SV_start", "SV_end"]].max(axis=1)
    call = c["CALL"].astype(str).str.lower() if "CALL" in c.columns else pd.Series([""] * len(c), index=c.index)
    svt = c["SV_type"].astype(str).str.lower() if "SV_type" in c.columns else pd.Series([""] * len(c), index=c.index)
    c["_PATHO"] = (call.str.contains(r"\bloh\b|loss|deletion|del", regex=True, na=False) |
                    svt.str.contains(r"\bloh\b|loss|deletion|del", regex=True, na=False)).astype(int)
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    for chrom, idx in out.groupby("CHROM").groups.items():
        pos = out.loc[idx, "POS"].to_numpy()
        cc = c.loc[c["SV_chrom"] == chrom, ["_START", "_END", "_PATHO"]]
        if cc.empty:
            continue
        starts, ends, patho = cc["_START"].to_numpy(), cc["_END"].to_numpy(), cc["_PATHO"].to_numpy()
        order = np.argsort(starts)
        starts, ends, patho = starts[order], ends[order], patho[order]
        j = np.searchsorted(starts, pos, side="right") - 1
        j2 = np.clip(j, 0, len(ends) - 1)
        overlap = (j >= 0) & (pos <= ends[j2])
        out.loc[idx, "CNV_HIT"] = overlap.astype(int)
        loh = overlap & (patho[j2] == 1)
        out.loc[idx, "LOH_PARTIAL_PATHO"] = loh.astype(int)
        out.loc[idx, "CNV_pathogenic"] = loh.astype(int)
    return out


def load_hom_plink(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if os.path.isdir(path):
        hits = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".hom", ".txt"))]
        path = hits[0] if hits else ""
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=r"\s+", engine="python", dtype=str)
    if "CHR" in df.columns:
        df["CHR"] = df["CHR"].map(normalize_chr)
    for c in ["POS1", "POS2", "PHOM", "PHET"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def annotate_variants_with_hom(variants: pd.DataFrame, hom: pd.DataFrame, phom_threshold: float = 0.5) -> pd.DataFrame:
    out = variants.copy()
    out["IN_ROH"] = 0
    out["HET_HIGH_HOM"] = 0
    if hom is None or hom.empty or not {"CHR", "POS1", "POS2"}.issubset(hom.columns):
        return out
    h = hom.copy()
    h["CHR"] = h["CHR"].astype(str).map(normalize_chr)
    h["POS1"] = pd.to_numeric(h["POS1"], errors="coerce")
    h["POS2"] = pd.to_numeric(h["POS2"], errors="coerce")
    h = h.dropna(subset=["CHR", "POS1", "POS2"])
    if h.empty:
        return out
    h["_START"] = h[["POS1", "POS2"]].min(axis=1)
    h["_END"] = h[["POS1", "POS2"]].max(axis=1)
    h["_HIGH"] = (pd.to_numeric(h.get("PHOM", 0), errors="coerce").fillna(0) >= float(phom_threshold)).astype(int)
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    for chrom, idx in out.groupby("CHROM").groups.items():
        pos = out.loc[idx, "POS"].to_numpy()
        hc = h.loc[h["CHR"] == chrom, ["_START", "_END", "_HIGH"]]
        if hc.empty:
            continue
        starts, ends, high = hc["_START"].to_numpy(), hc["_END"].to_numpy(), hc["_HIGH"].to_numpy()
        order = np.argsort(starts)
        starts, ends, high = starts[order], ends[order], high[order]
        j = np.searchsorted(starts, pos, side="right") - 1
        j2 = np.clip(j, 0, len(ends) - 1)
        overlap = (j >= 0) & (pos <= ends[j2])
        out.loc[idx, "IN_ROH"] = overlap.astype(int)
        out.loc[idx, "HET_HIGH_HOM"] = (overlap & (high[j2] == 1)).astype(int)
    return out


def enrich_with_cnv_het(df: pd.DataFrame, cnv_path: Optional[str], het_path: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    out = match_cnv_to_variants(out, load_cnv(cnv_path)) if cnv_path else out.assign(CNV_HIT=0, CNV_pathogenic=0, LOH_PARTIAL_PATHO=0)
    out = annotate_variants_with_hom(out, load_hom_plink(het_path)) if het_path else out.assign(IN_ROH=0, HET_HIGH_HOM=0)
    for c in ["CNV_HIT", "CNV_pathogenic", "LOH_PARTIAL_PATHO", "IN_ROH", "HET_HIGH_HOM"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


# =========================================================
# Source builders
# =========================================================


def build_source_table(vcf_path: str, exome_path: str, cnv_path: Optional[str], het_path: Optional[str],
                       family_id: str, patient_id: str, origin: str, source_label: str) -> pd.DataFrame:
    df = merge_vcf_with_exome(vcf_path, exome_path, source_label=source_label)
    if df.empty:
        return df
    df = enrich_with_cnv_het(df, cnv_path=cnv_path, het_path=het_path)
    df = prep_base_table(df)
    df["FAMILY_ID"] = str(family_id)
    df["PATIENT_ID"] = str(patient_id)
    df["ORIGIN"] = origin
    df["SOURCE_LABEL"] = source_label
    return sort_by_variant(df)


def vcf_to_exome_path(vcf_path: str) -> Optional[str]:
    if not vcf_path or not str(vcf_path).endswith((".vcf", ".vcf.gz")):
        return None
    folder = os.path.dirname(vcf_path)
    name = os.path.basename(vcf_path).replace(".vcf.gz", "").replace(".vcf", "")
    name = name.replace("_Genetic_Variants", "_Exome_Genetic_Variants")
    p = os.path.join(folder, name + ".txt")
    return p if os.path.exists(p) else None


def build_germline_table(family_id: str, patient_id: str, germ_cfg: dict) -> pd.DataFrame:
    vcf_path = germ_cfg.get("vcf")
    exome_path = vcf_to_exome_path(vcf_path) or pick_single_txt_from_dir(os.path.dirname(vcf_path) if vcf_path else None)
    return build_source_table(
        vcf_path=vcf_path,
        exome_path=exome_path,
        cnv_path=germ_cfg.get("cnv"),
        het_path=germ_cfg.get("het"),
        family_id=family_id,
        patient_id=patient_id,
        origin="germline",
        source_label="germline",
    )


def build_somatic_source_from_dirs(family_id: str, patient_id: str, run_cfg: dict, which: str) -> pd.DataFrame:
    if which not in {"genetic", "oncology"}:
        raise ValueError("which must be 'genetic' or 'oncology'")
    vcf_dir = run_cfg.get("vcf_genetic_dir" if which == "genetic" else "vcf_oncology_dir")
    cnv_dir = run_cfg.get("cnv_genetic_dir" if which == "genetic" else "cnv_oncology_dir")
    het_dir = run_cfg.get("het_dir")

    vcf_path = pick_single_vcf_from_dir(vcf_dir)
    exome_path = pick_single_txt_from_dir(vcf_dir)
    cnv_path = pick_single_txt_from_dir(cnv_dir)
    het_path = pick_single_txt_from_dir(het_dir) or list_files_one(het_dir, (".hom", ".txt")) if het_dir and os.path.isdir(het_dir) else het_dir

    return build_source_table(
        vcf_path=vcf_path,
        exome_path=exome_path,
        cnv_path=cnv_path,
        het_path=het_path,
        family_id=family_id,
        patient_id=patient_id,
        origin="somatic",
        source_label=which,
    )


# =========================================================
# Consolidation without duplicate-column forests
# =========================================================


def prefix_nonkey_columns(df: pd.DataFrame, prefix: str, keep_cols: Optional[set] = None) -> pd.DataFrame:
    keep_cols = keep_cols or set()
    rename = {}
    for c in df.columns:
        if c in set(PAT_KEY_COLS) | keep_cols:
            continue
        rename[c] = f"{prefix}{c}"
    return df.rename(columns=rename)


# def merge_parallel_blocks(left: pd.DataFrame, right: pd.DataFrame, left_prefix: str, right_prefix: str,
#                           overlap_flag_name: str, source_set_name: str) -> pd.DataFrame:
#     """
#     Merge two tables on the full patient+variant+transcript key while:
#     - preserving one row per shared key
#     - prefixing source-specific payload columns
#     - creating consolidated shared normalized columns
#     """
#     if left is None or left.empty:
#         out = right.copy() if right is not None else pd.DataFrame()
#         if out is not None and not out.empty:
#             out[overlap_flag_name] = 0
#             out[source_set_name] = right_prefix.rstrip("_")
#         return out
#     if right is None or right.empty:
#         out = left.copy()
#         out[overlap_flag_name] = 0
#         out[source_set_name] = left_prefix.rstrip("_")
#         return out

#     # keep_extra = {"ORIGIN", "SOURCE_LABEL"}
#     # l = prefix_nonkey_columns(left, left_prefix, keep_cols=keep_extra)
#     # r = prefix_nonkey_columns(right, right_prefix, keep_cols=keep_extra)
#     l = prefix_nonkey_columns(left, left_prefix, keep_cols=set())
#     r = prefix_nonkey_columns(right, right_prefix, keep_cols=set())

#     merged = l.merge(r, on=PAT_KEY_COLS, how="outer", suffixes=("", ""), indicator=True)
#     merged[overlap_flag_name] = (merged["_merge"] == "both").astype(int)
#     merged[source_set_name] = np.select(
#         [merged["_merge"].eq("both"), merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only")],
#         [f"{left_prefix.rstrip('_')}|{right_prefix.rstrip('_')}", left_prefix.rstrip("_"), right_prefix.rstrip("_")],
#         default="",
#     )

#     merged[f"HAS_{left_prefix.rstrip('_').upper()}"] = merged["_merge"].isin(["left_only", "both"]).astype(int)
#     merged[f"HAS_{right_prefix.rstrip('_').upper()}"] = merged["_merge"].isin(["right_only", "both"]).astype(int)

#     for col in SHARED_NORMALIZED_COLS:
#         lc = f"{left_prefix}{col}"
#         rc = f"{right_prefix}{col}"
#         if lc in merged.columns or rc in merged.columns:
#             merged[col] = np.nan
#             if lc in merged.columns:
#                 merged[col] = merged[lc]
#             if rc in merged.columns:
#                 mask = merged[col].isna() | merged[col].astype(str).isin(EMPTY_STR_SET)
#                 merged.loc[mask, col] = merged.loc[mask, rc]

#     merged["ORIGIN"] = np.select(
#         [merged["_merge"].eq("both"), merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only")],
#         ["both", left_prefix.rstrip("_"), right_prefix.rstrip("_")],
#         default="unknown",
#     )
#     return merged.drop(columns=["_merge"])
COALESCE_COLS = {
    "GENE_NAME", "Consequence", "IMPACT", "VARIANT_CLASS",
    "CLNSIG_UNI", "allele_1", "allele_2"
}

MAX_COLS = {
    "MAX_POP_AF",
    "ACMG_INT",
    "PATHWAY_HIT",
    "PATH_DNA_REPAIR",
    "CNV_HIT",
    "CNV_pathogenic",
    "LOH_PARTIAL_PATHO",
    "IN_ROH",
    "HET_HIGH_HOM",
}

def merge_parallel_blocks(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_prefix: str,
    right_prefix: str,
    overlap_flag_name: str,
    source_set_name: str
) -> pd.DataFrame:


    if left is None or left.empty:
        out = right.copy() if right is not None else pd.DataFrame()
        if not out.empty:
            out[overlap_flag_name] = 0
            out[source_set_name] = "right"
        return out

    if right is None or right.empty:
        out = left.copy()
        out[overlap_flag_name] = 0
        out[source_set_name] = "left"
        return out

    # ------------------------------------------------------------
    # merge on key
    # ------------------------------------------------------------
    merged = left.merge(
        right,
        on=PAT_KEY_COLS,
        how="outer",
        suffixes=("_L", "_R"),
        indicator=True
    )

    # ------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------
    merged[overlap_flag_name] = (merged["_merge"] == "both").astype(int)

    merged[source_set_name] = np.select(
        [
            merged["_merge"].eq("both"),
            merged["_merge"].eq("left_only"),
            merged["_merge"].eq("right_only"),
        ],
        ["both", "left", "right"],
        default="",
    )

    # ------------------------------------------------------------
    # columns using MAX
    # ------------------------------------------------------------
    MAX_COLS = {
        "MAX_POP_AF",
        "ACMG_INT",
        "PATHWAY_HIT",
        "PATH_DNA_REPAIR",
        "CNV_HIT",
        "CNV_pathogenic",
        "LOH_PARTIAL_PATHO",
        "IN_ROH",
        "HET_HIGH_HOM",
    }

    # ------------------------------------------------------------
    # reconstruct final columns
    # ------------------------------------------------------------
    final_cols = set()

    for col in set(left.columns).union(set(right.columns)):
        if col in PAT_KEY_COLS:
            continue

        col_L = f"{col}_L"
        col_R = f"{col}_R"

        has_L = col_L in merged.columns
        has_R = col_R in merged.columns

        # column exists in both
        if has_L and has_R:

            if col in MAX_COLS:
                merged[col] = pd.concat(
                    [
                        pd.to_numeric(merged[col_L], errors="coerce"),
                        pd.to_numeric(merged[col_R], errors="coerce"),
                    ],
                    axis=1
                ).max(axis=1, skipna=True).fillna(0)

            else:
                merged[col] = merged[col_L]

        # only left
        elif has_L:
            merged[col] = merged[col_L]

        # only right
        elif has_R:
            merged[col] = merged[col_R]

        final_cols.add(col)

    # ------------------------------------------------------------
    # clean output
    # ------------------------------------------------------------
    keep_cols = list(PAT_KEY_COLS) + list(final_cols) + [overlap_flag_name, source_set_name]

    merged = merged[keep_cols]

    return merged

def consolidate_somatic_run(family_id: str, patient_id: str, run_cfg: dict, run_index: int) -> pd.DataFrame:
    dg = build_somatic_source_from_dirs(family_id, patient_id, run_cfg, which="genetic")
    do = build_somatic_source_from_dirs(family_id, patient_id, run_cfg, which="oncology")
    out = merge_parallel_blocks(dg, do, left_prefix="GEN_", right_prefix="ONC_",
                                overlap_flag_name="SAME_VARIANT_IN_GENETIC_AND_ONCOLOGY",
                                source_set_name="SOMATIC_SOURCE_SET")
    if out is not None and not out.empty:
        out["SOMATIC_RUN_ID"] = f"run_{run_index}"
    return sort_by_variant(out)


def consolidate_all_somatic_runs(family_id: str, patient_id: str, cfg: dict) -> pd.DataFrame:
    runs = []
    for i, run_cfg in enumerate(cfg.get("somatic_runs", []) or [], start=1):
        dr = consolidate_somatic_run(family_id, patient_id, run_cfg, run_index=i)
        if dr is not None and not dr.empty:
            runs.append(dr)
    if not runs:
        return pd.DataFrame()
    out = runs[0]
    for i, nxt in enumerate(runs[1:], start=2):
        out = merge_parallel_blocks(out, nxt, left_prefix="SOMPREV_", right_prefix=f"RUN{i}_",
                                    overlap_flag_name="SAME_VARIANT_ACROSS_SOMATIC_RUNS",
                                    source_set_name="SOMATIC_RUN_SET")
    out["HAS_SOM"] = 1
    return sort_by_variant(out)


def build_patient_long(family_id: str, patient_id: str, cfg: dict) -> pd.DataFrame:
    germ = build_germline_table(family_id, patient_id, cfg.get("germline_files", {}))
    som = consolidate_all_somatic_runs(family_id, patient_id, cfg)
    out = merge_parallel_blocks(germ, som, left_prefix="GERM_", right_prefix="SOM_",
                                overlap_flag_name="SAME_VARIANT_IN_GERMLINE_AND_SOMATIC",
                                source_set_name="ORIGIN_SOURCE_SET")
    if out is not None and not out.empty:
        out["HAS_GERM"] = out.get("HAS_GERM", 0)
        out["HAS_SOM"] = out.get("HAS_SOM", 0)
    return sort_by_variant(out)


# =========================================================
# Family-level filters / summaries
# =========================================================


def consequence_pass_nonsyn(df: pd.DataFrame) -> pd.Series:
    cons = df["Consequence"].astype(str).str.lower() if "Consequence" in df.columns else pd.Series([""] * len(df), index=df.index)
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


def add_patient_level_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-row/patient-level filters on the patient LONG table.
    These are the building blocks used later for family-level filtering.
    """
    out = df.copy()

    out["MAX_POP_AF"] = pd.to_numeric(out.get("MAX_POP_AF", np.nan), errors="coerce")
    # out["SIRIUS_PASS_POPAF"] = ((out["MAX_POP_AF"].isna()) | (out["MAX_POP_AF"] < 0.05)).astype(int)
    out["SIRIUS_PASS_POPAF"] = ((out["MAX_POP_AF"] < 0.05)).astype(int)
    out["SIRIUS_PASS_NONSYN"] = consequence_pass_nonsyn(out)
    out["ACMG_INT"] = pd.to_numeric(out.get("ACMG_INT", np.nan), errors="coerce")
    out["SIRIUS_PASS_ACMG_SNV"] = out["ACMG_INT"].isin([3, 4, 5]).astype(int)
    out["SIRIUS_PASS_ACMG_CNV"] = out["ACMG_INT"].isin([4, 5]).astype(int)

    out["DP_UNI"] = pd.to_numeric(out.get("DP_UNI", np.nan), errors="coerce")
    out["VAF_UNI"] = pd.to_numeric(out.get("VAF_UNI", np.nan), errors="coerce")
    out["LOH_PARTIAL_PATHO"] = pd.to_numeric(out.get("LOH_PARTIAL_PATHO", 0), errors="coerce").fillna(0).astype(int)

    origin = out.get("ORIGIN", pd.Series([""] * len(out), index=out.index)).astype(str).str.lower()
    # is_germ = origin.eq("germline") | out.get("HAS_GERM", pd.Series([0] * len(out), index=out.index)).astype(int).eq(1)
    # is_som = origin.eq("somatic") | out.get("HAS_SOM", pd.Series([0] * len(out), index=out.index)).astype(int).eq(1)
    out["HAS_GERM"] = pd.to_numeric(
        out.get("HAS_GERM", pd.Series(0, index=out.index)),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    out["HAS_SOM"] = pd.to_numeric(
        out.get("HAS_SOM", pd.Series(0, index=out.index)),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    origin = out.get("ORIGIN", pd.Series([""] * len(out), index=out.index)).astype(str).str.lower()
    is_germ = origin.eq("germline") | out["HAS_GERM"].eq(1)
    is_som = origin.eq("somatic") | out["HAS_SOM"].eq(1)

    base_ok = (
        (pd.to_numeric(out["SIRIUS_PASS_POPAF"], errors="coerce").fillna(0).astype(int) == 1) &
        (pd.to_numeric(out["SIRIUS_PASS_NONSYN"], errors="coerce").fillna(0).astype(int) == 1) &
        (pd.to_numeric(out["SIRIUS_PASS_ACMG_SNV"], errors="coerce").fillna(0).astype(int) == 1)
    )

    out["SIRIUS_PASS_SNV_GERMLINE"] = (
        is_germ & base_ok & out["DP_UNI"].gt(10) & out["VAF_UNI"].gt(0.25)
    ).astype(int)

    out["SIRIUS_PASS_SNV_SOMATIC"] = (
        is_som & base_ok & out["DP_UNI"].gt(20) & out["VAF_UNI"].gt(0.05)
    ).astype(int)

    out["SIRIUS_PASS_CNV"] = (
        (out["LOH_PARTIAL_PATHO"] == 1) &
        (pd.to_numeric(out["SIRIUS_PASS_ACMG_CNV"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)

    out["SIRIUS_FLAG_ROW"] = (
        (out["SIRIUS_PASS_SNV_GERMLINE"] == 1) |
        (out["SIRIUS_PASS_SNV_SOMATIC"] == 1) |
        (out["SIRIUS_PASS_CNV"] == 1)
    ).astype(int)

    return out


def infer_patient_cancer_status(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic patient cancer status used for family filters.
    If somatic data exist for the patient, mark as cancer-like carrier.
    """
    if "HAS_SOM" in df.columns:
        return pd.to_numeric(df["HAS_SOM"], errors="coerce").fillna(0).astype(int)
    if "ORIGIN" in df.columns:
        return df["ORIGIN"].astype(str).str.lower().eq("somatic").astype(int)
    return pd.Series([0] * len(df), index=df.index)

def uniq_join_tokens(s: pd.Series, seps: str = r"[|,]") -> str:
    vals = []
    for x in s.dropna():
        sx = str(x).strip()
        if sx in EMPTY_STR_SET:
            continue
        parts = [p.strip() for p in re.split(seps, sx) if p.strip() and p.strip() not in EMPTY_STR_SET]
        vals.extend(parts)
    vals = sorted(set(vals))
    return ",".join(vals)

def collapse_patient_long_to_family_variants(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse transcript-aware patient LONG rows to family-level variant/transcript rows,
    while preserving ALL columns from LONG and adding family-level summary columns.
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    df = add_patient_level_filters(df_long.copy())
    df["PATIENT_CANCER_STATUS"] = infer_patient_cancer_status(df)

    key = FAM_KEY_COLS
    pat_key = PAT_KEY_COLS

    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _collapse_numeric(s: pd.Series):
        x = _to_num(s).dropna()
        if x.empty:
            return np.nan
        return x.max()

    def _collapse_text(s: pd.Series):
        vals = [str(v).strip() for v in s.dropna() if str(v).strip() not in EMPTY_STR_SET]
        if not vals:
            return ""
        return "|".join(sorted(set(vals)))

    def _collapse_any(s: pd.Series):
        if s is None or len(s) == 0:
            return np.nan

        x = _to_num(s)
        n_num = x.notna().sum()
        n_tot = s.notna().sum()

        if n_tot > 0 and (n_num / n_tot) >= 0.8:
            return _collapse_numeric(s)
        return _collapse_text(s)

    # ------------------------------------------------------------
    # patient-level collapse first
    # ------------------------------------------------------------
    per_patient = (
        df.groupby(pat_key, dropna=False)
          .agg(
              PATIENT_CANCER_STATUS=("PATIENT_CANCER_STATUS", lambda x: _to_num(x).fillna(0).astype(int).max()),
              PAT_SIRIUS_FLAG=("SIRIUS_FLAG_ROW", lambda x: _to_num(x).fillna(0).astype(int).max()),
              PAT_SIRIUS_GERM_SNV_PASS=("SIRIUS_PASS_SNV_GERMLINE", lambda x: _to_num(x).fillna(0).astype(int).max()),
              PAT_SIRIUS_SOM_SNV_PASS=("SIRIUS_PASS_SNV_SOMATIC", lambda x: _to_num(x).fillna(0).astype(int).max()),
              PAT_SIRIUS_CNV_PASS=("SIRIUS_PASS_CNV", lambda x: _to_num(x).fillna(0).astype(int).max()),
              HAS_GERM=("HAS_GERM", lambda x: _to_num(x).fillna(0).astype(int).max()) if "HAS_GERM" in df.columns else ("PATIENT_ID", lambda x: 0),
              HAS_SOM=("HAS_SOM", lambda x: _to_num(x).fillna(0).astype(int).max()) if "HAS_SOM" in df.columns else ("PATIENT_ID", lambda x: 0),
          )
          .reset_index()
    )

    # ------------------------------------------------------------
    # family summary block
    # ------------------------------------------------------------
    def fam_block(g: pd.DataFrame) -> pd.Series:
        n_pat = int(g["PATIENT_ID"].nunique())
        n_cancer = int(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"].nunique())
        n_non = int(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"].nunique())
        n_germ_only = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 0)).sum())
        n_germ_and_som = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 1)).sum())

        return pd.Series({
            "N_PATIENTS_WITH_VARIANT": n_pat,
            "PATIENT_ID_LIST": uniq_join_tokens(g["PATIENT_ID"]),
            "PATIENT_ID_LIST_CANCER": uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"]),
            "PATIENT_ID_LIST_NONCANCER": uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"]),
            "N_CANCER_WITH_VARIANT": n_cancer,
            "N_NONCANCER_WITH_VARIANT": n_non,
            "N_GERM_ONLY": n_germ_only,
            "N_GERM_AND_SOM": n_germ_and_som,
            "FAM_SIRIUS_FLAG": int(_to_num(g["PAT_SIRIUS_FLAG"]).fillna(0).astype(int).max()),
            "FAM_SIRIUS_GERM_SNV_PASS": int(_to_num(g["PAT_SIRIUS_GERM_SNV_PASS"]).fillna(0).astype(int).max()),
            "FAM_SIRIUS_SOM_SNV_PASS": int(_to_num(g["PAT_SIRIUS_SOM_SNV_PASS"]).fillna(0).astype(int).max()),
            "FAM_SIRIUS_CNV_PASS": int(_to_num(g["PAT_SIRIUS_CNV_PASS"]).fillna(0).astype(int).max()),
        })

    fam_counts = (
        per_patient.groupby(key, dropna=False)
        .apply(fam_block)
        .reset_index()
    )

    # ------------------------------------------------------------
    # collapse ALL LONG columns exactly once
    # ------------------------------------------------------------
    long_payload_cols = [c for c in df.columns if c not in key]

    fam_payload = (
        df.groupby(key, dropna=False)[long_payload_cols]
          .agg(lambda s: _collapse_any(s))
          .reset_index()
    )

    # ------------------------------------------------------------
    # merge family summaries on top of collapsed payload
    # ------------------------------------------------------------
    fam = fam_payload.merge(fam_counts, on=key, how="left")

    fam["FAMILY_ID_LIST"] = fam["FAMILY_ID"].astype(str)

    # ------------------------------------------------------------
    # labels and scores
    # ------------------------------------------------------------
    fam["label_of_interest"] = (
        (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1) &
        (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0) &
        (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) == 1)
    ).astype(int)

    fam["FAM_COUNT_GE_2"] = (
        (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) == 1) &
        (_to_num(fam["N_PATIENTS_WITH_VARIANT"]).fillna(0).astype(int) > 1)
    ).astype(int)

    fam["SEGREGATION_SCORE"] = 0
    fam.loc[_to_num(fam["N_PATIENTS_WITH_VARIANT"]).fillna(0).astype(int) == 1, "SEGREGATION_SCORE"] = 1
    fam.loc[_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1, "SEGREGATION_SCORE"] = 2
    fam.loc[
        (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) == 1) &
        (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0),
        "SEGREGATION_SCORE"
    ] = 3
    fam.loc[
        (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 2) &
        (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0),
        "SEGREGATION_SCORE"
    ] = 4

    fam["FAMILY_EVIDENCE_SCORE"] = (
        (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1).astype(int) +
        (_to_num(fam["N_GERM_AND_SOM"]).fillna(0).astype(int) >= 1).astype(int) +
        (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) >= 1).astype(int) +
        (_to_num(fam.get("PATH_DNA_REPAIR", 0)).fillna(0).astype(int) >= 1).astype(int)
    ).clip(0, 4)

    fam["INTEGRATED_SCORE"] = (
        _to_num(fam["FAMILY_EVIDENCE_SCORE"]).fillna(0).astype(int) +
        _to_num(fam["SEGREGATION_SCORE"]).fillna(0).astype(int) +
        (_to_num(fam.get("PATHWAY_HIT", 0)).fillna(0).astype(int) >= 1).astype(int) +
        (_to_num(fam.get("PATH_DNA_REPAIR", 0)).fillna(0).astype(int) >= 1).astype(int)
    ).clip(0, 12)

    fam["INTEGRATED_CLASS"] = "LOW"
    fam.loc[fam["INTEGRATED_SCORE"] >= 4, "INTEGRATED_CLASS"] = "MODERATE"
    fam.loc[fam["INTEGRATED_SCORE"] >= 7, "INTEGRATED_CLASS"] = "HIGH"
    fam.loc[fam["INTEGRATED_SCORE"] >= 9, "INTEGRATED_CLASS"] = "TOP"

    reasons = []
    for _, row in fam.iterrows():
        r = []

        if int(pd.to_numeric(pd.Series([row.get("FAM_SIRIUS_GERM_SNV_PASS", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("GERM_SNV_PASS")
        if int(pd.to_numeric(pd.Series([row.get("FAM_SIRIUS_SOM_SNV_PASS", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("SOM_SNV_PASS")
        if int(pd.to_numeric(pd.Series([row.get("FAM_SIRIUS_CNV_PASS", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("CNV_PASS")
        if int(pd.to_numeric(pd.Series([row.get("N_GERM_AND_SOM", 0)]), errors="coerce").fillna(0).iloc[0]) >= 1:
            r.append("GERM_AND_SOM")
        if int(pd.to_numeric(pd.Series([row.get("FAM_COUNT_GE_2", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("COUNT_GE_2")

        seg_score = int(pd.to_numeric(pd.Series([row.get("SEGREGATION_SCORE", 0)]), errors="coerce").fillna(0).iloc[0])
        if seg_score > 0:
            r.append(f"SEG+{seg_score}")

        fam_ev = int(pd.to_numeric(pd.Series([row.get("FAMILY_EVIDENCE_SCORE", 0)]), errors="coerce").fillna(0).iloc[0])
        if fam_ev > 0:
            r.append(f"FAM_EVD+{fam_ev}")

        if int(pd.to_numeric(pd.Series([row.get("PATHWAY_HIT", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("PATHWAY_HIT")
        if int(pd.to_numeric(pd.Series([row.get("PATH_DNA_REPAIR", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            r.append("DNA_REPAIR")

        reasons.append("|".join(r))

    fam["INTEGRATED_EVIDENCE"] = reasons

    return sort_by_variant(fam)

# def collapse_patient_long_to_family_variants(df_long: pd.DataFrame) -> pd.DataFrame:
#     """
#     Collapse transcript-aware patient LONG rows to family-level variant/transcript rows,
#     adding family filters and labels, while preserving ALL columns in collapsed form.
#     """
#     if df_long is None or df_long.empty:
#         return pd.DataFrame()

#     df = add_patient_level_filters(df_long.copy())
#     df["PATIENT_CANCER_STATUS"] = infer_patient_cancer_status(df)

#     key = FAM_KEY_COLS
#     pat_key = PAT_KEY_COLS

#     # ------------------------------------------------------------------
#     # helpers
#     # ------------------------------------------------------------------
#     def _to_num(s):
#         return pd.to_numeric(s, errors="coerce")

#     def _is_numeric_series(s: pd.Series) -> bool:
#         return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)

#     def _collapse_numeric(s: pd.Series):
#         x = _to_num(s)
#         x = x.dropna()
#         if x.empty:
#             return np.nan
#         # if values are integer-like / binary-like, max is often the safest
#         return x.max()

#     def _collapse_text(s: pd.Series):
#         vals = [str(v).strip() for v in s.dropna() if str(v).strip() != ""]
#         if not vals:
#             return ""
#         return uniq_join(vals)

#     def _collapse_mixed(s: pd.Series):
#         # Try numeric first if most values are numeric-like
#         x = _to_num(s)
#         n_num = x.notna().sum()
#         n_tot = s.notna().sum()
#         if n_tot > 0 and n_num / max(n_tot, 1) >= 0.8:
#             x = x.dropna()
#             return x.max() if not x.empty else np.nan
#         return _collapse_text(s)

#     def _collapse_any(s: pd.Series):
#         if s is None or len(s) == 0:
#             return np.nan
#         if _is_numeric_series(s):
#             return _collapse_numeric(s)
#         return _collapse_mixed(s)

#     # ------------------------------------------------------------------
#     # patient-wise aggregation first to avoid counting same patient twice
#     # for the same family variant key
#     # ------------------------------------------------------------------
#     per_patient = (
#         df.groupby(pat_key, dropna=False)
#           .agg(
#               PATIENT_CANCER_STATUS=("PATIENT_CANCER_STATUS", lambda x: _to_num(x).fillna(0).astype(int).max()),
#               PAT_SIRIUS_FLAG=("SIRIUS_FLAG_ROW", lambda x: _to_num(x).fillna(0).astype(int).max()),
#               PAT_SIRIUS_GERM_SNV_PASS=("SIRIUS_PASS_SNV_GERMLINE", lambda x: _to_num(x).fillna(0).astype(int).max()),
#               PAT_SIRIUS_SOM_SNV_PASS=("SIRIUS_PASS_SNV_SOMATIC", lambda x: _to_num(x).fillna(0).astype(int).max()),
#               PAT_SIRIUS_CNV_PASS=("SIRIUS_PASS_CNV", lambda x: _to_num(x).fillna(0).astype(int).max()),
#               HAS_GERM=("HAS_GERM", lambda x: _to_num(x).fillna(0).astype(int).max()) if "HAS_GERM" in df.columns else ("PATIENT_ID", lambda x: 0),
#               HAS_SOM=("HAS_SOM", lambda x: _to_num(x).fillna(0).astype(int).max()) if "HAS_SOM" in df.columns else ("PATIENT_ID", lambda x: 0),
#               ACMG_INT=("ACMG_INT", lambda x: _to_num(x).max()),
#               MAX_POP_AF=("MAX_POP_AF", lambda x: _to_num(x).max()),
#               PATHWAY_HIT=("PATHWAY_HIT", lambda x: _to_num(x).fillna(0).astype(int).max()) if "PATHWAY_HIT" in df.columns else ("PATIENT_ID", lambda x: 0),
#               PATH_DNA_REPAIR=("PATH_DNA_REPAIR", lambda x: _to_num(x).fillna(0).astype(int).max()) if "PATH_DNA_REPAIR" in df.columns else ("PATIENT_ID", lambda x: 0),
#               GENE_NAME=("GENE_NAME", uniq_join) if "GENE_NAME" in df.columns else ("PATIENT_ID", lambda x: ""),
#           )
#           .reset_index()
#     )

#     # ------------------------------------------------------------------
#     # family counts from patient-collapsed table
#     # ------------------------------------------------------------------
#     def fam_block(g: pd.DataFrame) -> pd.Series:
#         n_pat = int(g["PATIENT_ID"].nunique())
#         n_cancer = int(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"].nunique())
#         n_non = int(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"].nunique())
#         n_germ_only = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 0)).sum())
#         n_germ_and_som = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 1)).sum())
#         fam_sirius = int(_to_num(g["PAT_SIRIUS_FLAG"]).fillna(0).astype(int).max())

#         patient_id_list = uniq_join_tokens(g["PATIENT_ID"])
#         patient_id_list_cancer = uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"])
#         patient_id_list_noncancer = uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"])

#         return pd.Series({
#             "N_PATIENTS_WITH_VARIANT": n_pat,
#             "PATIENT_ID_LIST": patient_id_list,
#             "PATIENT_ID_LIST_CANCER": patient_id_list_cancer,
#             "PATIENT_ID_LIST_NONCANCER": patient_id_list_noncancer,
#             "N_CANCER_WITH_VARIANT": n_cancer,
#             "N_NONCANCER_WITH_VARIANT": n_non,
#             "N_GERM_ONLY": n_germ_only,
#             "N_GERM_AND_SOM": n_germ_and_som,
#             "FAM_SIRIUS_FLAG": fam_sirius,
#             "FAM_SIRIUS_GERM_SNV_PASS": int(_to_num(g["PAT_SIRIUS_GERM_SNV_PASS"]).fillna(0).astype(int).max()),
#             "FAM_SIRIUS_SOM_SNV_PASS": int(_to_num(g["PAT_SIRIUS_SOM_SNV_PASS"]).fillna(0).astype(int).max()),
#             "FAM_SIRIUS_CNV_PASS": int(_to_num(g["PAT_SIRIUS_CNV_PASS"]).fillna(0).astype(int).max()),
#             "FAM_ACMG_INT_MAX": _to_num(g["ACMG_INT"]).max(),
#             "FAM_MAX_POP_AF": _to_num(g["MAX_POP_AF"]).max(),
#             "FAM_PATHWAY_HIT": int(_to_num(g["PATHWAY_HIT"]).fillna(0).astype(int).max()),
#             "FAM_PATH_DNA_REPAIR": int(_to_num(g["PATH_DNA_REPAIR"]).fillna(0).astype(int).max()),
#         })

#     fam_counts = (
#         per_patient.groupby(key, dropna=False)
#         .apply(fam_block)
#         .reset_index()
#     )

#     # ------------------------------------------------------------------
#     # collapse ALL original columns from df to family level
#     # ------------------------------------------------------------------
#     extra_cols = [c for c in df.columns if c not in key]

#     payload = (
#         df.groupby(key, dropna=False)[extra_cols]
#           .agg(lambda s: _collapse_any(s))
#           .reset_index()
#     )

#     # ------------------------------------------------------------------
#     # merge counts + full payload
#     # ------------------------------------------------------------------
#     fam = fam_counts.merge(payload, on=key, how="left", suffixes=("", "_RAW"))

#     if "FAMILY_ID" in fam.columns:
#         fam["FAMILY_ID_LIST"] = fam["FAMILY_ID"].astype(str)
#     else:
#         fam["FAMILY_ID_LIST"] = ""

#     # ------------------------------------------------------------------
#     # labels and scores
#     # ------------------------------------------------------------------
#     fam["label_of_interest"] = (
#         (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1) &
#         (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0) &
#         (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) == 1)
#     ).astype(int)

#     fam["FAM_COUNT_GE_2"] = (
#         (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) == 1) &
#         (_to_num(fam["N_PATIENTS_WITH_VARIANT"]).fillna(0).astype(int) > 1)
#     ).astype(int)

#     fam["SEGREGATION_SCORE"] = 0
#     fam.loc[_to_num(fam["N_PATIENTS_WITH_VARIANT"]).fillna(0).astype(int) == 1, "SEGREGATION_SCORE"] = 1
#     fam.loc[_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1, "SEGREGATION_SCORE"] = 2
#     fam.loc[
#         (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) == 1) &
#         (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0),
#         "SEGREGATION_SCORE"
#     ] = 3
#     fam.loc[
#         (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 2) &
#         (_to_num(fam["N_NONCANCER_WITH_VARIANT"]).fillna(0).astype(int) == 0),
#         "SEGREGATION_SCORE"
#     ] = 4

#     fam["FAMILY_EVIDENCE_SCORE"] = (
#         (_to_num(fam["N_CANCER_WITH_VARIANT"]).fillna(0).astype(int) >= 1).astype(int) +
#         (_to_num(fam["N_GERM_AND_SOM"]).fillna(0).astype(int) >= 1).astype(int) +
#         (_to_num(fam["FAM_SIRIUS_FLAG"]).fillna(0).astype(int) >= 1).astype(int) +
#         (_to_num(fam.get("FAM_PATH_DNA_REPAIR", 0)).fillna(0).astype(int) >= 1).astype(int)
#     ).clip(0, 4)

#     fam["INTEGRATED_SCORE"] = (
#         _to_num(fam["FAMILY_EVIDENCE_SCORE"]).fillna(0).astype(int) +
#         _to_num(fam["SEGREGATION_SCORE"]).fillna(0).astype(int) +
#         (_to_num(fam.get("FAM_PATHWAY_HIT", 0)).fillna(0).astype(int) >= 1).astype(int) +
#         (_to_num(fam.get("FAM_PATH_DNA_REPAIR", 0)).fillna(0).astype(int) >= 1).astype(int)
#     ).clip(0, 12)

#     fam["INTEGRATED_CLASS"] = "LOW"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 4, "INTEGRATED_CLASS"] = "MODERATE"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 7, "INTEGRATED_CLASS"] = "HIGH"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 9, "INTEGRATED_CLASS"] = "TOP"

#     # ------------------------------------------------------------------
#     # corrected evidence: based only on columns that actually exist
#     # ------------------------------------------------------------------
#     reasons = []
#     for _, row in fam.iterrows():
#         r = []

#         if int(_to_num(pd.Series([row.get("FAM_SIRIUS_GERM_SNV_PASS", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("GERM_SNV_PASS")

#         if int(_to_num(pd.Series([row.get("FAM_SIRIUS_SOM_SNV_PASS", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("SOM_SNV_PASS")

#         if int(_to_num(pd.Series([row.get("FAM_SIRIUS_CNV_PASS", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("CNV_PASS")

#         if int(_to_num(pd.Series([row.get("N_GERM_AND_SOM", 0)])).fillna(0).iloc[0]) >= 1:
#             r.append("GERM_AND_SOM")

#         if int(_to_num(pd.Series([row.get("FAM_COUNT_GE_2", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("COUNT_GE_2")

#         seg_score = int(_to_num(pd.Series([row.get("SEGREGATION_SCORE", 0)])).fillna(0).iloc[0])
#         if seg_score > 0:
#             r.append(f"SEG+{seg_score}")

#         fam_ev = int(_to_num(pd.Series([row.get("FAMILY_EVIDENCE_SCORE", 0)])).fillna(0).iloc[0])
#         if fam_ev > 0:
#             r.append(f"FAM_EVD+{fam_ev}")

#         if int(_to_num(pd.Series([row.get("FAM_PATHWAY_HIT", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("PATHWAY_HIT")

#         if int(_to_num(pd.Series([row.get("FAM_PATH_DNA_REPAIR", 0)])).fillna(0).iloc[0]) == 1:
#             r.append("DNA_REPAIR")

#         if pd.notna(row.get("FAM_ACMG_INT_MAX", np.nan)):
#             r.append(f"ACMG_MAX={row.get('FAM_ACMG_INT_MAX')}")

#         if pd.notna(row.get("FAM_MAX_POP_AF", np.nan)):
#             r.append(f"MAX_POP_AF={row.get('FAM_MAX_POP_AF')}")

#         reasons.append("|".join(r))

#     fam["INTEGRATED_EVIDENCE"] = reasons

#     return sort_by_variant(fam)

# def collapse_patient_long_to_family_variants(df_long: pd.DataFrame) -> pd.DataFrame:
#     """
#     Collapse transcript-aware patient LONG rows to family-level variant/transcript rows,
#     adding family filters and labels.
#     """
#     if df_long is None or df_long.empty:
#         return pd.DataFrame()

#     df = add_patient_level_filters(df_long.copy())
#     # df = df_long.copy()
#     df["PATIENT_CANCER_STATUS"] = infer_patient_cancer_status(df)
#     key = FAM_KEY_COLS

#     # patient-wise aggregation first to avoid counting same patient twice for same key
#     per_patient = (
#         df.groupby(PAT_KEY_COLS, dropna=False)
#           .agg(
#               # FAMILY_ID=("FAMILY_ID", "first"),
#               PATIENT_CANCER_STATUS=("PATIENT_CANCER_STATUS", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#               PAT_SIRIUS_FLAG=("SIRIUS_FLAG_ROW", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#               PAT_SIRIUS_GERM_SNV_PASS=("SIRIUS_PASS_SNV_GERMLINE", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#               PAT_SIRIUS_SOM_SNV_PASS=("SIRIUS_PASS_SNV_SOMATIC", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#               PAT_SIRIUS_CNV_PASS=("SIRIUS_PASS_CNV", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#               HAS_GERM=("HAS_GERM", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()) if "HAS_GERM" in df.columns else ("PATIENT_ID", lambda x: 0),
#               HAS_SOM=("HAS_SOM", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()) if "HAS_SOM" in df.columns else ("PATIENT_ID", lambda x: 0),
#               ACMG_INT=("ACMG_INT", lambda x: pd.to_numeric(x, errors="coerce").max()),
#               MAX_POP_AF=("MAX_POP_AF", lambda x: pd.to_numeric(x, errors="coerce").max()),
#               PATHWAY_HIT=("PATHWAY_HIT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()) if "PATHWAY_HIT" in df.columns else ("PATIENT_ID", lambda x: 0),
#               PATH_DNA_REPAIR=("PATH_DNA_REPAIR", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()) if "PATH_DNA_REPAIR" in df.columns else ("PATIENT_ID", lambda x: 0),
#               GENE_NAME=("GENE_NAME", uniq_join) if "GENE_NAME" in df.columns else ("PATIENT_ID", lambda x: ""),
#           )
#           .reset_index()
#     )

#     def fam_block(g: pd.DataFrame) -> pd.Series:
#         n_pat = int(g["PATIENT_ID"].nunique())
#         n_cancer = int(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"].nunique())
#         n_non = int(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"].nunique())
#         n_germ_only = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 0)).sum())
#         n_germ_and_som = int(((g["HAS_GERM"] == 1) & (g["HAS_SOM"] == 1)).sum())
#         fam_sirius = int(pd.to_numeric(g["PAT_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int).max())

#         patient_id_list = uniq_join_tokens(g["PATIENT_ID"])
#         patient_id_list_cancer = uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 1, "PATIENT_ID"])
#         patient_id_list_noncancer = uniq_join_tokens(g.loc[g["PATIENT_CANCER_STATUS"] == 0, "PATIENT_ID"])

#         return pd.Series({
#             "N_PATIENTS_WITH_VARIANT": n_pat,
#             "PATIENT_ID_LIST": patient_id_list,
#             "PATIENT_ID_LIST_CANCER": patient_id_list_cancer,
#             "PATIENT_ID_LIST_NONCANCER": patient_id_list_noncancer,
#             "N_CANCER_WITH_VARIANT": n_cancer,
#             "N_NONCANCER_WITH_VARIANT": n_non,
#             "N_GERM_ONLY": n_germ_only,
#             "N_GERM_AND_SOM": n_germ_and_som,
#             "FAM_SIRIUS_FLAG": fam_sirius,
#             "FAM_SIRIUS_GERM_SNV_PASS": int(pd.to_numeric(g["PAT_SIRIUS_GERM_SNV_PASS"], errors="coerce").fillna(0).astype(int).max()),
#             "FAM_SIRIUS_SOM_SNV_PASS": int(pd.to_numeric(g["PAT_SIRIUS_SOM_SNV_PASS"], errors="coerce").fillna(0).astype(int).max()),
#             "FAM_SIRIUS_CNV_PASS": int(pd.to_numeric(g["PAT_SIRIUS_CNV_PASS"], errors="coerce").fillna(0).astype(int).max()),
#         })

#     fam_counts = per_patient.groupby(key, dropna=False).apply(fam_block).reset_index()

#     payload = (
#         per_patient.groupby(key, dropna=False)
#         .agg(
#             GENE_NAME=("GENE_NAME", uniq_join),
#             ACMG_INT=("ACMG_INT", lambda x: pd.to_numeric(x, errors="coerce").max()),
#             MAX_POP_AF=("MAX_POP_AF", lambda x: pd.to_numeric(x, errors="coerce").max()),
#             PATHWAY_HIT=("PATHWAY_HIT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#             PATH_DNA_REPAIR=("PATH_DNA_REPAIR", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int).max()),
#         )
#         .reset_index()
#     )

#     fam = fam_counts.merge(payload, on=key, how="left")
#     fam["FAMILY_ID_LIST"] = fam["FAMILY_ID"].astype(str)

#     fam["label_of_interest"] = (
#         (pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 1) &
#         (pd.to_numeric(fam["N_NONCANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 0) &
#         (pd.to_numeric(fam["FAM_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int) == 1)
#     ).astype(int)

#     fam["FAM_COUNT_GE_2"] = (
#         (pd.to_numeric(fam["FAM_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int) == 1) &
#         (pd.to_numeric(fam["N_PATIENTS_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) > 1)
#     ).astype(int)

#     fam["SEGREGATION_SCORE"] = 0
#     fam.loc[pd.to_numeric(fam["N_PATIENTS_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 1, "SEGREGATION_SCORE"] = 1
#     fam.loc[pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 1, "SEGREGATION_SCORE"] = 2
#     fam.loc[(pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 1) &
#             (pd.to_numeric(fam["N_NONCANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 0), "SEGREGATION_SCORE"] = 3
#     fam.loc[(pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 2) &
#             (pd.to_numeric(fam["N_NONCANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) == 0), "SEGREGATION_SCORE"] = 4

#     fam["FAMILY_EVIDENCE_SCORE"] = (
#         (pd.to_numeric(fam["N_CANCER_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) >= 1).astype(int) +
#         (pd.to_numeric(fam["N_GERM_AND_SOM"], errors="coerce").fillna(0).astype(int) >= 1).astype(int) +
#         (pd.to_numeric(fam["FAM_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int) >= 1).astype(int) +
#         (pd.to_numeric(fam.get("PATH_DNA_REPAIR", 0), errors="coerce").fillna(0).astype(int) >= 1).astype(int)
#     ).clip(0, 4)

#     fam["INTEGRATED_SCORE"] = (
#         pd.to_numeric(fam["FAMILY_EVIDENCE_SCORE"], errors="coerce").fillna(0).astype(int) +
#         pd.to_numeric(fam["SEGREGATION_SCORE"], errors="coerce").fillna(0).astype(int) +
#         (pd.to_numeric(fam.get("PATHWAY_HIT", 0), errors="coerce").fillna(0).astype(int) >= 1).astype(int) +
#         (pd.to_numeric(fam.get("PATH_DNA_REPAIR", 0), errors="coerce").fillna(0).astype(int) >= 1).astype(int)
#     ).clip(0, 12)

#     fam["INTEGRATED_CLASS"] = "LOW"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 4, "INTEGRATED_CLASS"] = "MODERATE"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 7, "INTEGRATED_CLASS"] = "HIGH"
#     fam.loc[fam["INTEGRATED_SCORE"] >= 9, "INTEGRATED_CLASS"] = "TOP"

#     reasons = []
#     for i in range(len(fam)):
#         r = []
#         if germ_pass.iat[i] == 1:
#             r.append("GERM_PASS")
#         if int(germ_tier_pts.iat[i]) > 0:
#             r.append(f"GERM_TIER+{int(germ_tier_pts.iat[i])}")
#         if two_hit_strict.iat[i] == 1:
#             r.append("TWO_HIT_STRICT")
#         elif two_hit_ev.iat[i] == 1:
#             r.append("TWO_HIT_EVIDENCE")
#         if int(seg_pts.iat[i]) > 0:
#             r.append(f"SEG+{int(seg_pts.iat[i])}")
#         if int(fam_ev_pts.iat[i]) > 0:
#             r.append(f"FAM_EVD+{int(fam_ev_pts.iat[i])}")
#         if int(ge2_pts.iat[i]) > 0:
#             r.append("COUNT_GE_2")
#         if pathway_hit.iat[i] == 1:
#             r.append("PATHWAY_HIT")
#         if int(origin_mod.iat[i]) != 0:
#             r.append(f"ORIGIN_MOD{int(origin_mod.iat[i]):+d}")
#         reasons.append("|".join(r))
#     fam["INTEGRATED_EVIDENCE"] = reasons

#     return sort_by_variant(fam)

# def collapse_all_families_to_global_variants(df_fam: pd.DataFrame) -> pd.DataFrame:
#     if df_fam is None or df_fam.empty:
#         return pd.DataFrame()

#     key = GLOBAL_KEY_COLS

#     def global_block(g: pd.DataFrame) -> pd.Series:
#         # family_id_list = uniq_list(g["FAMILY_ID"])
#         # patient_id_list = uniq_list(g["PATIENT_ID_LIST"]) if "PATIENT_ID_LIST" in g.columns else ""
#         # patient_id_list_cancer = uniq_list(g["PATIENT_ID_LIST_CANCER"]) if "PATIENT_ID_LIST_CANCER" in g.columns else ""
#         # patient_id_list_noncancer = uniq_list(g["PATIENT_ID_LIST_NONCANCER"]) if "PATIENT_ID_LIST_NONCANCER" in g.columns else ""
#         patient_id_list = uniq_join_tokens(g["PATIENT_ID_LIST"]) if "PATIENT_ID_LIST" in g.columns else ""
#         patient_id_list_cancer = uniq_join_tokens(g["PATIENT_ID_LIST_CANCER"]) if "PATIENT_ID_LIST_CANCER" in g.columns else ""
#         patient_id_list_noncancer = uniq_join_tokens(g["PATIENT_ID_LIST_NONCANCER"]) if "PATIENT_ID_LIST_NONCANCER" in g.columns else ""
#         family_id_list = uniq_join_tokens(g["FAMILY_ID"])
#         n_families = int(g["FAMILY_ID"].nunique())

#         return pd.Series({
#             "FAMILY_ID_LIST": family_id_list,
#             "PATIENT_ID_LIST_GLOBAL": patient_id_list,
#             "PATIENT_ID_LIST_CANCER_GLOBAL": patient_id_list_cancer,
#             "PATIENT_ID_LIST_NONCANCER_GLOBAL": patient_id_list_noncancer,
#             "N_FAMILIES_WITH_VARIANT": n_families,
#             "N_PATIENTS_WITH_VARIANT": int(pd.to_numeric(g.get("N_PATIENTS_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
#             "N_CANCER_WITH_VARIANT": int(pd.to_numeric(g.get("N_CANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
#             "N_NONCANCER_WITH_VARIANT": int(pd.to_numeric(g.get("N_NONCANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
#             "FAM_SIRIUS_FLAG": int(pd.to_numeric(g.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).max()),
#             "FAM_COUNT_GE_2": int(pd.to_numeric(g.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).max()),
#             "label_of_interest": int(pd.to_numeric(g.get("label_of_interest", 0), errors="coerce").fillna(0).max()),
#             "INTEGRATED_SCORE": float(pd.to_numeric(g.get("INTEGRATED_SCORE", 0), errors="coerce").fillna(0).max()),
#             "INTEGRATED_CLASS": uniq_list(g["INTEGRATED_CLASS"]) if "INTEGRATED_CLASS" in g.columns else "",
#             "GENE_NAME": uniq_list(g["GENE_NAME"]) if "GENE_NAME" in g.columns else "",
#         })

#     out = df_fam.groupby(key, dropna=False).apply(global_block).reset_index()
#     return sort_by_variant(out)

def uniq_join_tokens(s: pd.Series, seps: str = r"[|,]") -> str:
    vals = []
    for x in s.dropna():
        sx = str(x).strip()
        if sx in EMPTY_STR_SET:
            continue
        parts = [p.strip() for p in re.split(seps, sx) if p.strip() and p.strip() not in EMPTY_STR_SET]
        vals.extend(parts)
    vals = sorted(set(vals))
    return ",".join(vals)


def collapse_all_families_to_global_variants(df_fam: pd.DataFrame) -> pd.DataFrame:
    if df_fam is None or df_fam.empty:
        return pd.DataFrame()

    key = GLOBAL_KEY_COLS

    def global_block(g: pd.DataFrame) -> pd.Series:
        family_id_list = uniq_join_tokens(g["FAMILY_ID"]) if "FAMILY_ID" in g.columns else ""
        patient_id_list = uniq_join_tokens(g["PATIENT_ID_LIST"]) if "PATIENT_ID_LIST" in g.columns else ""
        patient_id_list_cancer = uniq_join_tokens(g["PATIENT_ID_LIST_CANCER"]) if "PATIENT_ID_LIST_CANCER" in g.columns else ""
        patient_id_list_noncancer = uniq_join_tokens(g["PATIENT_ID_LIST_NONCANCER"]) if "PATIENT_ID_LIST_NONCANCER" in g.columns else ""

        return pd.Series({
            "FAMILY_ID_LIST": family_id_list,
            "PATIENT_ID_LIST_GLOBAL": patient_id_list,
            "PATIENT_ID_LIST_CANCER_GLOBAL": patient_id_list_cancer,
            "PATIENT_ID_LIST_NONCANCER_GLOBAL": patient_id_list_noncancer,
            "N_FAMILIES_WITH_VARIANT": int(g["FAMILY_ID"].nunique()) if "FAMILY_ID" in g.columns else 0,
            "N_PATIENTS_WITH_VARIANT": int(pd.to_numeric(g.get("N_PATIENTS_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
            "N_CANCER_WITH_VARIANT": int(pd.to_numeric(g.get("N_CANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
            "N_NONCANCER_WITH_VARIANT": int(pd.to_numeric(g.get("N_NONCANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).sum()),
            "N_GERM_ONLY": int(pd.to_numeric(g.get("N_GERM_ONLY", 0), errors="coerce").fillna(0).sum()),
            "N_GERM_AND_SOM": int(pd.to_numeric(g.get("N_GERM_AND_SOM", 0), errors="coerce").fillna(0).sum()),
            "FAM_SIRIUS_FLAG": int(pd.to_numeric(g.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).max()),
            "FAM_COUNT_GE_2": int(pd.to_numeric(g.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).max()),
            "label_of_interest": int(pd.to_numeric(g.get("label_of_interest", 0), errors="coerce").fillna(0).max()),
            "SEGREGATION_SCORE": float(pd.to_numeric(g.get("SEGREGATION_SCORE", 0), errors="coerce").fillna(0).max()),
            "FAMILY_EVIDENCE_SCORE": float(pd.to_numeric(g.get("FAMILY_EVIDENCE_SCORE", 0), errors="coerce").fillna(0).max()),
            "INTEGRATED_SCORE": float(pd.to_numeric(g.get("INTEGRATED_SCORE", 0), errors="coerce").fillna(0).max()),
            "INTEGRATED_CLASS": uniq_join_tokens(g["INTEGRATED_CLASS"]) if "INTEGRATED_CLASS" in g.columns else "",
            "GENE_NAME": uniq_join_tokens(g["GENE_NAME"]) if "GENE_NAME" in g.columns else "",
        })

    out = df_fam.groupby(key, dropna=False).apply(global_block).reset_index()
    return sort_by_variant(out)


def export_family_filter_subsets(df_fam: pd.DataFrame, out_dir: str, family_id: str):
    if df_fam is None or df_fam.empty:
        return
    sirius = sort_by_variant(df_fam.loc[pd.to_numeric(df_fam.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    loi = sort_by_variant(df_fam.loc[pd.to_numeric(df_fam.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    ge2 = sort_by_variant(df_fam.loc[pd.to_numeric(df_fam.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1].copy())
    top = sort_by_variant(df_fam.loc[df_fam.get("INTEGRATED_CLASS", pd.Series([""] * len(df_fam))).astype(str).eq("TOP")].copy())
    high = sort_by_variant(df_fam.loc[df_fam.get("INTEGRATED_CLASS", pd.Series([""] * len(df_fam))).astype(str).isin(["HIGH", "TOP"])].copy())

    safe_save(sirius, os.path.join(out_dir, f"{family_id}__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv"), "\t")
    safe_save(loi, os.path.join(out_dir, f"{family_id}__LABEL_OF_INTEREST_EQ_1.tsv"), "\t")
    safe_save(ge2, os.path.join(out_dir, f"{family_id}__FAM_COUNT_GE_2.tsv"), "\t")
    safe_save(top, os.path.join(out_dir, f"{family_id}__INTEGRATED_TOP.tsv"), "\t")
    safe_save(high, os.path.join(out_dir, f"{family_id}__INTEGRATED_HIGH_OR_TOP.tsv"), "\t")


# =========================================================
# Simple family/global assembly
# =========================================================


# def process_one_family(family_id: str, fam_cfg: dict, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     fam_out = os.path.join(out_dir, str(family_id))
#     ensure_dir(fam_out)
#     patient_tables = []
#     for patient_id, pcfg in fam_cfg.items():
#         df = build_patient_long(str(family_id), str(patient_id), pcfg)
#         if df is None or df.empty:
#             continue
#         p_out = os.path.join(fam_out, f"{family_id}__{patient_id}__LONG.tsv")
#         safe_save(df, p_out, "\t")
#         patient_tables.append(df)
#     if not patient_tables:
#         return pd.DataFrame(), pd.DataFrame()
#     fam_long = sort_by_variant(pd.concat(patient_tables, ignore_index=True))
#     safe_save(fam_long, os.path.join(fam_out, f"{family_id}__ALL_PATIENTS__LONG.tsv"), "\t")

#     fam_variants = collapse_patient_long_to_family_variants(fam_long)
#     if fam_variants is not None and not fam_variants.empty:
#         safe_save(fam_variants, os.path.join(fam_out, f"{family_id}__FAMILY_VARIANTS.tsv"), "\t")
#         export_family_filter_subsets(fam_variants, fam_out, str(family_id))

#     return fam_long, fam_variants

def process_one_family(family_id: str, fam_cfg: dict, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fam_out = os.path.join(out_dir, str(family_id))
    ensure_dir(fam_out)

    fam_long_path = os.path.join(fam_out, f"{family_id}__ALL_PATIENTS__LONG.tsv")
    fam_variants_path = os.path.join(fam_out, f"{family_id}__FAMILY_VARIANTS.tsv")

    # If both family-level outputs already exist, load and return them directly
    if os.path.exists(fam_long_path) and os.path.exists(fam_variants_path):
        fam_long = pd.read_csv(fam_long_path, sep="\t", low_memory=False)
        fam_variants = pd.read_csv(fam_variants_path, sep="\t", low_memory=False)
        return fam_long, fam_variants

    patient_tables = []
    for patient_id, pcfg in fam_cfg.items():
        df = build_patient_long(str(family_id), str(patient_id), pcfg)
        if df is None or df.empty:
            continue
            
        p_out = os.path.join(fam_out, f"{family_id}__{patient_id}__LONG.tsv")

        # Save patient LONG only if it does not already exist
        if not os.path.exists(p_out):
            safe_save(df, p_out, "\t")

        patient_tables.append(df)

    if not patient_tables:
        return pd.DataFrame(), pd.DataFrame()

    fam_long = sort_by_variant(pd.concat(patient_tables, ignore_index=True))

    # Save fam_long only if it does not already exist
    if not os.path.exists(fam_long_path):
        safe_save(fam_long, fam_long_path, "\t")

    fam_variants = collapse_patient_long_to_family_variants(fam_long)

    if fam_variants is not None and not fam_variants.empty:
        # Save fam_variants only if it does not already exist
        if not os.path.exists(fam_variants_path):
            safe_save(fam_variants, fam_variants_path, "\t")
            export_family_filter_subsets(fam_variants, fam_out, str(family_id))

    return fam_long, fam_variants


# def process_all_families_from_json(json_path: str, out_dir: str, family_id_filter: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     ensure_dir(out_dir)
#     with open(json_path, "r", encoding="utf-8") as f:
#         families = json.load(f)
#     if family_id_filter is not None:
#         families = {str(family_id_filter): families[str(family_id_filter)]}

#     fam_longs = []
#     fam_vars = []
#     for family_id, fam_cfg in families.items():
#         df_long, df_fam = process_one_family(str(family_id), fam_cfg, out_dir)
#         if df_long is not None and not df_long.empty:
#             fam_longs.append(df_long)
#         if df_fam is not None and not df_fam.empty:
#             fam_vars.append(df_fam)
#         gc.collect()

#     if not fam_longs:
#         raise RuntimeError("No families produced any data.")

#     all_long = sort_by_variant(pd.concat(fam_longs, ignore_index=True))
#     safe_save(all_long, os.path.join(out_dir, "ALL_FAMILIES__LONG.tsv"), "\t")

#     all_family_variants = pd.DataFrame()
#     if fam_vars:
#         all_family_variants = sort_by_variant(pd.concat(fam_vars, ignore_index=True))
#         safe_save(all_family_variants, os.path.join(out_dir, "ALL_FAMILIES__FAMILY_VARIANTS.tsv"), "\t")

#         all_sirius = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_sirius, os.path.join(out_dir, "ALL_FAMILIES__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv"), "\t")

#         all_loi = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_loi, os.path.join(out_dir, "ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1.tsv"), "\t")

#         all_ge2 = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_ge2, os.path.join(out_dir, "ALL_FAMILIES__FAM_COUNT_GE_2.tsv"), "\t")

#         all_top = sort_by_variant(
#             all_family_variants.loc[
#                 all_family_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_family_variants))).astype(str).eq("TOP")
#             ].copy()
#         )
#         safe_save(all_top, os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_TOP.tsv"), "\t")

#         all_high = sort_by_variant(
#             all_family_variants.loc[
#                 all_family_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_family_variants))).astype(str).isin(["HIGH", "TOP"])
#             ].copy()
#         )
#         safe_save(all_high, os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_HIGH_OR_TOP.tsv"), "\t")

#     return all_long, all_family_variants

# def process_all_families_from_json(
#     json_path: str,
#     out_dir: str,
#     family_id_filter: Optional[str] = None
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     ensure_dir(out_dir)

#     with open(json_path, "r", encoding="utf-8") as f:
#         families = json.load(f)

#     if family_id_filter is not None:
#         families = {str(family_id_filter): families[str(family_id_filter)]}

#     fam_longs = []
#     fam_vars = []

#     for family_id, fam_cfg in families.items():
#         df_long, df_fam = process_one_family(str(family_id), fam_cfg, out_dir)

#         if df_long is not None and not df_long.empty:
#             fam_longs.append(df_long)

#         if df_fam is not None and not df_fam.empty:
#             fam_vars.append(df_fam)

#         gc.collect()

#     if not fam_longs:
#         raise RuntimeError("No families produced any data.")

#     # -------------------------------------------------
#     # ALL_FAMILIES LONG
#     # -------------------------------------------------
#     all_long = sort_by_variant(pd.concat(fam_longs, ignore_index=True))
#     safe_save(all_long, os.path.join(out_dir, "ALL_FAMILIES__LONG.tsv"), "\t")

#     # -------------------------------------------------
#     # ALL_FAMILIES FAMILY_VARIANTS (concatenation)
#     # -------------------------------------------------
#     all_family_variants = pd.DataFrame()
#     all_global_variants = pd.DataFrame()

#     if fam_vars:
#         all_family_variants = sort_by_variant(pd.concat(fam_vars, ignore_index=True))
#         safe_save(all_family_variants, os.path.join(out_dir, "ALL_FAMILIES__FAMILY_VARIANTS.tsv"), "\t")

#         # Existing filtered exports on concatenated family table
#         all_sirius = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_sirius, os.path.join(out_dir, "ALL_FAMILIES__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv"), "\t")

#         all_loi = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_loi, os.path.join(out_dir, "ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1.tsv"), "\t")

#         all_ge2 = sort_by_variant(
#             all_family_variants.loc[
#                 pd.to_numeric(all_family_variants.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1
#             ].copy()
#         )
#         safe_save(all_ge2, os.path.join(out_dir, "ALL_FAMILIES__FAM_COUNT_GE_2.tsv"), "\t")

#         all_top = sort_by_variant(
#             all_family_variants.loc[
#                 all_family_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_family_variants), index=all_family_variants.index))
#                 .astype(str).eq("TOP")
#             ].copy()
#         )
#         safe_save(all_top, os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_TOP.tsv"), "\t")

#         all_high = sort_by_variant(
#             all_family_variants.loc[
#                 all_family_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_family_variants), index=all_family_variants.index))
#                 .astype(str).isin(["HIGH", "TOP"])
#             ].copy()
#         )
#         safe_save(all_high, os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_HIGH_OR_TOP.tsv"), "\t")

#         # -------------------------------------------------
#         # TRUE GLOBAL COLLAPSE ACROSS FAMILIES
#         # -------------------------------------------------
#         all_global_variants = collapse_all_families_to_global_variants(all_family_variants)

#         if all_global_variants is not None and not all_global_variants.empty:
#             safe_save(all_global_variants, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL_VARIANTS.tsv"), "\t")

#             # Optional filtered exports on the true global table
#             global_sirius = sort_by_variant(
#                 all_global_variants.loc[
#                     pd.to_numeric(all_global_variants.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1
#                 ].copy()
#             )
#             safe_save(global_sirius, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__SIRIUS.tsv"), "\t")

#             global_loi = sort_by_variant(
#                 all_global_variants.loc[
#                     pd.to_numeric(all_global_variants.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1
#                 ].copy()
#             )
#             safe_save(global_loi, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__LABEL_OF_INTEREST.tsv"), "\t")

#             global_ge2 = sort_by_variant(
#                 all_global_variants.loc[
#                     pd.to_numeric(all_global_variants.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1
#                 ].copy()
#             )
#             safe_save(global_ge2, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__FAM_COUNT_GE_2.tsv"), "\t")

#             global_top = sort_by_variant(
#                 all_global_variants.loc[
#                     all_global_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_global_variants), index=all_global_variants.index))
#                     .astype(str).eq("TOP")
#                 ].copy()
#             )
#             safe_save(global_top, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__INTEGRATED_TOP.tsv"), "\t")

#             global_high = sort_by_variant(
#                 all_global_variants.loc[
#                     all_global_variants.get("INTEGRATED_CLASS", pd.Series([""] * len(all_global_variants), index=all_global_variants.index))
#                     .astype(str).isin(["HIGH", "TOP"])
#                 ].copy()
#             )
#             safe_save(global_high, os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__INTEGRATED_HIGH_OR_TOP.tsv"), "\t")

#     return all_long, all_family_variants, all_global_variants


def process_all_families_from_json(
    json_path: str,
    out_dir: str,
    family_id_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    print("*"*50)

    with open(json_path, "r", encoding="utf-8") as f:
        families = json.load(f)

    if family_id_filter is not None:
        families = {str(family_id_filter): families[str(family_id_filter)]}

    all_long_path = os.path.join(out_dir, "ALL_FAMILIES__LONG.tsv")
    all_family_variants_path = os.path.join(out_dir, "ALL_FAMILIES__FAMILY_VARIANTS.tsv")
    all_global_variants_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL_VARIANTS.tsv")

    # If global outputs already exist, load and return them directly
    if os.path.exists(all_long_path) and os.path.exists(all_family_variants_path):
        all_long = pd.read_csv(all_long_path, sep="\t", low_memory=False)
        all_family_variants = pd.read_csv(all_family_variants_path, sep="\t", low_memory=False)

        if os.path.exists(all_global_variants_path):
            all_global_variants = pd.read_csv(all_global_variants_path, sep="\t", low_memory=False)
        else:
            all_global_variants = pd.DataFrame()

        return all_long, all_family_variants, all_global_variants

    fam_longs = []
    fam_vars = []
    print("I am in")
    for family_id, fam_cfg in families.items():
        df_long, df_fam = process_one_family(str(family_id), fam_cfg, out_dir)
        print(df_fam.columns)
        if df_long is not None and not df_long.empty:
            fam_longs.append(df_long)

        if df_fam is not None and not df_fam.empty:
            fam_vars.append(df_fam)

        gc.collect()

    if not fam_longs:
        raise RuntimeError("No families produced any data.")

    for i, df in enumerate(fam_vars):
        print(f"fam_vars[{i}] columns:")
        print(df.columns.tolist())
    # -------------------------------------------------
    # ALL_FAMILIES LONG
    # -------------------------------------------------
    all_long = sort_by_variant(pd.concat(fam_longs, ignore_index=True))
    if not os.path.exists(all_long_path):
        safe_save(all_long, all_long_path, "\t")

    # -------------------------------------------------
    # ALL_FAMILIES FAMILY_VARIANTS (concatenation)
    # -------------------------------------------------
    all_family_variants = pd.DataFrame()
    all_global_variants = pd.DataFrame()

    if fam_vars:
        all_family_variants = sort_by_variant(pd.concat(fam_vars, ignore_index=True))
        if not os.path.exists(all_family_variants_path):
            safe_save(all_family_variants, all_family_variants_path, "\t")

        # Existing filtered exports on concatenated family table
        all_sirius_path = os.path.join(out_dir, "ALL_FAMILIES__SIRIUS_EQ_1__FAMILY_VARIANTS.tsv")
        if not os.path.exists(all_sirius_path):
            all_sirius = sort_by_variant(
                all_family_variants.loc[
                    pd.to_numeric(all_family_variants.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1
                ].copy()
            )
            safe_save(all_sirius, all_sirius_path, "\t")

        all_loi_path = os.path.join(out_dir, "ALL_FAMILIES__LABEL_OF_INTEREST_EQ_1.tsv")
        if not os.path.exists(all_loi_path):
            all_loi = sort_by_variant(
                all_family_variants.loc[
                    pd.to_numeric(all_family_variants.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1
                ].copy()
            )
            safe_save(all_loi, all_loi_path, "\t")

        all_ge2_path = os.path.join(out_dir, "ALL_FAMILIES__FAM_COUNT_GE_2.tsv")
        if not os.path.exists(all_ge2_path):
            all_ge2 = sort_by_variant(
                all_family_variants.loc[
                    pd.to_numeric(all_family_variants.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1
                ].copy()
            )
            safe_save(all_ge2, all_ge2_path, "\t")

        all_top_path = os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_TOP.tsv")
        if not os.path.exists(all_top_path):
            all_top = sort_by_variant(
                all_family_variants.loc[
                    all_family_variants.get(
                        "INTEGRATED_CLASS",
                        pd.Series([""] * len(all_family_variants), index=all_family_variants.index)
                    ).astype(str).eq("TOP")
                ].copy()
            )
            safe_save(all_top, all_top_path, "\t")

        all_high_path = os.path.join(out_dir, "ALL_FAMILIES__INTEGRATED_HIGH_OR_TOP.tsv")
        if not os.path.exists(all_high_path):
            all_high = sort_by_variant(
                all_family_variants.loc[
                    all_family_variants.get(
                        "INTEGRATED_CLASS",
                        pd.Series([""] * len(all_family_variants), index=all_family_variants.index)
                    ).astype(str).isin(["HIGH", "TOP"])
                ].copy()
            )
            safe_save(all_high, all_high_path, "\t")

        # -------------------------------------------------
        # TRUE GLOBAL COLLAPSE ACROSS FAMILIES
        # -------------------------------------------------
        all_global_variants = collapse_all_families_to_global_variants(all_family_variants)

        if all_global_variants is not None and not all_global_variants.empty:
            if not os.path.exists(all_global_variants_path):
                safe_save(all_global_variants, all_global_variants_path, "\t")

            global_sirius_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__SIRIUS.tsv")
            if not os.path.exists(global_sirius_path):
                global_sirius = sort_by_variant(
                    all_global_variants.loc[
                        pd.to_numeric(all_global_variants.get("FAM_SIRIUS_FLAG", 0), errors="coerce").fillna(0).astype(int) == 1
                    ].copy()
                )
                safe_save(global_sirius, global_sirius_path, "\t")

            global_loi_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__LABEL_OF_INTEREST.tsv")
            if not os.path.exists(global_loi_path):
                global_loi = sort_by_variant(
                    all_global_variants.loc[
                        pd.to_numeric(all_global_variants.get("label_of_interest", 0), errors="coerce").fillna(0).astype(int) == 1
                    ].copy()
                )
                safe_save(global_loi, global_loi_path, "\t")

            global_ge2_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__FAM_COUNT_GE_2.tsv")
            if not os.path.exists(global_ge2_path):
                global_ge2 = sort_by_variant(
                    all_global_variants.loc[
                        pd.to_numeric(all_global_variants.get("FAM_COUNT_GE_2", 0), errors="coerce").fillna(0).astype(int) == 1
                    ].copy()
                )
                safe_save(global_ge2, global_ge2_path, "\t")

            global_top_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__INTEGRATED_TOP.tsv")
            if not os.path.exists(global_top_path):
                global_top = sort_by_variant(
                    all_global_variants.loc[
                        all_global_variants.get(
                            "INTEGRATED_CLASS",
                            pd.Series([""] * len(all_global_variants), index=all_global_variants.index)
                        )
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .str.contains("TOP", regex=False)
                    ].copy()
                )
                safe_save(global_top, global_top_path, "\t")
            global_high_path = os.path.join(out_dir, "ALL_FAMILIES__GLOBAL__INTEGRATED_HIGH_OR_TOP.tsv")
            if not os.path.exists(global_high_path):
                global_high = sort_by_variant(
                    all_global_variants.loc[
                        all_global_variants.get(
                            "INTEGRATED_CLASS",
                            pd.Series([""] * len(all_global_variants), index=all_global_variants.index)
                        )
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .str.contains("TOP", regex=False)
                        |
                        all_global_variants.get(
                            "INTEGRATED_CLASS",
                            pd.Series([""] * len(all_global_variants), index=all_global_variants.index)
                        )
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .str.contains("HIGH", regex=False)
                    ].copy()
                )
                safe_save(global_high, global_high_path, "\t")
            # if not os.path.exists(global_high_path):
            #     global_high = sort_by_variant(
            #         all_global_variants.loc[
            #             all_global_variants.get(
            #                 "INTEGRATED_CLASS",
            #                 pd.Series([""] * len(all_global_variants), index=all_global_variants.index)
            #             ).astype(str).isin(["HIGH", "TOP"])
            #         ].copy()
            #     )
            #     safe_save(global_high, global_high_path, "\t")
            

    return all_long, all_family_variants, all_global_variants

# =========================================================
# CLI
# =========================================================


def main():
    print("GOOD")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON: {family_id: {patient_id: {...}}}")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--family_id", default=None, help="Run only one family_id")
    args = ap.parse_args()
    process_all_families_from_json(args.config, args.out_dir, args.family_id)
    print("[DONE]")


if __name__ == "__main__":
    main()
