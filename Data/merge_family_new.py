import os
import re
import gc
import json
import argparse
from typing import Dict, List, Tuple, Optional


import numpy as np
import pandas as pd



def normalize_chr(x) -> str:
    return str(x).replace("chr", "").replace("CHR", "").strip()

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def safe_save(df: pd.DataFrame, path: str, sep: str):
    ensure_dir(os.path.dirname(path))
    try:
        df.to_csv(path, sep=sep, index=False)
        print(f"[SAVED] {path} | rows={len(df)} cols={df.shape[1]}")
    except Exception as e:
        print(f"[SAVE_ERROR] {path} -> {type(e).__name__}: {e}")

def chrom_sort_key(ch: str):
    s = normalize_chr(ch)
    mapping = {str(i): i for i in range(1, 23)}
    mapping.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
    return mapping.get(s, 1000), s

def sort_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df2 = df.copy()
    for c in ["CHROM", "POS"]:
        if c not in df2.columns:
            return df2
    df2["_chr_rank"] = df2["CHROM"].astype(str).map(lambda x: chrom_sort_key(x)[0])
    df2["_chr_str"]  = df2["CHROM"].astype(str).map(lambda x: chrom_sort_key(x)[1])
    df2["POS"] = pd.to_numeric(df2["POS"], errors="coerce")
    sort_cols = ["_chr_rank", "_chr_str", "POS", "REF", "ALT"]
    if "TX" in df2.columns:
        sort_cols.append("TX")
    df2 = df2.sort_values(by=sort_cols, kind="mergesort")
    df2 = df2.drop(columns=["_chr_rank", "_chr_str"], errors="ignore")
    return df2

def list_files_one(dirpath: str, suffixes: Tuple[str, ...]) -> Optional[str]:
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

def num_max_or_first(x):
    xn = pd.to_numeric(x, errors="coerce")
    if xn.notna().any():
        return float(xn.max())
    return first_nonnull(x)

def build_keep_all_payload_aggs(
    df: pd.DataFrame,
    key_cols: List[str],
    fixed_aggs: Dict[str, tuple],
    join_cols: Optional[set] = None,
    drop_cols: Optional[set] = None,
) -> Dict[str, tuple]:

    join_cols = join_cols or set()
    drop_cols = drop_cols or set()

    agg = dict(fixed_aggs)

    skip = set(key_cols) | set(agg.keys()) | set(drop_cols)

    for c in df.columns:
        if c in skip:
            continue
        if c in join_cols:
            agg[c] = (c, uniq_join)
        else:
            agg[c] = (c, num_max_or_first)

    return agg



def pick_one_from_dirs(run: dict, key: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    p = run.get(key)
    if not p:
        return None
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        return list_files_one(p, suffixes)
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

    for c in ["REF", "ALT"]:
        if c not in df.columns:
            df[c] = ""

    return df

def standardize_tx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Feature_RefSeq" in out.columns:
        out["TX"] = out["Feature_RefSeq"].astype(str).replace({"nan": "", "None": ""}).fillna("")
        out.loc[out["TX"].isin(["", ".", "NA"]), "TX"] = "NO_TX"
    else:
        out["TX"] = "NO_TX"
    return out

def detect_sample_prefix(df: pd.DataFrame) -> Optional[str]:

    dp_cols = [c for c in df.columns if c.endswith(".DP")]
    if not dp_cols:
        return None
    return dp_cols[0].rsplit(".", 1)[0]  # "b2" from "b2.DP"

def add_unified_dp_vaf(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    prefix = detect_sample_prefix(out)
    dp_col  = f"{prefix}.DP"  if prefix and f"{prefix}.DP"  in out.columns else None
    baf_col = f"{prefix}.BAF" if prefix and f"{prefix}.BAF" in out.columns else None
    ad_col  = f"{prefix}.AD"  if prefix and f"{prefix}.AD"  in out.columns else None

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
            s = str(x).strip()
            if s in {"", ".", "NA", "nan", "None"}:
                return np.nan
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 2:
                return np.nan
            try:
                ref = float(parts[0]); alt = float(parts[1])
                tot = ref + alt
                return alt / tot if tot > 0 else np.nan
            except Exception:
                return np.nan
        out["VAF_UNI"] = out[ad_col].map(vaf_from_ad)
    else:
        out["VAF_UNI"] = np.nan

    return out

def parse_acmg_to_int(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = x[0]
    if x is None or (isinstance(x, float) and np.isnan(x)):
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

def add_acmg_int_from_exome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    acmg_col = None
    for c in ["Prediction_ACMG_tapes", "EXOME_Prediction_ACMG_tapes", "Prediction_ACMG",
              "EXOME_Prediction_ACMG", "Classification_ACMG", "EXOME_Classification_ACMG"]:
        if c in out.columns:
            acmg_col = c
            break
    out["ACMG_RAW"] = out[acmg_col] if acmg_col else np.nan
    out["ACMG_INT"] = out["ACMG_RAW"].map(parse_acmg_to_int)
    out["ACMG_INT"] = pd.to_numeric(out["ACMG_INT"], errors="coerce")
    return out


def compute_max_pop_af(df: pd.DataFrame) -> pd.Series:
    af_cols = [
        "IG_AF", "MAX_AF",
        "gnomAD_Global_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF",
        "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF",
        "1000G_Global_AF", "1000G_AFR_AF", "1000G_AMR_AF", "1000G_EAS_AF", "1000G_EUR_AF", "1000G_SAS_AF",
        "Kaviar_AF", "EVS_EA_MAF", "EVS_CA",
    ]
    present = [c for c in af_cols if c in df.columns]
    if not present:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[present].apply(lambda x: pd.to_numeric(x, errors="coerce")).max(axis=1, skipna=True)


def consequence_pass_nonsyn(df: pd.DataFrame) -> pd.Series:
    cons_col = None
    for c in ["EXOME_Consequence", "Consequence"]:
        if c in df.columns:
            cons_col = c
            break
    if cons_col is None:
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

def match_cnv_to_variants(variants: pd.DataFrame, cnv: pd.DataFrame) -> pd.DataFrame:
    out = variants.copy()
    if "LOH_PARTIAL_PATHO" not in out.columns:
        out["LOH_PARTIAL_PATHO"] = 0
    return out

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
    return out

def add_pathway_flags_from_exome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create PATHWAY_HIT and PATH_DNA_REPAIR using EXOME annotation columns if present.
    This is origin-agnostic (works on germline and somatic rows alike).
    """
    out = df.copy()

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
        "KEGG_Pathway",
        "GO_biological_process",
        "Function_description",
        "Disease_description",
    ]

    texts = []
    for i in range(len(out)):
        parts = []
        for c in col_candidates:
            if c in out.columns:
                v = norm_text(out.iloc[i][c])
                if v:
                    parts.append(v)
        texts.append(" | ".join(parts))
    text_series = pd.Series(texts, index=out.index)

    dna_repair_re = re.compile(
        r"(?:dna repair|dna damage|response to dna damage|double[- ]strand break|homologous recombination|"
        r"mismatch repair|nucleotide excision repair|base excision repair|non[- ]homologous end joining|"
        r"fanconi|genome stability|checkpoint)",
        re.IGNORECASE
    )

    cancer_path_re = re.compile(
        r"(?:p53|pi3k|akt|mtor|mapk|ras|wnt|tgf[- ]?beta|notch|hippo|jak[- ]stat|cell cycle|apoptosis|"
        r"dna repair|dna damage|homologous recombination|mismatch repair|excisi?on repair|fanconi)",
        re.IGNORECASE
    )

    out["PATH_DNA_REPAIR"] = text_series.str.contains(dna_repair_re, na=False).astype(int)
    out["PATHWAY_HIT"]     = text_series.str.contains(cancer_path_re, na=False).astype(int)
    return out




def add_sirius_pass_flags_long(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    if "ORIGIN" not in df.columns:
        raise ValueError("Expected ORIGIN in LONG table.")

    # Ensure transcript + ACMG + pathways exist
    if "TX" not in df.columns:
        df = standardize_tx(df)

    if "ACMG_INT" not in df.columns:
        df = add_acmg_int_from_exome(df)

    for c in ["PATHWAY_HIT", "PATH_DNA_REPAIR"]:
        if c not in df.columns:
            df[c] = 0

    # MAX_POP_AF and nonsyn flags
    df["MAX_POP_AF"] = compute_max_pop_af(df)
    df["SIRIUS_PASS_POPAF"] = ((df["MAX_POP_AF"].isna()) | (df["MAX_POP_AF"] < 0.01)).astype(int)
    df["SIRIUS_PASS_NONSYN"] = consequence_pass_nonsyn(df)

    # ACMG pass flags
    df["ACMG_INT"] = pd.to_numeric(df["ACMG_INT"], errors="coerce")
    df["SIRIUS_PASS_ACMG_SNV"] = df["ACMG_INT"].isin([0, 3, 4, 5]).astype(int)
    df["SIRIUS_PASS_ACMG_CNV"] = df["ACMG_INT"].isin([4, 5]).astype(int)

    # Ensure DP_UNI/VAF_UNI exist for germline rows
    if "DP_UNI" not in df.columns or "VAF_UNI" not in df.columns:
        df = add_unified_dp_vaf(df)

    df["DP_UNI"] = pd.to_numeric(df["DP_UNI"], errors="coerce")
    df["VAF_UNI"] = pd.to_numeric(df["VAF_UNI"], errors="coerce")

    is_germ = df["ORIGIN"].astype(str).str.lower().eq("germline")
    is_som  = df["ORIGIN"].astype(str).str.lower().eq("somatic")

    #DP/VAF used for QC (critical fix) ---
    dp_eff = df["DP_UNI"].copy()
    vaf_eff = df["VAF_UNI"].copy()

    if "DP_SOM" in df.columns:
        dp_eff = np.where(is_som, pd.to_numeric(df["DP_SOM"], errors="coerce"), dp_eff)
    if "VAF_SOM" in df.columns:
        vaf_eff = np.where(is_som, pd.to_numeric(df["VAF_SOM"], errors="coerce"), vaf_eff)

    dp_eff = pd.to_numeric(dp_eff, errors="coerce")
    vaf_eff = pd.to_numeric(vaf_eff, errors="coerce")

    # QC thresholds
    pass_dp_g  = (dp_eff > 10).fillna(False)
    pass_vaf_g = (vaf_eff > 0.25).fillna(False)

    pass_dp_s  = (dp_eff > 20).fillna(False)
    pass_vaf_s = (vaf_eff > 0.05).fillna(False)

    base_req = ["SIRIUS_PASS_ACMG_SNV", "SIRIUS_PASS_POPAF", "SIRIUS_PASS_NONSYN"]
    base_ok = (df[base_req].sum(axis=1) == len(base_req))

    df["SIRIUS_PASS_SNV_GERMLINE"] = (is_germ & base_ok & pass_dp_g & pass_vaf_g).astype(int)
    df["SIRIUS_PASS_SNV_SOMATIC"]  = (is_som  & base_ok & pass_dp_s & pass_vaf_s).astype(int)

    # CNV/LOH
    if "LOH_PARTIAL_PATHO" not in df.columns:
        df["LOH_PARTIAL_PATHO"] = 0
    df["LOH_PARTIAL_PATHO"] = pd.to_numeric(df["LOH_PARTIAL_PATHO"], errors="coerce").fillna(0).astype(int)

    df["SIRIUS_PASS_CNV"] = ((df["LOH_PARTIAL_PATHO"] == 1) & (df["SIRIUS_PASS_ACMG_CNV"] == 1)).astype(int)

    df["SIRIUS_FLAG_ROW"] = (
        (df["SIRIUS_PASS_SNV_GERMLINE"] == 1) |
        (df["SIRIUS_PASS_SNV_SOMATIC"]  == 1) |
        (df["SIRIUS_PASS_CNV"]          == 1)
    ).astype(int)

    return df


def resolve_gene_col(df: pd.DataFrame) -> str:
    candidates = ["GENE_NAME", "Gene_Name", "Gene", "GENE", "SYMBOL", "HGNC_Name", "HGNC"]
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def attach_global_lists(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "FAMILY_ID_LIST_GLOBAL" not in out.columns:
        out["FAMILY_ID_LIST_GLOBAL"] = out.get("FAMILY_ID_LIST", out.get("FAMILY_ID", "")).astype(str)
    if "PATIENT_ID_LIST_GLOBAL" not in out.columns:
        out["PATIENT_ID_LIST_GLOBAL"] = out.get("PATIENT_ID_LIST", "").astype(str)
    out["FAMILY_ID_LIST_GLOBAL"] = out["FAMILY_ID_LIST_GLOBAL"].replace({"nan": "", "None": ""}).fillna("")
    out["PATIENT_ID_LIST_GLOBAL"] = out["PATIENT_ID_LIST_GLOBAL"].replace({"nan": "", "None": ""}).fillna("")
    return out

def export_minimal_csv(df: pd.DataFrame, out_csv_path: str, label_cols: List[str]):
    base_cols = ["CHROM", "POS", "REF", "ALT", "TX", "GENE_NAME",
                 "PATIENT_ID_LIST_GLOBAL", "FAMILY_ID_LIST_GLOBAL"] + label_cols

    if df is None or df.empty:
        safe_save(pd.DataFrame(columns=base_cols), out_csv_path, sep=",")
        return

    tmp = attach_global_lists(df)
    gene_col = resolve_gene_col(tmp)
    if gene_col and gene_col != "GENE_NAME":
        tmp = tmp.rename(columns={gene_col: "GENE_NAME"})
    if "GENE_NAME" not in tmp.columns:
        tmp["GENE_NAME"] = ""

    if "TX" not in tmp.columns:
        tmp["TX"] = "NO_TX"

    for c in base_cols:
        if c not in tmp.columns:
            tmp[c] = "" if c in ["GENE_NAME", "TX", "PATIENT_ID_LIST_GLOBAL", "FAMILY_ID_LIST_GLOBAL"] else 0

    out = tmp[base_cols].copy()
    out = sort_by_variant(out)
    safe_save(out, out_csv_path, sep=",")




def add_cnv_and_het_annotations(
    df_vars: pd.DataFrame,
    cnv_path: Optional[str],
    het_path: Optional[str],
) -> pd.DataFrame:
    out = df_vars.copy()

    # --- CNV / LOH ---
    if cnv_path and os.path.exists(cnv_path):
        cnv = load_cnv(cnv_path)
        out = match_cnv_to_variants(out, cnv)
    else:
        out["CNV_LOH_status"] = "Unknown"
        out["CNV_pathogenic"] = 0
        out["LOH_PARTIAL_PATHO"] = 0

    # --- HET / ROH ---
    if het_path and os.path.exists(het_path):
        hom = load_hom_plink(het_path)
        out = annotate_variants_with_hom(out, hom)
    else:
        out["IN_ROH"] = 0
        out["HET_HIGH_HOM"] = 0

    for c in ["CNV_pathogenic", "LOH_PARTIAL_PATHO", "IN_ROH", "HET_HIGH_HOM"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(0).astype(int)

    return out

def first_nonnull(s: pd.Series):
    t = s.dropna()
    if t.empty:
        return np.nan
    t2 = t.astype(str)
    mask = ~t2.isin(["", ".", "nan", "None", "NA"])
    t3 = t[mask]
    return t3.iloc[0] if len(t3) else np.nan

def uniq_join(s: pd.Series) -> str:
    vals = [str(x) for x in s.dropna().astype(str).unique().tolist()]
    vals = [v for v in vals if v not in {"", ".", "nan", "None", "NA"}]
    return "|".join(sorted(vals))

def pick_one_from_path_or_dir(p: Optional[str], suffixes: Tuple[str, ...]) -> Optional[str]:
    if not p:
        return None
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        return list_files_one(p, suffixes)
    return None

def prep_exome_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply common standardization for both germline and somatic exome tables.
    """
    out = df.copy()
    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out = standardize_tx(out)
    out = add_unified_dp_vaf(out)
    out = add_acmg_int_from_exome(out)
    out = add_pathway_flags_from_exome(out)
    if "LOH_PARTIAL_PATHO" not in out.columns:
        out["LOH_PARTIAL_PATHO"] = 0
    out["LOH_PARTIAL_PATHO"] = pd.to_numeric(out["LOH_PARTIAL_PATHO"], errors="coerce").fillna(0).astype(int)
    return out




def collapse_somatic_sources_one_run(df_som_concat: pd.DataFrame) -> pd.DataFrame:

    if df_som_concat is None or df_som_concat.empty:
        return pd.DataFrame()

    key = ["FAMILY_ID", "PATIENT_ID", "CHROM", "POS", "REF", "ALT", "TX"]
    df = df_som_concat.copy()
    df["_ones"] = 1

    fixed = {
        "SOMATIC_REPEAT_COUNT": ("_ones", "sum"),
        "SOMATIC_SOURCE_SET": ("SOMATIC_SOURCE", uniq_join),

        "DP_SOM": ("DP_UNI", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "VAF_SOM": ("VAF_UNI", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "ACMG_INT_SOM": ("ACMG_INT", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "LOH_PARTIAL_PATHO_SOM": ("LOH_PARTIAL_PATHO", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PATHWAY_HIT": ("PATHWAY_HIT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PATH_DNA_REPAIR": ("PATH_DNA_REPAIR", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
    }

    join_cols = {
        "SOMATIC_SOURCE",           
        "FILTER", "ClinVar_CLNSIG", "ClinVar_CLNDN", "ClinVar_CLNDISDB",
        "COSMIC", "Hotspots_Kit", "ONCOGENIC", "HIGHEST_LEVEL",
        "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4",
    }

    drop_cols = {"_ones"}

    agg_map = build_keep_all_payload_aggs(
        df=df,
        key_cols=key,
        fixed_aggs=fixed,
        join_cols=join_cols,
        drop_cols=drop_cols,
    )

    out = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    out = out.drop(columns=["_ones"], errors="ignore")

    out["ORIGIN"] = "somatic"
    out["HAS_SOM"] = 1
    return out


def collapse_somatic_across_runs(som_runs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Collapse multiple run-level somatic tables into ONE somatic-per-patient table,
    but KEEP ALL other columns.
    """
    if not som_runs:
        return pd.DataFrame()

    df = pd.concat(som_runs, ignore_index=True)
    if df.empty:
        return df

    key = ["FAMILY_ID", "PATIENT_ID", "CHROM", "POS", "REF", "ALT", "TX"]
    df["_ones"] = 1

    fixed = {
        "SOMATIC_REPEAT_COUNT": ("SOMATIC_REPEAT_COUNT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()),
        "SOMATIC_SOURCE_SET": ("SOMATIC_SOURCE_SET", uniq_join),

        "DP_SOM": ("DP_SOM", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "VAF_SOM": ("VAF_SOM", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "ACMG_INT_SOM": ("ACMG_INT_SOM", lambda x: pd.to_numeric(x, errors="coerce").max()),
        "LOH_PARTIAL_PATHO_SOM": ("LOH_PARTIAL_PATHO_SOM", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PATHWAY_HIT": ("PATHWAY_HIT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PATH_DNA_REPAIR": ("PATH_DNA_REPAIR", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
    }

    join_cols = {
        "SOMATIC_SOURCE_SET",
        "FILTER", "ClinVar_CLNSIG", "ClinVar_CLNDN", "ClinVar_CLNDISDB",
        "COSMIC", "Hotspots_Kit", "ONCOGENIC", "HIGHEST_LEVEL",
        "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4",
    }

    agg_map = build_keep_all_payload_aggs(
        df=df,
        key_cols=key,
        fixed_aggs=fixed,
        join_cols=join_cols,
        drop_cols={"_ones"},
    )

    out = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    out = out.drop(columns=["_ones"], errors="ignore")

    out["ORIGIN"] = "somatic"
    out["HAS_SOM"] = 1
    return out


def vcf_to_exome_path(vcf_path: str) -> Optional[str]:

    if not vcf_path:
        return None

    vcf_path = str(vcf_path)

    if not vcf_path.endswith((".vcf", ".vcf.gz")):
        return None

    folder = os.path.dirname(vcf_path)
    name = os.path.basename(vcf_path)

    name = name.replace(".vcf.gz", "").replace(".vcf", "")

    name = name.replace("_Genetic_Variants", "_Exome_Genetic_Variants")

    exome = os.path.join(folder, name + ".txt")

    return exome if os.path.exists(exome) else None

def normalize_acmg_exome_origin(df: pd.DataFrame, origin: str) -> pd.DataFrame:

    out = df.copy()

    acmg_col = None
    for c in [
        "Prediction_ACMG_tapes",
        "EXOME_Prediction_ACMG_tapes",
        "Prediction_ACMG",
        "EXOME_Prediction_ACMG",
    ]:
        if c in out.columns:
            acmg_col = c
            break

    if acmg_col is None:
        out["ACMG_INT"] = np.nan
        return out

    out["ACMG_INT"] = out[acmg_col].map(parse_acmg_to_int)
    out["ACMG_INT"] = pd.to_numeric(out["ACMG_INT"], errors="coerce")

    if origin == "germline":
        out["GERM_Prediction_ACMG_tapes"] = out[acmg_col]
    else:
        out["SOM_Prediction_ACMG_tapes"] = out[acmg_col]

    return out




def pick_single_txt_from_dir(dirpath: Optional[str]) -> Optional[str]:

    if not dirpath:
        return None

    if os.path.isfile(dirpath):
        return dirpath

    if os.path.isdir(dirpath):
        txt_files = [
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.endswith(".txt")
        ]

        if len(txt_files) == 0:
            return None

        if len(txt_files) > 1:
            raise ValueError(
                f"Expected exactly one .txt in {dirpath}, found {len(txt_files)}:\n"
                + "\n".join(txt_files)
            )

        return txt_files[0]

    return None

def pick_single_txt_from_dir(dirpath: Optional[str]) -> Optional[str]:

    if not dirpath:
        return None

    if os.path.isfile(dirpath):
        return dirpath

    if os.path.isdir(dirpath):
        txt_files = [
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.endswith(".txt")
        ]

        if len(txt_files) == 0:
            return None

        if len(txt_files) > 1:
            raise ValueError(
                f"Expected exactly one .txt in {dirpath}, found {len(txt_files)}:\n"
                + "\n".join(txt_files)
            )

        return txt_files[0]

    return None




def build_patient_long_exome_only(family_id: str, patient_id: str, cfg: dict) -> pd.DataFrame:

    frames: List[pd.DataFrame] = []

    has_somatic_patient = int(bool(cfg.get("somatic_runs")))

    germ = (cfg.get("germline_files") or {})
    germ_ptr = germ.get("vcf")

    germ_exome = pick_one_from_path_or_dir(vcf_to_exome_path(germ_ptr), (".tsv", ".txt", ".csv"))
    if germ_exome is None:
        germ_exome = vcf_to_exome_path(germ_ptr)

    if germ_exome and os.path.exists(germ_exome):
        df_g = load_exome_table(germ_exome)
        if df_g is not None and not df_g.empty:
            df_g = prep_exome_common(df_g)
            df_g = add_cnv_and_het_annotations(df_g, cnv_path=germ.get("cnv"), het_path=germ.get("het"))
            df_g = normalize_acmg_exome_origin(df_g, "germline")

            df_g["FAMILY_ID"] = str(family_id)
            df_g["PATIENT_ID"] = str(patient_id)
            df_g["ORIGIN"] = "germline"
            df_g["HAS_SOMATIC_PATIENT"] = has_somatic_patient
            frames.append(df_g)


    som_runs_out: List[pd.DataFrame] = []

    for run in (cfg.get("somatic_runs", []) or []):
        g_exome = pick_single_txt_from_dir(run.get("vcf_genetic_dir"))
        o_exome = pick_single_txt_from_dir(run.get("vcf_oncology_dir"))

        cnv_path_g = pick_single_txt_from_dir(run.get("cnv_genetic_dir"))
        cnv_path_o = pick_single_txt_from_dir(run.get("cnv_oncology_dir"))

        het_path = pick_single_txt_from_dir(run.get("het_dir"))

        som_frames: List[pd.DataFrame] = []

        if g_exome and os.path.exists(g_exome):
            dg = load_exome_table(g_exome)
            if dg is not None and not dg.empty:
                dg = prep_exome_common(dg)
                dg = add_cnv_and_het_annotations(dg, cnv_path=cnv_path_g, het_path=het_path)
                dg = normalize_acmg_exome_origin(dg, "somatic")
                dg["SOMATIC_SOURCE"] = "genetic"
                som_frames.append(dg)

        if o_exome and os.path.exists(o_exome):
            do = load_exome_table(o_exome)
            if do is not None and not do.empty:
                do = prep_exome_common(do)
                do = add_cnv_and_het_annotations(do, cnv_path=cnv_path_o, het_path=het_path)
                do = normalize_acmg_exome_origin(do, "somatic")
                do["SOMATIC_SOURCE"] = "oncology"
                som_frames.append(do)

        if not som_frames:
            continue

        d = pd.concat(som_frames, ignore_index=True)
        d["FAMILY_ID"] = str(family_id)
        d["PATIENT_ID"] = str(patient_id)
        d["ORIGIN"] = "somatic"
        d["HAS_SOMATIC_PATIENT"] = 1

        d_run = collapse_somatic_sources_one_run(d)
        if d_run is not None and not d_run.empty:
            som_runs_out.append(d_run)

    df_som = collapse_somatic_across_runs(som_runs_out) if som_runs_out else pd.DataFrame()
    if df_som is not None and not df_som.empty:
        df_som["FAMILY_ID"] = str(family_id)
        df_som["PATIENT_ID"] = str(patient_id)
        df_som["ORIGIN"] = "somatic"
        df_som["HAS_SOMATIC_PATIENT"] = 1
        frames.append(df_som)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    for c in ["CHROM", "POS", "REF", "ALT", "Feature_RefSeq"]:
        if c not in out.columns:
            out[c] = "" if c != "POS" else np.nan

    out["CHROM"] = out["CHROM"].astype(str).map(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out = sort_by_variant(out)
    return out



def ensure_has_somatic_patient(df_pat: pd.DataFrame, df_fam_long: pd.DataFrame) -> pd.DataFrame:

    out = df_pat.copy()

    if "HAS_SOMATIC_PATIENT" in out.columns:
        out["HAS_SOMATIC_PATIENT"] = pd.to_numeric(out["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int)
        return out

    if df_fam_long is None or df_fam_long.empty:
        out["HAS_SOMATIC_PATIENT"] = 0
        return out

    if "HAS_SOMATIC_PATIENT" in df_fam_long.columns:
        hs = (
            df_fam_long.groupby(["FAMILY_ID", "PATIENT_ID"], dropna=False)["HAS_SOMATIC_PATIENT"]
            .max()
            .reset_index()
        )
        out = out.merge(hs, on=["FAMILY_ID", "PATIENT_ID"], how="left")
        out["HAS_SOMATIC_PATIENT"] = pd.to_numeric(out["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int)
    else:
        out["HAS_SOMATIC_PATIENT"] = 0

    return out


def collapse_family_long_to_patient_variants_strategy_b(df_family_long: pd.DataFrame) -> pd.DataFrame:

    if df_family_long is None or df_family_long.empty:
        return pd.DataFrame()

    df = df_family_long.copy()

    for c in [
        "SIRIUS_PASS_SNV_GERMLINE", "SIRIUS_PASS_SNV_SOMATIC", "SIRIUS_PASS_CNV", "SIRIUS_FLAG_ROW",
        "PATHWAY_HIT", "PATH_DNA_REPAIR",
        "DP_UNI", "VAF_UNI", "ACMG_INT"
    ]:
        if c not in df.columns:
            df[c] = 0

    key = ["FAMILY_ID", "PATIENT_ID", "CHROM", "POS", "REF", "ALT", "TX"]

    is_germ = df["ORIGIN"].astype(str).str.lower().eq("germline")
    is_som  = df["ORIGIN"].astype(str).str.lower().eq("somatic")

    df["_is_germ"] = is_germ.astype(int)
    df["_is_som"]  = is_som.astype(int)

    df["_dp_g"]   = np.where(is_germ, pd.to_numeric(df.get("DP_UNI", np.nan), errors="coerce"), np.nan)
    df["_vaf_g"]  = np.where(is_germ, pd.to_numeric(df.get("VAF_UNI", np.nan), errors="coerce"), np.nan)
    df["_acmg_g"] = np.where(is_germ, pd.to_numeric(df.get("ACMG_INT", np.nan), errors="coerce"), np.nan)

    som_dp_col   = "DP_SOM" if "DP_SOM" in df.columns else "DP_UNI"
    som_vaf_col  = "VAF_SOM" if "VAF_SOM" in df.columns else "VAF_UNI"
    som_acmg_col = "ACMG_INT_SOM" if "ACMG_INT_SOM" in df.columns else "ACMG_INT"

    df["_dp_s"]   = np.where(is_som, pd.to_numeric(df.get(som_dp_col, np.nan), errors="coerce"), np.nan)
    df["_vaf_s"]  = np.where(is_som, pd.to_numeric(df.get(som_vaf_col, np.nan), errors="coerce"), np.nan)
    df["_acmg_s"] = np.where(is_som, pd.to_numeric(df.get(som_acmg_col, np.nan), errors="coerce"), np.nan)

    if "SOMATIC_REPEAT_COUNT" not in df.columns:
        df["SOMATIC_REPEAT_COUNT"] = 0

    fixed = {
        "HAS_GERM": ("_is_germ", "max"),
        "HAS_SOM":  ("_is_som",  "max"),

        "DP_GERM": ("_dp_g", "max"),
        "VAF_GERM": ("_vaf_g", "max"),
        "ACMG_INT_GERM": ("_acmg_g", "max"),

        "DP_SOM": ("_dp_s", "max"),
        "VAF_SOM": ("_vaf_s", "max"),
        "ACMG_INT_SOM": ("_acmg_s", "max"),

        "PAT_SIRIUS_GERM_SNV_PASS": ("SIRIUS_PASS_SNV_GERMLINE", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PAT_SIRIUS_SOM_SNV_PASS":  ("SIRIUS_PASS_SNV_SOMATIC",  lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PAT_SIRIUS_CNV_PASS":      ("SIRIUS_PASS_CNV",          lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PAT_SIRIUS_FLAG":          ("SIRIUS_FLAG_ROW",          lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),

        "SOMATIC_REPEAT_COUNT": ("SOMATIC_REPEAT_COUNT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),

        "PATHWAY_HIT": ("PATHWAY_HIT", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
        "PATH_DNA_REPAIR": ("PATH_DNA_REPAIR", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).max()),
    }

    join_cols = {
        "SOMATIC_SOURCE_SET",
        "FILTER", "ClinVar_CLNSIG", "ClinVar_CLNDN", "ClinVar_CLNDISDB",
        "COSMIC", "Hotspots_Kit", "ONCOGENIC", "HIGHEST_LEVEL",
        "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4",
    }

    drop_cols = {c for c in df.columns if c.startswith("_")}

    agg_map = build_keep_all_payload_aggs(
        df=df,
        key_cols=key,
        fixed_aggs=fixed,
        join_cols=join_cols,
        drop_cols=drop_cols,
    )

    out = df.groupby(key, dropna=False).agg(**agg_map).reset_index()
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")

    out["ORIGIN_PROFILE"] = np.select(
        [
            (out["HAS_GERM"] == 1) & (out["HAS_SOM"] == 1),
            (out["HAS_GERM"] == 1) & (out["HAS_SOM"] == 0),
            (out["HAS_GERM"] == 0) & (out["HAS_SOM"] == 1),
        ],
        ["both", "germline-only", "somatic-only"],
        default="unknown"
    )

    return sort_by_variant(out)




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
    for c in ["HAS_GERM", "HAS_SOM", "ACMG_INT_GERM", "ACMG_INT_SOM",
              "LOH_PARTIAL_PATHO_SOM",
              "PAT_SIRIUS_GERM_SNV_PASS", "PAT_SIRIUS_SOM_SNV_PASS", "PAT_SIRIUS_CNV_PASS"]:
        if c not in df.columns:
            df[c] = 0

    df["HAS_GERM"] = pd.to_numeric(df["HAS_GERM"], errors="coerce").fillna(0).astype(int)
    df["HAS_SOM"]  = pd.to_numeric(df["HAS_SOM"], errors="coerce").fillna(0).astype(int)
    df["ACMG_INT_GERM"] = pd.to_numeric(df["ACMG_INT_GERM"], errors="coerce")
    df["ACMG_INT_SOM"]  = pd.to_numeric(df["ACMG_INT_SOM"], errors="coerce")
    df["LOH_PARTIAL_PATHO_SOM"] = pd.to_numeric(df.get("LOH_PARTIAL_PATHO_SOM", 0), errors="coerce").fillna(0).astype(int)

    cancer_gene_cols = [
        "Cancer_Genes", "Cancer_Genes_Oncokb", "Cancer_Genes_OncoKB", "GENE_IN_ONCOKB",
        "Gene_Cancer_Status",
    ]
    has_cancer_gene = any_positive(df, cancer_gene_cols)

    df["predisposition_flag"] = (
        (df["HAS_GERM"] == 1) &
        (df["ACMG_INT_GERM"].isin([3, 4, 5])) &
        (has_cancer_gene == 1)
    ).astype(int)

    driver_cols = [
        "ONCOGENIC", "HIGHEST_LEVEL", "LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4",
        "IS-A-HOTSPOT", "IS-A-3D-HOTSPOT",
        "VARIANT_IN_ONCOKB", "MUTATION_EFFECT",
        "CIViC_Variant_clinical_significance", "CIViC_Region_clinical_significance",
        "COSMIC", "COSMIC_FATHMM", "Hotspots_Kit",
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
        ((df["LOH_PARTIAL_PATHO_SOM"] == 1) | (second_path_snv == 1))
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


def add_segregation_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SEGREGATION_SCORE" not in out.columns:
        out["SEGREGATION_SCORE"] = 0

    n_cancer = pd.to_numeric(out.get("N_CANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).astype(int)
    n_non    = pd.to_numeric(out.get("N_NONCANCER_WITH_VARIANT", 0), errors="coerce").fillna(0).astype(int)
    n_pat    = pd.to_numeric(out.get("N_PATIENTS_WITH_VARIANT", 0), errors="coerce").fillna(0).astype(int)

    out["SEGREGATION_SCORE"] = 0
    out.loc[(n_pat == 1), "SEGREGATION_SCORE"] = 1
    out.loc[(n_cancer >= 1), "SEGREGATION_SCORE"] = 2
    out.loc[(n_cancer == 1) & (n_non == 0), "SEGREGATION_SCORE"] = 3
    out.loc[(n_cancer >= 2) & (n_non == 0), "SEGREGATION_SCORE"] = 4
    return out

def add_family_evidence_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def num(col, default=0):
        return pd.to_numeric(out.get(col, default), errors="coerce").fillna(default)

    n_cancer = num("N_CANCER_WITH_VARIANT", 0).astype(int)
    n_non    = num("N_NONCANCER_WITH_VARIANT", 0).astype(int)
    n_both   = num("N_GERM_AND_SOM", 0).astype(int)
    sirius   = num("FAM_SIRIUS_FLAG", 0).astype(int)

    two_hit_strict = num("TWO_HIT_STRICT", 0).astype(int)
    two_hit_ev     = num("TWO_HIT_EVIDENCE", 0).astype(int)
    loh            = num("LOH_PARTIAL_PATHO_SOM", 0).astype(int)

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

def prepare_integrated_inputs_family(df_fam_var: pd.DataFrame) -> pd.DataFrame:
    out = df_fam_var.copy()

    def numcol(c, default=0):
        return pd.to_numeric(out.get(c, default), errors="coerce").fillna(default)

    if "COUNT_GE_2__LABELS" not in out.columns:
        out["COUNT_GE_2__LABELS"] = (numcol("FAM_COUNT_GE_2", 0) >= 1).astype(int)

    if "GERM_PASS" not in out.columns:
        out["GERM_PASS"] = (numcol("PAT_SIRIUS_GERM_SNV_PASS", 0) >= 1).astype(int)

    if "GERM_TIER" not in out.columns:
        ag = numcol("ACMG_INT_GERM", np.nan)
        out["GERM_TIER"] = np.nan
        out.loc[ag == 5, "GERM_TIER"] = 1
        out.loc[ag == 4, "GERM_TIER"] = 2

    if "TWO_HIT_STRICT" not in out.columns:
        out["TWO_HIT_STRICT"] = (numcol("two_hit_flag", 0) >= 1).astype(int)

    if "TWO_HIT_EVIDENCE" not in out.columns:
        out["TWO_HIT_EVIDENCE"] = ((numcol("LOH_PARTIAL_PATHO_SOM", 0) >= 1) | (numcol("PAT_SIRIUS_CNV_PASS", 0) >= 1)).astype(int)

    if "ORIGIN_PROFILE" not in out.columns:
        n_germ_only = numcol("N_GERM_ONLY", 0)
        n_germ_som  = numcol("N_GERM_AND_SOM", 0)
        op = pd.Series(["somatic-only"] * len(out), index=out.index)
        op[(n_germ_only > 0) & (n_germ_som == 0)] = "germline-only"
        op[(n_germ_som > 0)] = "both"
        out["ORIGIN_PROFILE"] = op

    for c in ["PATHWAY_HIT", "PATH_DNA_REPAIR"]:
        if c not in out.columns:
            out[c] = 0

    return out

def add_scientific_integrated_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def num(col, default=0):
        return pd.to_numeric(out.get(col, default), errors="coerce").fillna(default)

    germ_pass = (num("GERM_PASS", 0) >= 1).astype(int)
    germ_tier = num("GERM_TIER", np.nan)
    germ_tier_pts = pd.Series(0, index=out.index)
    germ_tier_pts[(germ_tier == 1)] = 2
    germ_tier_pts[(germ_tier == 2)] = 1
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

def collapse_patient_variants_to_family_variants_strategy_b(df_pat: pd.DataFrame) -> pd.DataFrame:
    if df_pat is None or df_pat.empty:
        return pd.DataFrame()

    df = df_pat.copy()

    key = ["FAMILY_ID", "CHROM", "POS", "REF", "ALT", "TX"]

    def uniq_list(series):
        vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
        vals = [v for v in vals if v not in {"", "nan", "None", ".", "NA"}]
        return ",".join(sorted(vals))

    def agg_block(g):
        needed = ["PATIENT_ID", "HAS_SOMATIC_PATIENT", "HAS_GERM", "HAS_SOM", "PAT_SIRIUS_FLAG"]
        for c in needed:
            if c not in g.columns:
                g[c] = 0
        g0 = g[needed].drop_duplicates()

        n_pat = int(g0["PATIENT_ID"].nunique())
        n_cancer = int(g0.loc[pd.to_numeric(g0["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int) == 1, "PATIENT_ID"].nunique())
        n_noncancer = int(g0.loc[pd.to_numeric(g0["HAS_SOMATIC_PATIENT"], errors="coerce").fillna(0).astype(int) == 0, "PATIENT_ID"].nunique())

        has_germ = pd.to_numeric(g0["HAS_GERM"], errors="coerce").fillna(0).astype(int)
        has_som  = pd.to_numeric(g0["HAS_SOM"], errors="coerce").fillna(0).astype(int)
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

    max_cols = [c for c in [
        "PAT_SIRIUS_GERM_SNV_PASS", "PAT_SIRIUS_SOM_SNV_PASS", "PAT_SIRIUS_CNV_PASS", "PAT_SIRIUS_FLAG",
        "predisposition_flag", "somatic_driver_flag", "two_hit_flag", "literature_flag",
        "patient_cancer_evidence_score", "patient_cancer_evidence_class",
        "LOH_PARTIAL_PATHO_SOM", "SOMATIC_REPEAT_COUNT",
        "PATHWAY_HIT", "PATH_DNA_REPAIR",
        "DP_GERM", "VAF_GERM", "ACMG_INT_GERM",
        "DP_SOM", "VAF_SOM", "ACMG_INT_SOM",
    ] if c in df.columns]

    payload_cols = [c for c in df.columns if c not in (key + ["PATIENT_ID"])]

    agg_map = {}
    for c in payload_cols:
        if c in max_cols:
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

    fam["FAM_COUNT_GE_2"] = (
        (pd.to_numeric(fam["FAM_SIRIUS_FLAG"], errors="coerce").fillna(0).astype(int) == 1) &
        (pd.to_numeric(fam["N_PATIENTS_WITH_VARIANT"], errors="coerce").fillna(0).astype(int) > 1)
    ).astype(int)

    return sort_by_variant(fam)


def build_global_unique_variants_strategy_b(df_fam_all: pd.DataFrame) -> pd.DataFrame:
    if df_fam_all is None or df_fam_all.empty:
        return pd.DataFrame()

    df = df_fam_all.copy()
    key = ["CHROM", "POS", "REF", "ALT", "TX"]

    def uniq_list(series):
        vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
        vals = [v for v in vals if v not in {"", "nan", "None", ".", "NA"}]
        return ",".join(sorted(vals))

    if "PATIENT_ID_LIST" not in df.columns:
        df["PATIENT_ID_LIST"] = ""

    lists = df.groupby(key, dropna=False).agg(
        FAMILY_ID_LIST_GLOBAL=("FAMILY_ID", uniq_list),
        PATIENT_ID_LIST_GLOBAL=("PATIENT_ID_LIST", uniq_list),
    ).reset_index()

    max_cols = [c for c in [
        "FAM_SIRIUS_FLAG", "label_of_interest", "FAM_COUNT_GE_2",
        "INTEGRATED_SCORE",
        "patient_cancer_evidence_class",
        "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT",
        "N_GERM_ONLY", "N_GERM_AND_SOM",
        "SOMATIC_REPEAT_COUNT", "LOH_PARTIAL_PATHO_SOM",
        "ACMG_INT_GERM", "ACMG_INT_SOM",
        "PATHWAY_HIT", "PATH_DNA_REPAIR",
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

    out2 = prepare_integrated_inputs_family(out)
    out2 = add_scientific_integrated_labels(out2)
    return sort_by_variant(out2)


def load_tsv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path) and os.path.isfile(path):
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception as e:
            print(f"[WARN] Could not read cached: {path} -> {type(e).__name__}: {e}")
    return None

def process_all_families_from_json(
    json_path: str,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    ensure_dir(out_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        families = json.load(f)

    if not isinstance(families, dict) or not families:
        raise ValueError("JSON must be {family_id: {patient_id: {...}}}")

    fam_long_all = []
    fam_pat_all  = []
    fam_var_all  = []

    for family_id, fam in families.items():
        if not isinstance(fam, dict):
            continue

        print(f"\n==== FAMILY: {family_id} ====")
        fam_out = os.path.join(out_dir, str(family_id))
        ensure_dir(fam_out)


        patient_longs = []
        for patient_id, cfg in fam.items():
            print(f"  -> patient: {patient_id}")

            p_long_path = os.path.join(fam_out, f"{family_id}__{patient_id}__LONG.tsv")
            df_p_long = load_tsv_if_exists(p_long_path)

            if df_p_long is not None and not df_p_long.empty:
                print(f"     [CACHE] {p_long_path}")
                has_somatic_patient = int(bool(cfg.get("somatic_runs")))
                if "HAS_SOMATIC_PATIENT" not in df_p_long.columns:
                    df_p_long["HAS_SOMATIC_PATIENT"] = has_somatic_patient
                else:
                    df_p_long["HAS_SOMATIC_PATIENT"] = has_somatic_patient
                patient_longs.append(df_p_long)
                continue

            df_p_long = build_patient_long_exome_only(str(family_id), str(patient_id), cfg)
            has_somatic_patient = int(bool(cfg.get("somatic_runs")))

            df_p_long["HAS_SOMATIC_PATIENT"] = has_somatic_patient

            if df_p_long is None or df_p_long.empty:
                print("     [WARN] no rows")
                continue

            safe_save(df_p_long, p_long_path, sep="\t")
            patient_longs.append(df_p_long)

        if not patient_longs:
            print("  [WARN] no patient data")
            continue

        fam_long_path = os.path.join(fam_out, f"{family_id}__ALL_PATIENTS__LONG.tsv")
        df_fam_long = load_tsv_if_exists(fam_long_path)

        if df_fam_long is None or df_fam_long.empty:
            print("  [BUILD] Rebuilding FAMILY_LONG")
            df_fam_long = sort_by_variant(pd.concat(patient_longs, ignore_index=True))
            safe_save(df_fam_long, fam_long_path, sep="\t")
        else:
            print(f"  [CACHE] {fam_long_path}")

        if "ORIGIN" not in df_fam_long.columns:
            raise ValueError("FAMILY_LONG must contain ORIGIN (germline/somatic).")

        if "TX" not in df_fam_long.columns:
            df_fam_long = standardize_tx(df_fam_long)

        if "ACMG_INT" not in df_fam_long.columns:
            df_fam_long = add_acmg_int_from_exome(df_fam_long)

        for c in ["PATHWAY_HIT", "PATH_DNA_REPAIR"]:
            if c not in df_fam_long.columns:
                df_fam_long[c] = 0

        if ("DP_UNI" not in df_fam_long.columns) or ("VAF_UNI" not in df_fam_long.columns):
            df_fam_long = add_unified_dp_vaf(df_fam_long)

        if "LOH_PARTIAL_PATHO" not in df_fam_long.columns:
            df_fam_long["LOH_PARTIAL_PATHO"] = 0

        df_fam_long = add_sirius_pass_flags_long(df_fam_long)


        pat_path = os.path.join(fam_out, f"{family_id}__PATIENT_VARIANTS.tsv")
        df_pat = load_tsv_if_exists(pat_path)

        if df_pat is None or df_pat.empty:
            print("  [BUILD] Rebuilding PATIENT_VARIANTS")
            df_pat = collapse_family_long_to_patient_variants_strategy_b(df_fam_long)


            df_pat = compute_patient_internal_flags(df_pat)
            df_pat = sort_by_variant(df_pat)
            safe_save(df_pat, pat_path, sep="\t")
        else:
            print(f"  [CACHE] {pat_path}")
            for c in ["HAS_GERM","HAS_SOM","HAS_SOMATIC_PATIENT","PAT_SIRIUS_FLAG","PAT_SIRIUS_GERM_SNV_PASS",
                      "PAT_SIRIUS_SOM_SNV_PASS","PAT_SIRIUS_CNV_PASS","LOH_PARTIAL_PATHO_SOM",
                      "predisposition_flag","somatic_driver_flag","two_hit_flag","literature_flag",
                      "patient_cancer_evidence_score","patient_cancer_evidence_class",
                      "ACMG_INT_GERM","ACMG_INT_SOM","SOMATIC_REPEAT_COUNT","PATHWAY_HIT","PATH_DNA_REPAIR"]:
                if c in df_pat.columns:
                    df_pat[c] = pd.to_numeric(df_pat[c], errors="coerce")
            df_pat = sort_by_variant(df_pat)


        df_fam_var = collapse_patient_variants_to_family_variants_strategy_b(df_pat)
        df_fam_var = prepare_integrated_inputs_family(df_fam_var)
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

    df_global_unique = build_global_unique_variants_strategy_b(df_fam_all)
    df_global_unique = sort_by_variant(df_global_unique)

    return df_pat_all, df_fam_all, df_global_unique, df_long_all




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON: {family_id: {patient_id: {...}}}")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df_pat_all, df_fam_all, df_global_unique, df_long_all = process_all_families_from_json(
        json_path=args.config,
        out_dir=args.out_dir
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
    export_minimal_csv(df_global_unique,
                       os.path.join(args.out_dir, "ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS__LABELS.csv"),
                       global_csv_labels)

    del df_pat_all, df_fam_all, df_global_unique, df_long_all
    gc.collect()
    print("[DONE]")

if __name__ == "__main__":
    main()

