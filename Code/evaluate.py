from __future__ import annotations

import os
import re
import gc
import math
import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import umap
from logistic_regression imort *
from model import TCL_TDA_Model_RefAlt


JOIN = ["FAMILY_ID", "VARIANT_KEY"]
NA_LIKE = {".", "NA", "nan", "None", "", "NULL", "NaN"}


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("Downstream UMAP for z_seq, z_bio, z_fused_proxy")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seq_tsv", type=str, required=True)
    p.add_argument("--bio_tsv", type=str, required=True)
    p.add_argument("--onto_tsv", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--seq_model_name", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seq_max_len", type=int, default=512)

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--init_tau", type=float, default=0.07)
    p.add_argument("--learnable_tau", action="store_true")

    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_points_umap", type=int, default=20000)
    p.add_argument(
    "--force_recompute_embeddings",
    action="store_true",
    help="Recompute embeddings even if cached memmap files already exist."
)

    return p.parse_args()


# =============================================================================
# Utils
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv_usecols(path: str, want: List[str]) -> pd.DataFrame:
    cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    use = [c for c in want if c in cols]
    if not use:
        raise ValueError(f"{path}: none of requested columns exist")
    return pd.read_csv(path, sep="\t", usecols=use, dtype=str)


def safe_dedup(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(JOIN)
    after = len(df)
    if after < before:
        print(f"[DEDUP] {name}: {before} -> {after}", flush=True)
    return df


def infer_bio_cols_from_checkpoint(bio_tsv: str, ckpt_bio_cols: Optional[List[str]]) -> List[str]:
    all_cols = pd.read_csv(bio_tsv, sep="\t", nrows=0).columns.tolist()

    if ckpt_bio_cols is None:
        drop = {"FAMILY_ID", "VARIANT_KEY", "labels"}
        return [c for c in all_cols if c not in drop]

    missing = [c for c in ckpt_bio_cols if c not in all_cols]
    if missing:
        print(f"[WARN] {len(missing)} training bio columns are missing in downstream bio_tsv", flush=True)
        print(f"[WARN] Missing examples: {missing[:20]}", flush=True)

    return ckpt_bio_cols


def normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


# =============================================================================
# Ontology mapping
# =============================================================================

ONTOLOGY_CATEGORY_PATTERNS = {
    "immune response": [
        r"\bimmune\b", r"\bimmun", r"\binflamm", r"\bcytokine\b", r"\bchemokine\b",
        r"\bleukocyte\b", r"\blymphocyte\b", r"\bantigen\b", r"\binterferon\b",
        r"\binnate immune\b", r"\badaptive immune\b", r"\bhost defense\b",
    ],
    "signal transduction": [
        r"\bsignal transduction\b", r"\bsignaling\b", r"\bsignalling\b",
        r"\bkinase cascade\b", r"\bmapk\b", r"\bjak\b", r"\bstat\b", r"\bpi3k\b",
        r"\bakt\b", r"\bmtor\b", r"\bwnt\b", r"\bnotch\b", r"\btgf", r"\bsmad\b",
        r"\bnf[- ]?kappa", r"\bgpcr\b", r"\breceptor signaling\b", r"\breceptor signalling\b",
    ],
    "DNA cycle/cell cycle": [
        r"\bdna repair\b", r"\bdna replication\b", r"\bdna recombination\b",
        r"\bdna damage\b", r"\bcell cycle\b", r"\bmitotic\b", r"\bmitosis\b",
        r"\bmeiosis\b", r"\bcheckpoint\b", r"\bchromosome segregation\b",
        r"\bchromatin segregation\b", r"\bnuclear division\b", r"\bcell division\b",
    ],
    "Metabolic process": [
        r"\bmetabolic process\b", r"\bmetabolism\b", r"\bbiosynthetic\b",
        r"\bcatabolic\b", r"\bglycolysis\b", r"\boxidative phosphorylation\b",
        r"\brespiratory chain\b", r"\bmitochond", r"\blipid metabolic\b",
        r"\bamino acid metabolic\b", r"\bnucleotide metabolic\b", r"\bglucose metabolic\b",
    ],
    "development process": [
        r"\bdevelopment\b", r"\bdevelopmental\b", r"\bdifferentiation\b",
        r"\bmorphogenesis\b", r"\bembryo", r"\borgan development\b",
        r"\btissue development\b", r"\bcell fate\b", r"\bpattern specification\b",
    ],
    "cell adhesion and migration": [
        r"\badhesion\b", r"\bcell adhesion\b", r"\bcell[- ]cell adhesion\b",
        r"\bextracellular matrix\b", r"\becm\b", r"\bmigration\b",
        r"\bcell migration\b", r"\bmotility\b", r"\bchemotaxis\b",
        r"\binvasion\b", r"\bcell junction\b", r"\bfocal adhesion\b",
    ],
}


def ontology_text_from_row(row: pd.Series) -> str:
    parts = []
    for c in ["GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]:
        v = str(row.get(c, ""))
        if v not in NA_LIKE:
            parts.append(v)
    return " | ".join(parts).lower()


def assign_categories(text: str) -> List[str]:
    out = []
    for cat, patterns in ONTOLOGY_CATEGORY_PATTERNS.items():
        if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            out.append(cat)
    return out


def primary_category(cats: List[str]) -> str:
    return cats[0] if cats else "other / unassigned"


# =============================================================================
# Data
# =============================================================================

class DownstreamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bio_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.bio_cols = list(bio_cols)

        bio_df = self.df.reindex(columns=self.bio_cols, fill_value=0.0)
        self.bio_np = (
            bio_df.apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
            .values
        )

        self.labels = (
            pd.to_numeric(self.df.get("labels", 0), errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .values
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and math.isnan(x):
            return ""
        s = str(x)
        return "" if s == "nan" else s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        return {
            "seq_ref": self._safe_str(r.get("SEQ_REF", "")),
            "seq_alt": self._safe_str(r.get("SEQ_ALT", "")),
            "bio": torch.from_numpy(self.bio_np[idx]),
            "label": int(self.labels[idx]),
            "FAMILY_ID": self._safe_str(r.get("FAMILY_ID", "")),
            "VARIANT_KEY": self._safe_str(r.get("VARIANT_KEY", "")),
            "GO_BP": self._safe_str(r.get("GO_BP", "")),
            "GO_MF": self._safe_str(r.get("GO_MF", "")),
            "GO_CC": self._safe_str(r.get("GO_CC", "")),
            "KEGG": self._safe_str(r.get("KEGG", "")),
            "HPO": self._safe_str(r.get("HPO", "")),
        }


def build_merged_df(seq_tsv: str, bio_tsv: str, onto_tsv: Optional[str], bio_cols: List[str]) -> pd.DataFrame:
    df_seq = read_tsv_usecols(seq_tsv, JOIN + ["SEQ_REF", "SEQ_ALT"])
    df_bio = read_tsv_usecols(bio_tsv, JOIN + bio_cols + ["labels"])

    df_seq = safe_dedup(df_seq, "seq")
    df_bio = safe_dedup(df_bio, "bio")

    df = df_seq.merge(df_bio, on=JOIN, how="inner")
    print(f"[MERGE] seqxbio rows={len(df)} cols={df.shape[1]}", flush=True)

    if onto_tsv is not None:
        onto_cols = JOIN + ["GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]
        df_onto = read_tsv_usecols(onto_tsv, onto_cols)
        df_onto = safe_dedup(df_onto, "onto")
        df = df.merge(df_onto, on=JOIN, how="left")
        del df_onto
        gc.collect()
    else:
        for c in ["GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]:
            if c not in df.columns:
                df[c] = ""

    del df_seq, df_bio
    gc.collect()
    return df


def build_collate_fn(seq_tokenizer, seq_max_len: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seq_ref = [b["seq_ref"] for b in batch]
        seq_alt = [b["seq_alt"] for b in batch]

        ref_tok = seq_tokenizer(
            seq_ref,
            padding=True,
            truncation=True,
            max_length=seq_max_len,
            return_tensors="pt",
        )
        alt_tok = seq_tokenizer(
            seq_alt,
            padding=True,
            truncation=True,
            max_length=seq_max_len,
            return_tensors="pt",
        )

        return {
            "ref_tokens": ref_tok,
            "alt_tokens": alt_tok,
            "bio": torch.stack([b["bio"] for b in batch], dim=0),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "FAMILY_ID": [b["FAMILY_ID"] for b in batch],
            "VARIANT_KEY": [b["VARIANT_KEY"] for b in batch],
            "GO_BP": [b["GO_BP"] for b in batch],
            "GO_MF": [b["GO_MF"] for b in batch],
            "GO_CC": [b["GO_CC"] for b in batch],
            "KEGG": [b["KEGG"] for b in batch],
            "HPO": [b["HPO"] for b in batch],
        }

    return collate


def move_tokens_to_device(d: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in d.items()}


@torch.no_grad()
def extract_embeddings_memmap(model, loader, device, out_dir: str, embed_dim: int):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    n_total = len(loader.dataset)

    zseq_path = os.path.join(out_dir, "z_seq.float32.memmap")
    zbio_path = os.path.join(out_dir, "z_bio.float32.memmap")
    zfused_path = os.path.join(out_dir, "z_fused.float32.memmap")
    meta_path = os.path.join(out_dir, "metadata.tsv")

    Z_seq = np.memmap(zseq_path, dtype="float32", mode="w+", shape=(n_total, embed_dim))
    Z_bio = np.memmap(zbio_path, dtype="float32", mode="w+", shape=(n_total, embed_dim))
    Z_fused = np.memmap(zfused_path, dtype="float32", mode="w+", shape=(n_total, embed_dim))

    meta_chunks = []
    cursor = 0

    for step, batch in enumerate(loader):
        ref_tokens = move_tokens_to_device(batch["ref_tokens"], device)
        alt_tokens = move_tokens_to_device(batch["alt_tokens"], device)
        x_bio = batch["bio"].to(device, non_blocking=(device.type == "cuda")).float()

        z_seq = model.encode_variant(ref_tokens, alt_tokens)
        z_bio = model.encode_bio(x_bio)

        z_seq_np = z_seq.detach().cpu().numpy().astype(np.float32)
        z_bio_np = z_bio.detach().cpu().numpy().astype(np.float32)
        z_fused_np = normalize_np((z_seq_np + z_bio_np) / 2.0).astype(np.float32)

        bs = z_seq_np.shape[0]
        start = cursor
        end = cursor + bs

        Z_seq[start:end] = z_seq_np
        Z_bio[start:end] = z_bio_np
        Z_fused[start:end] = z_fused_np

        meta_batch = pd.DataFrame({
            "FAMILY_ID": batch["FAMILY_ID"],
            "VARIANT_KEY": batch["VARIANT_KEY"],
            "label": batch["label"].cpu().numpy(),
            "GO_BP": batch["GO_BP"],
            "GO_MF": batch["GO_MF"],
            "GO_CC": batch["GO_CC"],
            "KEGG": batch["KEGG"],
            "HPO": batch["HPO"],
        })
        meta_chunks.append(meta_batch)

        cursor = end

        if (step + 1) % 20 == 0:
            print(f"[EMBED] processed {cursor} variants", flush=True)

        del z_seq, z_bio, z_seq_np, z_bio_np, z_fused_np, meta_batch
        gc.collect()

    Z_seq.flush()
    Z_bio.flush()
    Z_fused.flush()

    meta_df = pd.concat(meta_chunks, axis=0, ignore_index=True)
    meta_df.to_csv(meta_path, sep="\t", index=False)

    info = {
        "n_total": int(n_total),
        "embed_dim": int(embed_dim),
        "zseq_path": zseq_path,
        "zbio_path": zbio_path,
        "zfused_path": zfused_path,
        "meta_path": meta_path,
    }

    print(f"[SAVED] metadata -> {meta_path}", flush=True)
    print(f"[SAVED] z_seq memmap -> {zseq_path}", flush=True)
    print(f"[SAVED] z_bio memmap -> {zbio_path}", flush=True)
    print(f"[SAVED] z_fused memmap -> {zfused_path}", flush=True)

    return info


def load_memmap_embeddings(info: Dict[str, Any]):
    n_total = int(info["n_total"])
    embed_dim = int(info["embed_dim"])

    Z_seq = np.memmap(
        info["zseq_path"],
        dtype="float32",
        mode="r",
        shape=(n_total, embed_dim),
    )
    Z_bio = np.memmap(
        info["zbio_path"],
        dtype="float32",
        mode="r",
        shape=(n_total, embed_dim),
    )
    Z_fused = np.memmap(
        info["zfused_path"],
        dtype="float32",
        mode="r",
        shape=(n_total, embed_dim),
    )
    meta_df = pd.read_csv(info["meta_path"], sep="\t")

    return meta_df, Z_seq, Z_bio, Z_fused


# =============================================================================
# UMAP + plots
# =============================================================================

def run_umap(x: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x)


def add_ontology_columns(df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    cats_json = []
    primary = []

    for _, row in df.iterrows():
        txt = ontology_text_from_row(row)
        cats = assign_categories(txt)
        texts.append(txt)
        cats_json.append(json.dumps(cats, ensure_ascii=False))
        primary.append(primary_category(cats))

    df = df.copy()
    df["ontology_text"] = texts
    df["ontology_categories"] = cats_json
    df["primary_ontology_category"] = primary

    ordered = [
        "immune response",
        "signal transduction",
        "DNA cycle/cell cycle",
        "Metabolic process",
        "development process",
        "cell adhesion and migration",
    ]
    for cat in ordered:
        df[f"is_{cat}"] = [int(cat in json.loads(x)) for x in df["ontology_categories"]]

    return df


def save_umap_global(df: pd.DataFrame, xcol: str, ycol: str, out_png: str, title: str):
    plt.figure(figsize=(9, 7))
    plt.scatter(df[xcol], df[ycol], s=8, alpha=0.7)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_umap_by_label(df: pd.DataFrame, xcol: str, ycol: str, out_png: str, title: str):
    plt.figure(figsize=(9, 7))
    uniq = sorted(df["label"].dropna().unique().tolist())
    for lab in uniq:
        sub = df[df["label"] == lab]
        plt.scatter(sub[xcol], sub[ycol], s=10, alpha=0.8, label=f"label={lab}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_umap_by_primary_ontology(df: pd.DataFrame, xcol: str, ycol: str, out_png: str, title: str):
    plt.figure(figsize=(10, 8))
    ordered = [
        "immune response",
        "signal transduction",
        "DNA cycle/cell cycle",
        "Metabolic process",
        "development process",
        "cell adhesion and migration",
        "other / unassigned",
    ]
    for cat in ordered:
        sub = df[df["primary_ontology_category"] == cat]
        if len(sub) == 0:
            continue
        plt.scatter(sub[xcol], sub[ycol], s=10, alpha=0.8, label=f"{cat} (n={len(sub)})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_one_vs_rest(df: pd.DataFrame, xcol: str, ycol: str, cat: str, out_png: str, title: str):
    mask = df[f"is_{cat}"].astype(bool)

    plt.figure(figsize=(9, 7))
    bg = df[~mask]
    fg = df[mask]

    plt.scatter(bg[xcol], bg[ycol], s=8, alpha=0.12, label="others")
    plt.scatter(fg[xcol], fg[ycol], s=16, alpha=0.95, label=f"{cat} (n={len(fg)})")

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_all_plots(df: pd.DataFrame, prefix: str, out_dir: str):
    xcol = f"{prefix}_umap1"
    ycol = f"{prefix}_umap2"

    save_umap_global(
        df, xcol, ycol,
        os.path.join(out_dir, f"{prefix}_umap_global.png"),
        f"UMAP of {prefix}"
    )

    save_umap_by_label(
        df, xcol, ycol,
        os.path.join(out_dir, f"{prefix}_umap_by_label.png"),
        f"UMAP of {prefix} colored by label"
    )

    save_umap_by_primary_ontology(
        df, xcol, ycol,
        os.path.join(out_dir, f"{prefix}_umap_by_primary_ontology.png"),
        f"UMAP of {prefix} colored by ontology"
    )

    ordered = [
        "immune response",
        "signal transduction",
        "DNA cycle/cell cycle",
        "Metabolic process",
        "development process",
        "cell adhesion and migration",
    ]
    for cat in ordered:
        safe_name = cat.replace(" ", "_").replace("/", "_").replace("-", "_")
        save_one_vs_rest(
            df, xcol, ycol, cat,
            os.path.join(out_dir, f"{prefix}_{safe_name}.png"),
            f"UMAP of {prefix}: {cat}"
        )





def embeddings_exist(out_dir: str) -> bool:
    zseq_path = os.path.join(out_dir, "z_seq.float32.memmap")
    zbio_path = os.path.join(out_dir, "z_bio.float32.memmap")
    zfused_path = os.path.join(out_dir, "z_fused.float32.memmap")

    exists = all(os.path.exists(p) for p in [zseq_path, zbio_path, zfused_path])

    print("[DEBUG] embeddings existence check:", flush=True)
    print(f"  z_seq   : {zseq_path} -> {os.path.exists(zseq_path)}", flush=True)
    print(f"  z_bio   : {zbio_path} -> {os.path.exists(zbio_path)}", flush=True)
    print(f"  z_fused : {zfused_path} -> {os.path.exists(zfused_path)}", flush=True)

    return exists

import itertools
import numpy as np
import torch
import torch.nn.functional as F


def _values_from_rule(rule, n_points=3, default_min=0.0, default_max=1.0):
    """
    Convert one rule into a list of representative numeric values.
    """
    # exact scalar
    if isinstance(rule, (int, float, np.integer, np.floating)):
        return [float(rule)]

    # tuple/list range
    if isinstance(rule, (tuple, list)) and len(rule) == 2:
        lo, hi = float(rule[0]), float(rule[1])
        return np.linspace(lo, hi, n_points).tolist()

    # dict
    if isinstance(rule, dict):
        if "value" in rule:
            return [float(rule["value"])]

        if "values" in rule:
            return [float(x) for x in rule["values"]]

        if "min" in rule and "max" in rule:
            lo, hi = float(rule["min"]), float(rule["max"])
            return np.linspace(lo, hi, n_points).tolist()

        if "max" in rule:
            hi = float(rule["max"])
            lo = float(rule.get("assume_min", default_min))
            return np.linspace(lo, hi, n_points).tolist()

        if "min" in rule:
            lo = float(rule["min"])
            hi = float(rule.get("assume_max", default_max))
            return np.linspace(lo, hi, n_points).tolist()

    raise ValueError(f"Unsupported rule format: {rule}")


def build_multiple_bio_prototypes(
    model,
    bio_feature_cols,
    conditions,
    device=None,
    fill_value=0.0,
    n_points=3,
    default_bounds=None,
    max_prototypes=64,
    normalize_output=True,
    return_raw=False,
):
    """
    Build multiple bio prototypes from interval-like conditions.

    Parameters
    ----------
    model : torch.nn.Module
        Must expose encode_bio(x_bio)
    bio_feature_cols : list[str]
        Ordered bio columns used by the model
    conditions : dict
        Example:
        {
            "CADD": {"max": 20},
            "REVEL": {"max": 0.5},
            "gnomAD_AF": {"min": 0.01, "max": 0.1},
            "Cancer_Genes": 1,
        }

    default_bounds : dict or None
        Optional per-column defaults for open intervals.
        Example:
        {
            "CADD": {"min": 0, "max": 40},
            "REVEL": {"min": 0, "max": 1},
            "gnomAD_AF": {"min": 0, "max": 0.5},
        }

    max_prototypes : int
        Safety limit to avoid combinatorial explosion

    return_raw : bool
        If True, also returns raw prototype vectors before encoding

    Returns
    -------
    z_protos : np.ndarray, shape (K, d)
        Encoded prototypes
    x_protos : np.ndarray, shape (K, p), optional
        Raw bio vectors before encoding
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if default_bounds is None:
        default_bounds = {}

    col_to_idx = {c: i for i, c in enumerate(bio_feature_cols)}

    # Build candidate values per constrained column
    constrained_cols = []
    candidate_values = []

    for col, rule in conditions.items():
        if col not in col_to_idx:
            raise ValueError(f"Column '{col}' not found in bio_feature_cols")

        bounds = default_bounds.get(col, {})
        vals = _values_from_rule(
            rule,
            n_points=n_points,
            default_min=bounds.get("min", 0.0),
            default_max=bounds.get("max", 1.0),
        )

        constrained_cols.append(col)
        candidate_values.append(vals)

    # Cartesian product across all constrained columns
    combos = list(itertools.product(*candidate_values))

    if len(combos) > max_prototypes:
        raise ValueError(
            f"Too many prototypes would be generated ({len(combos)} > {max_prototypes}). "
            f"Reduce n_points or number of interval-constrained columns."
        )

    x_protos = np.full((len(combos), len(bio_feature_cols)), fill_value, dtype=np.float32)

    for k, combo in enumerate(combos):
        for col, val in zip(constrained_cols, combo):
            j = col_to_idx[col]
            x_protos[k, j] = float(val)

    x_tensor = torch.tensor(x_protos, dtype=torch.float32, device=device)

    with torch.no_grad():
        z_protos = model.encode_bio(x_tensor)

    if normalize_output:
        z_protos = F.normalize(z_protos, dim=1)

    z_protos = z_protos.detach().cpu().numpy()

    if return_raw:
        return z_protos, x_protos
    return z_protos



# conditions = {
#     "CADD": {"max": 20},
#     "REVEL": {"max": 0.5},
#     "gnomAD_AF": {"min": 0.01, "max": 0.2},
#     "Cancer_Genes": 0,
# }

# default_bounds = {
#     "CADD": {"min": 0, "max": 40},
#     "REVEL": {"min": 0, "max": 1},
#     "gnomAD_AF": {"min": 0, "max": 0.5},
# }

# z_protos = build_multiple_bio_prototypes(
#     model=model,
#     bio_feature_cols=bio_feature_cols,
#     conditions=conditions,
#     n_points=3,
#     default_bounds=default_bounds,
# )

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BOOT] device={device}", flush=True)

    # ------------------------------------------------------------
    # Load checkpoint / infer config
    # ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {})
    ckpt_bio_cols = ckpt.get("bio_feature_cols", None)

    bio_cols = infer_bio_cols_from_checkpoint(args.bio_tsv, ckpt_bio_cols)

    # ------------------------------------------------------------
    # Build merged dataframe
    # ------------------------------------------------------------
    df = build_merged_df(
        seq_tsv=args.seq_tsv,
        bio_tsv=args.bio_tsv,
        onto_tsv=args.onto_tsv,
        bio_cols=bio_cols,
    )
    print(f"[DATA] merged rows={len(df)}", flush=True)

    # ------------------------------------------------------------
    # Build tokenizer / model
    # ------------------------------------------------------------
    seq_tokenizer = AutoTokenizer.from_pretrained(
        args.seq_model_name,
        trust_remote_code=True,
    )

    model = PolyClip(
        seq_model_name=args.seq_model_name,
        bio_in_dim=len(bio_cols),
        embed_dim=int(ckpt_cfg.get("embed_dim", args.embed_dim)),
        text_model_name=None,
        dropout=float(ckpt_cfg.get("dropout", args.dropout)),
        learnable_tau=bool(ckpt_cfg.get("learnable_tau", args.learnable_tau)),
        init_tau=float(ckpt_cfg.get("init_tau", args.init_tau)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print("[MODEL] checkpoint loaded", flush=True)

    # ------------------------------------------------------------
    # Dataset / loader
    # ------------------------------------------------------------
    ds = DownstreamDataset(df, bio_cols=bio_cols)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=build_collate_fn(seq_tokenizer, args.seq_max_len),
    )

    # ------------------------------------------------------------
    # Reuse existing embeddings if available
    # ------------------------------------------------------------
    info = resolve_existing_embedding_info(args.out_dir)
    info = embeddings_exist(args.out_dir)

    if info:
        print("[EMBED] existing memmap embeddings found, loading from disk", flush=True)
    else:
        print("[EMBED] no existing embeddings found, extracting now", flush=True)
        info = extract_embeddings_memmap(
            model=model,
            loader=loader,
            device=device,
            out_dir=args.out_dir,
            embed_dim=model.embed_dim,
        )

    df_emb, z_seq, z_bio, z_fused = load_memmap_embeddings(info)
    df_emb = add_ontology_columns(df_emb)

    print(f"[EMBED] df_emb rows     = {len(df_emb)}", flush=True)
    print(f"[EMBED] z_seq shape    = {z_seq.shape}", flush=True)
    print(f"[EMBED] z_bio shape    = {z_bio.shape}", flush=True)
    print(f"[EMBED] z_fused shape  = {z_fused.shape}", flush=True)

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    n_total = len(df_emb)
    if z_seq.shape[0] != n_total or z_bio.shape[0] != n_total or z_fused.shape[0] != n_total:
        raise ValueError(
            f"Embedding/data size mismatch: len(df_emb)={n_total}, "
            f"z_seq={z_seq.shape}, z_bio={z_bio.shape}, z_fused={z_fused.shape}"
        )

    # ------------------------------------------------------------
    # Optional subsampling for UMAP / plotting
    # ------------------------------------------------------------
    if n_total > args.max_points_umap:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, size=args.max_points_umap, replace=False))
        print(f"[INFO] UMAP subsampling: {n_total} -> {len(idx)} points", flush=True)

        df_plot = df_emb.iloc[idx].reset_index(drop=True)
        z_seq_plot = np.asarray(z_seq[idx], dtype=np.float32)
        z_bio_plot = np.asarray(z_bio[idx], dtype=np.float32)
        z_fused_plot = np.asarray(z_fused[idx], dtype=np.float32)
        used_subsample = True
    else:
        df_plot = df_emb.copy().reset_index(drop=True)
        z_seq_plot = np.asarray(z_seq, dtype=np.float32)
        z_bio_plot = np.asarray(z_bio, dtype=np.float32)
        z_fused_plot = np.asarray(z_fused, dtype=np.float32)
        used_subsample = False

    print(f"[UMAP] z_seq_plot shape   = {z_seq_plot.shape}", flush=True)
    print(f"[UMAP] z_bio_plot shape   = {z_bio_plot.shape}", flush=True)
    print(f"[UMAP] z_fused_plot shape = {z_fused_plot.shape}", flush=True)

    # ------------------------------------------------------------
    # Run UMAP on plotting subset
    # ------------------------------------------------------------
    xy_seq = run_umap(
        z_seq_plot,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    xy_bio = run_umap(
        z_bio_plot,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    xy_fused = run_umap(
        z_fused_plot,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )

    df_plot["z_seq_umap1"] = xy_seq[:, 0]
    df_plot["z_seq_umap2"] = xy_seq[:, 1]
    df_plot["z_bio_umap1"] = xy_bio[:, 0]
    df_plot["z_bio_umap2"] = xy_bio[:, 1]
    df_plot["z_fused_umap1"] = xy_fused[:, 0]
    df_plot["z_fused_umap2"] = xy_fused[:, 1]

    print(df[["z_seq_umap1", "z_seq_umap2"]].head())

    # ------------------------------------------------------------
    # Save UMAP table
    # ------------------------------------------------------------
    umap_name = (
        "downstream_embeddings_umap_subsample.tsv"
        if used_subsample
        else "downstream_embeddings_umap.tsv"
    )
    out_tsv = os.path.join(args.out_dir, umap_name)
    df_plot.to_csv(out_tsv, sep="\t", index=False)
    print(f"[SAVED] {out_tsv}", flush=True)

    # ------------------------------------------------------------
    # Save ontology counts
    # Choose df_emb for global counts, df_plot for plotted subset counts
    # ------------------------------------------------------------
    ontology_categories = [
        "immune response",
        "signal transduction",
        "DNA cycle/cell cycle",
        "Metabolic process",
        "development process",
        "cell adhesion and migration",
    ]

    counts = []
    for cat in ontology_categories:
        col = f"is_{cat}"
        if col not in df_emb.columns:
            print(f"[WARN] missing ontology column: {col}", flush=True)
            count_all = 0
            count_plot = 0
        else:
            count_all = int(df_emb[col].sum())
            count_plot = int(df_plot[col].sum())

        counts.append(
            {
                "category": cat,
                "count_all_embeddings": count_all,
                "count_plot_subset": count_plot,
            }
        )

    counts_tsv = os.path.join(args.out_dir, "ontology_category_counts.tsv")
    pd.DataFrame(counts).to_csv(counts_tsv, sep="\t", index=False)
    print(f"[SAVED] {counts_tsv}", flush=True)

    # ------------------------------------------------------------
    # Generate plots from df_plot (the only df containing UMAP columns)
    # ------------------------------------------------------------
    save_all_plots(df_plot, prefix="z_seq", out_dir=args.out_dir)
    save_all_plots(df_plot, prefix="z_bio", out_dir=args.out_dir)
    save_all_plots(df_plot, prefix="z_fused", out_dir=args.out_dir)

    print("[DONE] All plots generated.", flush=True)

# def main():
#     args = parse_args()
#     ensure_dir(args.out_dir)
#     seed_everything(args.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[BOOT] device={device}", flush=True)

#     ckpt = torch.load(args.checkpoint, map_location="cpu")
#     ckpt_cfg = ckpt.get("cfg", {})
#     ckpt_bio_cols = ckpt.get("bio_feature_cols", None)

#     bio_cols = infer_bio_cols_from_checkpoint(args.bio_tsv, ckpt_bio_cols)

#     df = build_merged_df(
#         seq_tsv=args.seq_tsv,
#         bio_tsv=args.bio_tsv,
#         onto_tsv=args.onto_tsv,
#         bio_cols=bio_cols,
#     )

#     seq_tokenizer = AutoTokenizer.from_pretrained(
#         args.seq_model_name,
#         trust_remote_code=True,
#     )

#     model = PolyClip(
#         seq_model_name=args.seq_model_name,
#         bio_in_dim=len(bio_cols),
#         embed_dim=int(ckpt_cfg.get("embed_dim", args.embed_dim)),
#         text_model_name=None,
#         dropout=float(ckpt_cfg.get("dropout", args.dropout)),
#         learnable_tau=bool(ckpt_cfg.get("learnable_tau", args.learnable_tau)),
#         init_tau=float(ckpt_cfg.get("init_tau", args.init_tau)),
#     ).to(device)

#     model.load_state_dict(ckpt["model_state"], strict=False)
#     model.eval()
#     print("[MODEL] checkpoint loaded", flush=True)

#     ds = DownstreamDataset(df, bio_cols=bio_cols)
#     loader = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=(device.type == "cuda"),
#         collate_fn=build_collate_fn(seq_tokenizer, args.seq_max_len),
#     )

#     # df_emb = extract_embeddings(model, loader, device=device)
#     # df_emb = add_ontology_columns(df_emb)

#     # zseq_cols = [c for c in df_emb.columns if c.startswith("zseq_")]
#     # zbio_cols = [c for c in df_emb.columns if c.startswith("zbio_")]
#     # zfused_cols = [c for c in df_emb.columns if c.startswith("zfused_")]

#     # z_seq = df_emb[zseq_cols].values.astype(np.float32)
#     # z_bio = df_emb[zbio_cols].values.astype(np.float32)
#     # z_fused = df_emb[zfused_cols].values.astype(np.float32)

#     info = extract_embeddings_memmap(
#         model=model,
#         loader=loader,
#         device=device,
#         out_dir=args.out_dir,
#         embed_dim=model.embed_dim,
#     )

#     df_emb, z_seq, z_bio, z_fused = load_memmap_embeddings(info)
#     df_emb = add_ontology_columns(df_emb)

#     print(f"[UMAP] z_seq shape   = {z_seq.shape}", flush=True)
#     print(f"[UMAP] z_bio shape   = {z_bio.shape}", flush=True)
#     print(f"[UMAP] z_fused shape = {z_fused.shape}", flush=True)


#     n_total = len(df_emb)
#     if n_total > args.max_points_umap:
#         rng = np.random.default_rng(args.seed)
#         idx = np.sort(rng.choice(n_total, size=args.max_points_umap, replace=False))
#         print(f"[INFO] UMAP subsampling: {n_total} -> {len(idx)} points", flush=True)

#         df_plot = df_emb.iloc[idx].reset_index(drop=True)
#         z_seq_plot = np.asarray(z_seq[idx], dtype=np.float32)
#         z_bio_plot = np.asarray(z_bio[idx], dtype=np.float32)
#         z_fused_plot = np.asarray(z_fused[idx], dtype=np.float32)
#     else:
#         df_plot = df_emb.copy()
#         z_seq_plot = np.asarray(z_seq, dtype=np.float32)
#         z_bio_plot = np.asarray(z_bio, dtype=np.float32)
#         z_fused_plot = np.asarray(z_fused, dtype=np.float32)


#     # xy_seq = run_umap(
#     #     z_seq,
#     #     n_neighbors=args.umap_n_neighbors,
#     #     min_dist=args.umap_min_dist,
#     #     metric=args.umap_metric,
#     #     seed=args.seed,
#     # )
#     # xy_bio = run_umap(
#     #     z_bio,
#     #     n_neighbors=args.umap_n_neighbors,
#     #     min_dist=args.umap_min_dist,
#     #     metric=args.umap_metric,
#     #     seed=args.seed,
#     # )
#     # xy_fused = run_umap(
#     #     z_fused,
#     #     n_neighbors=args.umap_n_neighbors,
#     #     min_dist=args.umap_min_dist,
#     #     metric=args.umap_metric,
#     #     seed=args.seed,
#     # )

#     # df_emb["z_seq_umap1"] = xy_seq[:, 0]
#     # df_emb["z_seq_umap2"] = xy_seq[:, 1]
#     # df_emb["z_bio_umap1"] = xy_bio[:, 0]
#     # df_emb["z_bio_umap2"] = xy_bio[:, 1]
#     # df_emb["z_fused_umap1"] = xy_fused[:, 0]
#     # df_emb["z_fused_umap2"] = xy_fused[:, 1]

#     # out_tsv = os.path.join(args.out_dir, "downstream_embeddings_umap.tsv")
#     # df_emb.to_csv(out_tsv, sep="\t", index=False)
#     # print(f"[SAVED] {out_tsv}", flush=True)

#     print(f"[UMAP] z_seq shape   = {z_seq_plot.shape}", flush=True)
#     print(f"[UMAP] z_bio shape   = {z_bio_plot.shape}", flush=True)
#     print(f"[UMAP] z_fused shape = {z_fused_plot.shape}", flush=True)

#     xy_seq = run_umap(
#         z_seq_plot,
#         n_neighbors=args.umap_n_neighbors,
#         min_dist=args.umap_min_dist,
#         metric=args.umap_metric,
#         seed=args.seed,
#     )
#     xy_bio = run_umap(
#         z_bio_plot,
#         n_neighbors=args.umap_n_neighbors,
#         min_dist=args.umap_min_dist,
#         metric=args.umap_metric,
#         seed=args.seed,
#     )
#     xy_fused = run_umap(
#         z_fused_plot,
#         n_neighbors=args.umap_n_neighbors,
#         min_dist=args.umap_min_dist,
#         metric=args.umap_metric,
#         seed=args.seed,
#     )

#     df_plot["z_seq_umap1"] = xy_seq[:, 0]
#     df_plot["z_seq_umap2"] = xy_seq[:, 1]
#     df_plot["z_bio_umap1"] = xy_bio[:, 0]
#     df_plot["z_bio_umap2"] = xy_bio[:, 1]
#     df_plot["z_fused_umap1"] = xy_fused[:, 0]
#     df_plot["z_fused_umap2"] = xy_fused[:, 1]

#     out_tsv = os.path.join(args.out_dir, "downstream_embeddings_umap.tsv")
#     df_plot.to_csv(out_tsv, sep="\t", index=False)
#     print(f"[SAVED] {out_tsv}", flush=True)

#     counts = []
#     for cat in [
#         "immune response",
#         "signal transduction",
#         "DNA cycle/cell cycle",
#         "Metabolic process",
#         "development process",
#         "cell adhesion and migration",
#     ]:
#         counts.append({"category": cat, "count": int(df_emb[f'is_{cat}'].sum())})

#     pd.DataFrame(counts).to_csv(
#         os.path.join(args.out_dir, "ontology_category_counts.tsv"),
#         sep="\t",
#         index=False,
#     )

#     save_all_plots(df_emb, prefix="z_seq", out_dir=args.out_dir)
#     save_all_plots(df_emb, prefix="z_bio", out_dir=args.out_dir)
#     save_all_plots(df_emb, prefix="z_fused", out_dir=args.out_dir)

#     print("[DONE] All plots generated.", flush=True)


if __name__ == "__main__":
    main()