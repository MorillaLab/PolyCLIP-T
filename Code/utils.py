from libs import *
import re


DNA_ALPHABET = {"A", "C", "G", "T", "N"}

JOIN_COLS = ["FAMILY_ID", "VARIANT_KEY"]


BIO_DROP_COLS = {
    "FAMILY_ID",
    "VARIANT_KEY",
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "labels",
    "label_of_interest",
    "label_clinvar",
    "label_acmg",



    "ACMG_INT",
    "ACMG_INT_GERM",
    "ACMG_INT_SOM",
    "Prediction_ACMG_tapes",
    "Probability_Path",
    "INTEGRATED_SCORE",
    "INTEGRATED_CLASS",
    "FAMILY_EVIDENCE_SCORE",
    "SEGREGATION_SCORE",
    "patient_cancer_evidence_score",
    "predisposition_flag",
    "somatic_driver_flag",
    "two_hit_flag",
    "literature_flag",
    "TWO_HIT_STRICT",
    "FAM_COUNT_GE_2",
    "FAM_SIRIUS_FLAG",
    "PAT_SIRIUS_FLAG",
    "SIRIUS_PASS_POPAF",
    "SIRIUS_PASS_NONSYN",
    "SIRIUS_PASS_SNV_GERMLINE",
    "SIRIUS_PASS_SNV_SOMATIC",
    "SIRIUS_PASS_CNV",
    "HGNC_Name",
    "Gene_name",
}

STEP1_EXTRA_DROP = {
    "ACMG_INT",
    "ACMG_INT_GERM",
    "ACMG_INT_SOM",
    "Prediction_ACMG_tapes",
    "Probability_Path",
    "INTEGRATED_SCORE",
    "INTEGRATED_CLASS",
    "FAMILY_EVIDENCE_SCORE",
    "SEGREGATION_SCORE",
    "patient_cancer_evidence_score",
    "predisposition_flag",
    "somatic_driver_flag",
    "two_hit_flag",
    "literature_flag",
    "TWO_HIT_STRICT",
    "FAM_COUNT_GE_2",
    "FAM_SIRIUS_FLAG",
    "PAT_SIRIUS_FLAG",
    "SIRIUS_PASS_POPAF",
    "SIRIUS_PASS_NONSYN",
    "SIRIUS_PASS_SNV_GERMLINE",
    "SIRIUS_PASS_SNV_SOMATIC",
    "SIRIUS_PASS_CNV",
    "HGNC_Name",
}

ONTO_COLS = ["GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"]

ONCO_COLS = [
    "CLINICAL_TEXT",
    "TXT_FUNC",
    "TXT_DISEASE",
    "TXT_CLNSIG",
    "TXT_CLNDN",
    "TXT_OMIM",
    "TXT_DISGENET",
    "TXT_GWAS",
    "TXT_COSMIC",
    "TXT_CIVIC",
    "TXT_CGI",
    "TXT_ONCOKB",
    "GENE_NAME",
    "CANCER_GENE",
    "ONCOGENICITY",
]

META_COLS = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "GENE_NAME",
    "Gene_Name",
    "FAMILY_ID",
    "VARIANT_KEY",
]


class VariantsDatasetRefAlt(Dataset):
    def __init__(
        self,
        seq_refs: Optional[List[str]],
        seq_alts: Optional[List[str]],
        bio_feats: np.ndarray,
        labels: np.ndarray,
        texts: Optional[List[str]] = None,
        ref_embs: Optional[np.ndarray] = None,
        alt_embs: Optional[np.ndarray] = None,
        text_embs: Optional[np.ndarray] = None,
        onto_data: Optional[Dict[str, List[str]]] = None,
        onco_data: Optional[Dict[str, List[str]]] = None,
        meta_data: Optional[Dict[str, List[str]]] = None,
        bio_feature_cols: Optional[List[str]] = None,
    ):
        n = int(np.asarray(bio_feats).shape[0])

        if labels is None or len(labels) != n:
            raise ValueError("labels length mismatch")

        if seq_refs is not None and len(seq_refs) != n:
            raise ValueError("seq_refs length mismatch")

        if seq_alts is not None and len(seq_alts) != n:
            raise ValueError("seq_alts length mismatch")

        if texts is not None and len(texts) != n:
            raise ValueError("texts length mismatch")

        if ref_embs is not None and len(ref_embs) != n:
            raise ValueError("ref_embs length mismatch")

        if alt_embs is not None and len(alt_embs) != n:
            raise ValueError("alt_embs length mismatch")

        if text_embs is not None and len(text_embs) != n:
            raise ValueError("text_embs length mismatch")

        self.seq_refs = list(seq_refs) if seq_refs is not None else None
        self.seq_alts = list(seq_alts) if seq_alts is not None else None
        self.texts = list(texts) if texts is not None else None

        self.bio = torch.tensor(np.asarray(bio_feats), dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

        self.ref_embs = (
            torch.tensor(np.asarray(ref_embs), dtype=torch.float32)
            if ref_embs is not None else None
        )
        self.alt_embs = (
            torch.tensor(np.asarray(alt_embs), dtype=torch.float32)
            if alt_embs is not None else None
        )
        self.text_embs = (
            torch.tensor(np.asarray(text_embs), dtype=torch.float32)
            if text_embs is not None else None
        )

        self.bio_feature_cols = list(bio_feature_cols or [])
        self.bio_in_dim = int(self.bio.shape[1]) if self.bio.ndim == 2 else 0

        self.onto_data = onto_data or {}
        self.onco_data = onco_data or {}
        self.meta_data = meta_data or {}

        for name, d in [
            ("onto_data", self.onto_data),
            ("onco_data", self.onco_data),
            ("meta_data", self.meta_data),
        ]:
            for k, v in d.items():
                if len(v) != n:
                    raise ValueError(f"{name}[{k}] length mismatch")

    def __len__(self):
        return self.bio.shape[0]

    def __getitem__(self, idx):
        item = {
            "idx": idx,
            "bio": self.bio[idx],
            "label": self.labels[idx],
        }

        if self.seq_refs is not None:
            item["seq_ref"] = self.seq_refs[idx]

        if self.seq_alts is not None:
            item["seq_alt"] = self.seq_alts[idx]

        if self.texts is not None:
            item["text"] = self.texts[idx]

        if self.ref_embs is not None:
            item["ref_emb"] = self.ref_embs[idx]

        if self.alt_embs is not None:
            item["alt_emb"] = self.alt_embs[idx]

        if self.text_embs is not None:
            item["text_emb"] = self.text_embs[idx]

        if self.onto_data:
            item["ontology"] = {k: self.onto_data[k][idx] for k in self.onto_data}

        if self.onco_data:
            item["oncology"] = {k: self.onco_data[k][idx] for k in self.onco_data}

        if self.meta_data:
            item["meta"] = {k: self.meta_data[k][idx] for k in self.meta_data}

        return item


def _safe_text_col(df: pd.DataFrame, col: str) -> List[str]:
    if col in df.columns:
        return df[col].fillna("").astype(str).tolist()
    return [""] * len(df)


def _safe_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return (
        df.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _normalize_text_value(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "na", "."}:
        return ""
    return s


def build_texts_from_df(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
    prefix_with_colname: bool = True,
) -> List[str]:
    if text_cols is None:
        text_cols = [c for c in ONCO_COLS if c in df.columns]

    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in text_cols:
            val = _normalize_text_value(row.get(c, ""))
            if not val:
                continue
            if prefix_with_colname:
                parts.append(f"{c}: {val}")
            else:
                parts.append(val)
        texts.append(" | ".join(parts))
    return texts


def clean_sequence(seq: str) -> str:
    if seq is None:
        return ""

    s = str(seq).upper().strip()
    s = re.sub(r"\s+", "", s)

    if s == "":
        return ""

    cleaned = "".join(base if base in DNA_ALPHABET else "N" for base in s)
    return cleaned


def validate_sequences(df, column_name="REF_with_flank", max_len=512):
    bad_idx = []
    for i, seq in enumerate(df[column_name]):
        if not isinstance(seq, str):
            bad_idx.append(i)
            continue

        seq_clean = seq.strip().upper()

        if len(seq_clean) == 0:
            print(f"Row {i}: empty sequence {seq}")
            bad_idx.append(i)
        elif len(seq_clean) > max_len:
            print(f"Row {i}: sequence too long ({len(seq_clean)} > {max_len})")
            bad_idx.append(i)
        elif " " in seq_clean:
            print(f"Row {i}: contains spaces")
            bad_idx.append(i)
        elif not all(c in "ATCGN" for c in seq_clean):
            print(f"Row {i}: contains invalid characters -> {set(seq_clean) - set('ATCGN')}")
            bad_idx.append(i)

    return bad_idx


def _clean_seq_series(s: pd.Series) -> List[str]:
    return s.fillna("").astype(str).map(clean_sequence).tolist()


def _read_header(path: str) -> List[str]:
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def _read_tsv_usecols(path: str, wanted: List[str]) -> pd.DataFrame:
    cols = _read_header(path)
    usecols = [c for c in wanted if c in cols]

    if not usecols:
        raise ValueError(f"{path}: none of requested columns were found")

    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        low_memory=False,
    )


def _dedup_on_join(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=JOIN_COLS).reset_index(drop=True)
    after = len(df)

    if after != before:
        print(f"[DEDUP] {name}: {before} -> {after}")

    return df


def _infer_bio_feature_cols(
    bio_tsv: str,
    label_col: str = "labels",
) -> List[str]:
    cols = _read_header(bio_tsv)
    drop = set(BIO_DROP_COLS)
    drop.add(label_col)
    return [c for c in cols if c not in drop]


def _build_merged_dataframe(
    seq_tsv: str,
    bio_tsv: str,
    onto_tsv: Optional[str] = None,
    text_tsv: Optional[str] = None,
    textfields_tsv: Optional[str] = None,
    label_col: str = "labels",
) -> pd.DataFrame:
    seq_cols = JOIN_COLS + [
        "SEQ_REF",
        "SEQ_ALT",
        "REF_with_flank",
        "ALT_with_flank",
        "MOTHER_with_flank",
        "FATHER_with_flank",
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "allele_1",
        "allele_2",
        "GENE_NAME",
    ]

    df_seq = _read_tsv_usecols(seq_tsv, seq_cols)
    df_seq = _dedup_on_join(df_seq, "seq")

    bio_feature_cols = _infer_bio_feature_cols(bio_tsv, label_col=label_col)
    bio_cols = JOIN_COLS + bio_feature_cols

    bio_header = _read_header(bio_tsv)
    if label_col in bio_header:
        bio_cols.append(label_col)
    if "labels" in bio_header:
        bio_cols.append("labels")

    df_bio = _read_tsv_usecols(bio_tsv, bio_cols)
    df_bio = _dedup_on_join(df_bio, "bio")

    df = df_seq.merge(df_bio, on=JOIN_COLS, how="inner")
    print(f"[MERGE] seq x bio -> rows={len(df)} cols={df.shape[1]}")

    if onto_tsv is not None:
        df_onto = _read_tsv_usecols(onto_tsv, JOIN_COLS + ONTO_COLS)
        df_onto = _dedup_on_join(df_onto, "ontology")
        df = df.merge(df_onto, on=JOIN_COLS, how="left")

    if text_tsv is not None:
        text_header = _read_header(text_tsv)
        text_cols = [c for c in ONCO_COLS if c in text_header]

        if text_cols:
            df_text = _read_tsv_usecols(text_tsv, JOIN_COLS + text_cols)
            df_text = _dedup_on_join(df_text, "text")
            df = df.merge(df_text, on=JOIN_COLS, how="left")

    if textfields_tsv is not None:
        tf_header = _read_header(textfields_tsv)
        tf_cols = [c for c in ONCO_COLS if c in tf_header]

        if tf_cols:
            df_tf = _read_tsv_usecols(textfields_tsv, JOIN_COLS + tf_cols)
            df_tf = _dedup_on_join(df_tf, "textfields")

            keep_cols = [c for c in df_tf.columns if c not in df.columns or c in JOIN_COLS]
            df_tf = df_tf[keep_cols]
            df = df.merge(df_tf, on=JOIN_COLS, how="left")

    return df


def _make_dataset_cache_key(
    seq_tsv: str,
    bio_tsv: str,
    onto_tsv: Optional[str],
    text_tsv: Optional[str],
    textfields_tsv: Optional[str],
    labels_path: Optional[str],
    label_col: str,
    include_ontology: bool,
    include_oncology: bool,
    use_text: bool = False,
    seq_offline_embeddings: bool = False,
    text_offline_embeddings: bool = False,
    dna_model_name: Optional[str] = None,
    text_model_name: Optional[str] = None,
    precomputed_ref_path: Optional[str] = None,
    precomputed_alt_path: Optional[str] = None,
    precomputed_text_path: Optional[str] = None,
) -> str:
    payload = {
        "seq_tsv": os.path.abspath(os.path.expanduser(seq_tsv)) if seq_tsv else None,
        "bio_tsv": os.path.abspath(os.path.expanduser(bio_tsv)) if bio_tsv else None,
        "onto_tsv": os.path.abspath(os.path.expanduser(onto_tsv)) if onto_tsv else None,
        "text_tsv": os.path.abspath(os.path.expanduser(text_tsv)) if text_tsv else None,
        "textfields_tsv": os.path.abspath(os.path.expanduser(textfields_tsv)) if textfields_tsv else None,
        "labels_path": os.path.abspath(os.path.expanduser(labels_path)) if labels_path else None,
        "label_col": label_col,
        "include_ontology": include_ontology,
        "include_oncology": include_oncology,
        "use_text": use_text,
        "seq_offline_embeddings": seq_offline_embeddings,
        "text_offline_embeddings": text_offline_embeddings,
        "dna_model_name": dna_model_name,
        "text_model_name": text_model_name,
        "precomputed_ref_path": os.path.abspath(os.path.expanduser(precomputed_ref_path)) if precomputed_ref_path else None,
        "precomputed_alt_path": os.path.abspath(os.path.expanduser(precomputed_alt_path)) if precomputed_alt_path else None,
        "precomputed_text_path": os.path.abspath(os.path.expanduser(precomputed_text_path)) if precomputed_text_path else None,
    }

    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _save_dataset_cache(cache_path: str, dataset) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    payload = {
        "seq_refs": list(dataset.seq_refs) if dataset.seq_refs is not None else None,
        "seq_alts": list(dataset.seq_alts) if dataset.seq_alts is not None else None,
        "texts": list(dataset.texts) if dataset.texts is not None else None,
        "bio_feats": dataset.bio.cpu().numpy(),
        "labels": dataset.labels.cpu().numpy(),
        "ref_embs": dataset.ref_embs.cpu().numpy() if dataset.ref_embs is not None else None,
        "alt_embs": dataset.alt_embs.cpu().numpy() if dataset.alt_embs is not None else None,
        "text_embs": dataset.text_embs.cpu().numpy() if dataset.text_embs is not None else None,
        "onto_data": dataset.onto_data,
        "onco_data": dataset.onco_data,
        "meta_data": dataset.meta_data,
        "bio_feature_cols": getattr(dataset, "bio_feature_cols", None),
    }

    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_dataset_cache(cache_path: str):
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    cached_dataset = VariantsDatasetRefAlt(
        seq_refs=payload.get("seq_refs", None),
        seq_alts=payload.get("seq_alts", None),
        bio_feats=payload["bio_feats"],
        labels=payload["labels"],
        texts=payload.get("texts", None),
        ref_embs=payload.get("ref_embs", None),
        alt_embs=payload.get("alt_embs", None),
        text_embs=payload.get("text_embs", None),
        onto_data=payload.get("onto_data", {}),
        onco_data=payload.get("onco_data", {}),
        meta_data=payload.get("meta_data", {}),
        bio_feature_cols=payload.get("bio_feature_cols", None),
    )

    return cached_dataset


def _truncate_variant_dataset(dataset, max_bio_line=None):
    if max_bio_line is None:
        return dataset

    max_bio_line = int(max_bio_line)
    if max_bio_line <= 0:
        raise ValueError("max_bio_line must be a positive integer")

    n = len(dataset)
    if max_bio_line >= n:
        return dataset

    truncated_dataset = VariantsDatasetRefAlt(
        seq_refs=dataset.seq_refs[:max_bio_line] if dataset.seq_refs is not None else None,
        seq_alts=dataset.seq_alts[:max_bio_line] if dataset.seq_alts is not None else None,
        texts=dataset.texts[:max_bio_line] if dataset.texts is not None else None,
        bio_feats=dataset.bio[:max_bio_line].cpu().numpy(),
        labels=dataset.labels[:max_bio_line].cpu().numpy(),
        ref_embs=dataset.ref_embs[:max_bio_line].cpu().numpy() if dataset.ref_embs is not None else None,
        alt_embs=dataset.alt_embs[:max_bio_line].cpu().numpy() if dataset.alt_embs is not None else None,
        text_embs=dataset.text_embs[:max_bio_line].cpu().numpy() if dataset.text_embs is not None else None,
        onto_data={k: v[:max_bio_line] for k, v in getattr(dataset, "onto_data", {}).items()},
        onco_data={k: v[:max_bio_line] for k, v in getattr(dataset, "onco_data", {}).items()},
        meta_data={k: v[:max_bio_line] for k, v in getattr(dataset, "meta_data", {}).items()},
        bio_feature_cols=getattr(dataset, "bio_feature_cols", None),
    )

    return truncated_dataset


EPS = 1e-8

def _to_numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def preprocess_bio_dataframe(
    df: pd.DataFrame,
    cont_cols,
    ordinal_cols,
    binary_cols,
    unit_cols,
    cont_stats=None,
    ordinal_max_map=None,
    corr_threshold=None,
):
    """
    Preprocess a dataframe of bio features into a flat float32 matrix-compatible dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing bio columns.
    cont_cols : list[str]
        Continuous columns to z-score.
    ordinal_cols : list[str]
        Ordinal columns, e.g. 0..5 -> scaled to [0,1].
    binary_cols : list[str]
        Binary columns kept as 0/1.
    unit_cols : list[str]
        Columns expected in [0,1], clamped to [0,1].
    cont_stats : dict or None
        If None, fit mean/std on this df. Otherwise reuse given stats.
    ordinal_max_map : dict or None
        Per-column maximum for ordinal scaling. Default 5.
    corr_threshold : float or None
        If provided and cont_stats is None, drop highly correlated continuous cols.

    Returns
    -------
    out_df : pd.DataFrame
        Processed dataframe ready for numeric conversion.
    fitted_stats : dict
        Contains means/stds and dropped continuous columns.
    """
    ordinal_max_map = ordinal_max_map or {}

    fitted = {
        "cont_stats": {},
        "drop_cont_cols": [],
    } if cont_stats is None else cont_stats

    out = pd.DataFrame(index=df.index)

    # ---------------------------
    # Fit / apply continuous cols
    # ---------------------------
    if cont_stats is None:
        tmp_cont = pd.DataFrame(index=df.index)

        for c in cont_cols:
            x = _to_numeric_series(df, c)
            mu = float(x.mean()) if x.notna().any() else 0.0
            sd = float(x.std()) if x.notna().any() else 1.0
            if not np.isfinite(sd) or sd < EPS:
                sd = 1.0
            fitted["cont_stats"][c] = {"mean": mu, "std": sd}
            tmp_cont[c] = x.fillna(mu)

        if corr_threshold is not None and len(cont_cols) > 1:
            corr = tmp_cont.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
            fitted["drop_cont_cols"] = to_drop

    kept_cont_cols = [c for c in cont_cols if c not in fitted.get("drop_cont_cols", [])]

    for c in kept_cont_cols:
        x = _to_numeric_series(df, c)
        mu = fitted["cont_stats"][c]["mean"]
        sd = fitted["cont_stats"][c]["std"]
        x = x.fillna(mu)
        out[c] = ((x - mu) / sd).astype(np.float32)

    # ---------------------------
    # Ordinal cols: 0..K -> [0,1]
    # ---------------------------
    for c in ordinal_cols:
        max_val = ordinal_max_map.get(c, 5)
        denom = float(max_val if max_val > 0 else 1)
        x = _to_numeric_series(df, c, default=0).fillna(0)
        x = x.clip(0, max_val) / denom
        out[c] = x.astype(np.float32)

    # ---------------------------
    # Binary cols: keep 0/1
    # ---------------------------
    for c in binary_cols:
        x = _to_numeric_series(df, c, default=0).fillna(0)
        x = x.clip(0, 1)
        out[c] = x.astype(np.float32)

    # ---------------------------
    # Unit cols: clamp [0,1]
    # ---------------------------
    for c in unit_cols:
        x = _to_numeric_series(df, c, default=0).fillna(0)
        x = x.clip(0, 1)
        out[c] = x.astype(np.float32)

    # preserve stable order
    final_cols = kept_cont_cols + list(ordinal_cols) + list(binary_cols) + list(unit_cols)
    out = out[final_cols]

    return out, fitted

# def get_data_all(
#     seq_tsv: str,
#     bio_tsv: str,
#     onto_tsv: Optional[str] = None,
#     text_tsv: Optional[str] = None,
#     textfields_tsv: Optional[str] = None,
#     labels_path: Optional[str] = None,
#     label_col: str = "labels",
#     include_ontology: bool = True,
#     include_oncology: bool = True,
#     cache_dir: str = "dataset",
#     use_cache: bool = True,
#     rebuild_cache: bool = False,
#     max_bio_line=None,
#     use_text: bool = False,
#     seq_offline_embeddings: bool = False,
#     text_offline_embeddings: bool = False,
#     dna_model_name: Optional[str] = None,
#     text_model_name: Optional[str] = None,
#     seq_pool: str = "mean",
#     text_pool: str = "mean",
#     seq_max_len: int = 512,
#     text_max_len: int = 256,
#     precompute_batch_size: int = 32,
#     embedding_device: str = "cuda",
#     precomputed_ref_path: Optional[str] = None,
#     precomputed_alt_path: Optional[str] = None,
#     precomputed_text_path: Optional[str] = None,
# ):
def get_data_all(
    seq_tsv: str,
    bio_tsv: str,
    onto_tsv: Optional[str] = None,
    text_tsv: Optional[str] = None,
    textfields_tsv: Optional[str] = None,
    labels_path: Optional[str] = None,
    label_col: str = "labels",
    include_ontology: bool = True,
    include_oncology: bool = True,
    cache_dir: str = "dataset",
    use_cache: bool = True,
    rebuild_cache: bool = False,
    max_bio_line=None,
    use_text: bool = False,
    seq_offline_embeddings: bool = False,
    text_offline_embeddings: bool = False,
    dna_model_name: Optional[str] = None,
    text_model_name: Optional[str] = None,
    seq_pool: str = "mean",
    text_pool: str = "mean",
    seq_max_len: int = 512,
    text_max_len: int = 256,
    precompute_batch_size: int = 32,
    embedding_device: str = "cuda",
    precomputed_ref_path: Optional[str] = None,
    precomputed_alt_path: Optional[str] = None,
    precomputed_text_path: Optional[str] = None,
    cont_cols: Optional[list] = None,
    ordinal_cols: Optional[list] = None,
    binary_cols: Optional[list] = None,
    unit_cols: Optional[list] = None,
    ordinal_max_map: Optional[dict] = None,
    corr_threshold: Optional[float] = None,
):
    try:
        cache_key = _make_dataset_cache_key(
            seq_tsv,
            bio_tsv,
            onto_tsv,
            text_tsv,
            textfields_tsv,
            labels_path,
            label_col,
            include_ontology,
            include_oncology,
            use_text=use_text,
            seq_offline_embeddings=seq_offline_embeddings,
            text_offline_embeddings=text_offline_embeddings,
            dna_model_name=dna_model_name,
            text_model_name=text_model_name,
            precomputed_ref_path=precomputed_ref_path,
            precomputed_alt_path=precomputed_alt_path,
            precomputed_text_path=precomputed_text_path,
        )

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"dataset_{cache_key}.pkl")

        if use_cache and os.path.exists(cache_path) and not rebuild_cache:
            print("*" * 50)
            print("Loading cached dataset:", cache_path)

            cached_dataset = _load_dataset_cache(cache_path)
            print("Full cached dataset size:", len(cached_dataset))

            cached_dataset = _truncate_variant_dataset(
                cached_dataset,
                max_bio_line=max_bio_line,
            )
            cached_length = len(cached_dataset)

            print("Returned cached dataset size:", cached_length)
            return cached_dataset, cached_length

        from model import precompute_offline_text_embeddings, precompute_offline_refalt_embeddings

        df = _build_merged_dataframe(
            seq_tsv=seq_tsv,
            bio_tsv=bio_tsv,
            onto_tsv=onto_tsv,
            text_tsv=text_tsv,
            textfields_tsv=textfields_tsv,
            label_col=label_col,
        )

        if max_bio_line is not None:
            max_bio_line = int(max_bio_line)
            if max_bio_line <= 0:
                raise ValueError("max_bio_line must be a positive integer")
            df = df.iloc[:max_bio_line].copy()

        print("The shape of the data to be kept is", df.shape)

        if labels_path is not None and os.path.exists(os.path.expanduser(labels_path)):
            labels = np.load(os.path.expanduser(labels_path))
            labels = labels[: len(df)]

        elif label_col in df.columns:
            labels = (
                pd.to_numeric(df[label_col], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

        elif "labels" in df.columns:
            labels = (
                pd.to_numeric(df["labels"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

        else:
            raise ValueError(f"No labels found. Expected labels_path or column '{label_col}'.")

        print("*" * 50)
        print("merged df shape:", df.shape)
        print("label length:", len(labels))

        if "SEQ_REF" in df.columns and "SEQ_ALT" in df.columns:
            seq_ref = _clean_seq_series(df["SEQ_REF"])
            seq_alt = _clean_seq_series(df["SEQ_ALT"])

        elif "REF_with_flank" in df.columns and "ALT_with_flank" in df.columns:
            seq_ref = _clean_seq_series(df["REF_with_flank"])
            seq_alt = _clean_seq_series(df["ALT_with_flank"])

        elif "MOTHER_with_flank" in df.columns and "FATHER_with_flank" in df.columns:
            seq_ref = _clean_seq_series(df["MOTHER_with_flank"])
            seq_alt = _clean_seq_series(df["FATHER_with_flank"])

        else:
            raise ValueError(
                "No valid sequence columns found. Expected one of: "
                "('SEQ_REF','SEQ_ALT'), "
                "('REF_with_flank','ALT_with_flank'), or "
                "('MOTHER_with_flank','FATHER_with_flank')."
            )

        bad_ref = validate_sequences(pd.DataFrame({"seq": seq_ref}), "seq")
        bad_alt = validate_sequences(pd.DataFrame({"seq": seq_alt}), "seq")
        print(f"bad_ref={len(bad_ref)} bad_alt={len(bad_alt)}")

        # bio_cols = [
        #     c for c in _infer_bio_feature_cols(bio_tsv, label_col=label_col)
        #     if c in df.columns
        # ]
        # bio_np = _safe_numeric_matrix(df[bio_cols])
        bio_cols = [
            c for c in _infer_bio_feature_cols(bio_tsv, label_col=label_col)
            if c in df.columns
        ]

        # Default: if no groups are provided, keep old behavior
        if cont_cols is None:
            cont_cols = []
        if ordinal_cols is None:
            ordinal_cols = []
        if binary_cols is None:
            binary_cols = []
        if unit_cols is None:
            unit_cols = []
        if ordinal_max_map is None:
            ordinal_max_map = {}

        # Only keep declared columns that truly exist in bio_cols
        cont_cols = [c for c in cont_cols if c in bio_cols]
        ordinal_cols = [c for c in ordinal_cols if c in bio_cols]
        binary_cols = [c for c in binary_cols if c in bio_cols]
        unit_cols = [c for c in unit_cols if c in bio_cols]

        typed_cols = set(cont_cols) | set(ordinal_cols) | set(binary_cols) | set(unit_cols)

        # Any leftover bio columns fall back to continuous
        # remaining_cols = [c for c in bio_cols if c not in typed_cols]
        # cont_cols = cont_cols + remaining_cols
        ALL_ALLOWED = set(cont_cols) | set(ordinal_cols) | set(binary_cols) | set(unit_cols)

        bio_cols = [c for c in bio_cols if c in ALL_ALLOWED]

        bio_df_processed, bio_prep = preprocess_bio_dataframe(
            df=df,
            cont_cols=cont_cols,
            ordinal_cols=ordinal_cols,
            binary_cols=binary_cols,
            unit_cols=unit_cols,
            cont_stats=None,
            ordinal_max_map=ordinal_max_map,
            corr_threshold=corr_threshold,
        )

        bio_cols = list(bio_df_processed.columns)
        bio_np = _safe_numeric_matrix(bio_df_processed)

        n = len(df)
        if bio_np.shape[0] != n:
            raise ValueError(f"Row mismatch: df has {n} rows but bio matrix has {bio_np.shape[0]} rows")

        if len(labels) != n:
            raise ValueError(f"Row mismatch: df has {n} rows but labels has {len(labels)} rows")

        onto_data = {}
        if include_ontology:
            onto_data = {c: _safe_text_col(df, c) for c in ONTO_COLS if c in df.columns}

        onco_data = {}
        if include_oncology:
            onco_data = {c: _safe_text_col(df, c) for c in ONCO_COLS if c in df.columns}

        meta_data = {c: _safe_text_col(df, c) for c in META_COLS if c in df.columns}

        texts = None
        if use_text:
            candidate_text_cols = [c for c in ONCO_COLS if c in df.columns]
            texts = build_texts_from_df(
                df,
                text_cols=candidate_text_cols,
                prefix_with_colname=True,
            )

        ref_embs = None
        alt_embs = None
        text_embs = None

        if seq_offline_embeddings:
            if precomputed_ref_path is not None and os.path.exists(precomputed_ref_path):
                ref_embs = torch.load(precomputed_ref_path, map_location="cpu")
                ref_embs = ref_embs.cpu().numpy()

            if precomputed_alt_path is not None and os.path.exists(precomputed_alt_path):
                alt_embs = torch.load(precomputed_alt_path, map_location="cpu")
                alt_embs = alt_embs.cpu().numpy()

            if ref_embs is None or alt_embs is None:
                if dna_model_name is None:
                    raise ValueError("dna_model_name is required for seq_offline_embeddings=True")

                if precomputed_ref_path is None or precomputed_alt_path is None:
                    raise ValueError(
                        "precomputed_ref_path and precomputed_alt_path must be provided "
                        "when seq_offline_embeddings=True"
                    )

                ref_t, alt_t = precompute_offline_refalt_embeddings(
                    seq_refs=seq_ref,
                    seq_alts=seq_alt,
                    model_name=dna_model_name,
                    out_ref_path=precomputed_ref_path,
                    out_alt_path=precomputed_alt_path,
                    batch_size=precompute_batch_size,
                    pool=seq_pool,
                    max_len=seq_max_len,
                    device=embedding_device,
                )
                ref_embs = ref_t.cpu().numpy()
                alt_embs = alt_t.cpu().numpy()

        if use_text and text_offline_embeddings:
            if precomputed_text_path is not None and os.path.exists(precomputed_text_path):
                text_embs = torch.load(precomputed_text_path, map_location="cpu")
                text_embs = text_embs.cpu().numpy()

            if text_embs is None:
                if text_model_name is None:
                    raise ValueError("text_model_name is required for text_offline_embeddings=True")
                if texts is None:
                    raise ValueError("texts is None but text_offline_embeddings=True")
                if precomputed_text_path is None:
                    raise ValueError("precomputed_text_path must be provided when text_offline_embeddings=True")

                text_t = precompute_offline_text_embeddings(
                    texts=texts,
                    model_name=text_model_name,
                    out_text_path=precomputed_text_path,
                    batch_size=precompute_batch_size,
                    pool=text_pool,
                    max_len=text_max_len,
                    trust_remote_code=False,
                    device=embedding_device,
                )
                text_embs = text_t.cpu().numpy()

        dataset_seq_ref = None if seq_offline_embeddings else seq_ref
        dataset_seq_alt = None if seq_offline_embeddings else seq_alt
        dataset_texts = None if (use_text and text_offline_embeddings) else texts

        dataset = VariantsDatasetRefAlt(
            seq_refs=dataset_seq_ref,
            seq_alts=dataset_seq_alt,
            bio_feats=bio_np,
            labels=labels,
            texts=dataset_texts,
            ref_embs=ref_embs,
            alt_embs=alt_embs,
            text_embs=text_embs,
            onto_data=onto_data,
            onco_data=onco_data,
            meta_data=meta_data,
            bio_feature_cols=bio_cols,
        )

        length = len(dataset)

        if use_cache:
            print("Saving dataset cache ->", cache_path)
            _save_dataset_cache(cache_path, dataset)
            print("Dataset cached successfully")

        del df, bio_np, labels
        gc.collect()

        return dataset, length

    except Exception as e:
        print("\nERROR OCCURRED:", e)
        gc.collect()
        raise


@torch.no_grad()
def encode_in_batches_texts(encoder, texts, batch_size=32, device="cuda"):
    encoder = encoder.to(device)
    encoder.eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encoder(batch)
        all_embs.append(emb.detach().cpu())

    return torch.cat(all_embs, dim=0)






# =========================================================
# Offline embedding utilities
# =========================================================
from typing import List, Optional, Tuple
import os
import gc
import re

import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings using attention_mask.
    last_hidden_state: (B, L, D)
    attention_mask:    (B, L)
    returns:           (B, D)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, D)
    denom = mask.sum(dim=1).clamp(min=1e-8)                         # (B, 1)
    return summed / denom


def cls_pool_last_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    CLS pooling: take token 0.
    """
    return last_hidden_state[:, 0, :]


def pool_hidden_states(
    outputs,
    attention_mask: torch.Tensor,
    pool: str = "mean",
) -> torch.Tensor:
    """
    pool in {"mean", "cls"}
    """
    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError("Model outputs do not contain last_hidden_state")

    last_hidden = outputs.last_hidden_state

    if pool == "mean":
        return mean_pool_last_hidden(last_hidden, attention_mask)
    elif pool == "cls":
        return cls_pool_last_hidden(last_hidden)
    else:
        raise ValueError(f"Unsupported pool='{pool}'. Use 'mean' or 'cls'.")


def load_hf_encoder_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    device: str = "cuda",
):
    """
    Generic HuggingFace encoder loader.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model = model.to(device)
    model.eval()
    return tokenizer, model


def clean_sequence_for_tokenizer(seq: str) -> str:
    """
    Clean DNA sequence and uppercase.
    Non-ACGTN -> N
    """
    if seq is None:
        return ""
    s = str(seq).upper().strip()
    s = re.sub(r"\s+", "", s)
    if s == "":
        return ""
    return "".join(ch if ch in {"A", "C", "G", "T", "N"} else "N" for ch in s)


def tokenize_dna_batch(
    tokenizer,
    seqs: List[str],
    max_len: int = 512,
):
    """
    Generic DNA tokenization.
    For DNABERT-like models, space-separated characters usually works robustly.
    Example: ACGTN -> "A C G T N"
    """
    clean = [clean_sequence_for_tokenizer(s) for s in seqs]
    as_spaced = [" ".join(list(s)) for s in clean]

    toks = tokenizer(
        as_spaced,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return toks


@torch.no_grad()
def encode_sequences_hf(
    seqs: List[str],
    model_name: str,
    batch_size: int = 32,
    pool: str = "mean",
    max_len: int = 512,
    trust_remote_code: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode a list of DNA sequences into embeddings.
    returns: (N, D) torch.Tensor on CPU
    """
    tokenizer, model = load_hf_encoder_and_tokenizer(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )

    all_embs = []

    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        toks = tokenize_dna_batch(tokenizer, batch, max_len=max_len)
        toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model(**toks)
        emb = pool_hidden_states(outputs, toks["attention_mask"], pool=pool)
        all_embs.append(emb.detach().cpu())

        del toks, outputs, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts_hf(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    pool: str = "mean",
    max_len: int = 256,
    trust_remote_code: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode a list of text strings into embeddings.
    returns: (N, D) torch.Tensor on CPU
    """
    tokenizer, model = load_hf_encoder_and_tokenizer(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )

    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch = ["" if x is None else str(x) for x in batch]

        toks = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model(**toks)
        emb = pool_hidden_states(outputs, toks["attention_mask"], pool=pool)
        all_embs.append(emb.detach().cpu())

        del toks, outputs, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0)