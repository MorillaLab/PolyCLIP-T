from __future__ import annotations
from libs import *

def rank0_print(*args, **kwargs):
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        print(*args, **kwargs)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_warmup(epoch: int, warmup_epochs: int, max_val: float) -> float:
    if warmup_epochs <= 0:
        return float(max_val)
    return float(max_val) * min(1.0, epoch / float(warmup_epochs))


def check_nans(tensor: torch.Tensor, name: str):
    if tensor is None:
        return
    if torch.isnan(tensor).any():
        rank0_print(f"[WARN] NaN detected in {name} (rank={dist.get_rank() if dist.is_initialized() else 0})")
    if torch.isinf(tensor).any():
        rank0_print(f"[WARN] Inf detected in {name} (rank={dist.get_rank() if dist.is_initialized() else 0})")



def setup_distributed(backend: str | None = None):

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size

    has_cuda = torch.cuda.is_available()
    has_nccl = dist.is_nccl_available()

    if backend is None:
        backend = "nccl" if (has_cuda and has_nccl) else "gloo"
    else:
        backend = backend.lower()
        if backend == "nccl" and not (has_cuda and has_nccl):
            print("[DDP] NCCL requested but not available -> switching to GLOO")
            backend = "gloo"


    dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if has_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    return rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
def _reduce_split(
    X: np.ndarray,
    y: np.ndarray,
    r: list[str],
    a: list[str],
    max_items: int,
    seed: int = 42,
    stratify: bool = True,
):

    n = int(X.shape[0])
    if max_items is None or n <= int(max_items):
        return X, y, r, a

    max_items = int(max_items)
    rng = np.random.RandomState(int(seed))

    if stratify and y is not None and len(np.unique(y)) >= 2:
        idx_keep_parts = []
        classes, counts = np.unique(y, return_counts=True)
        for c, cnt in zip(classes, counts):
            idx_c = np.where(y == c)[0]
            k_c = int(round(max_items * (cnt / n)))
            k_c = max(1, min(k_c, len(idx_c)))
            idx_keep_parts.append(rng.choice(idx_c, size=k_c, replace=False))

        idx_keep = np.concatenate(idx_keep_parts)
        if idx_keep.size > max_items:
            idx_keep = rng.choice(idx_keep, size=max_items, replace=False)
        elif idx_keep.size < max_items:
            missing = max_items - idx_keep.size
            rest = np.setdiff1d(np.arange(n), idx_keep, assume_unique=False)
            if rest.size > 0:
                extra = rng.choice(rest, size=min(missing, rest.size), replace=False)
                idx_keep = np.concatenate([idx_keep, extra])
    else:
        idx_keep = rng.choice(np.arange(n), size=max_items, replace=False)

    idx_keep = np.sort(idx_keep)

    X2 = X[idx_keep]
    y2 = y[idx_keep]
    r2 = [r[i] for i in idx_keep]
    a2 = [a[i] for i in idx_keep]
    return X2, y2, r2, a2



class VariantsDatasetRefAlt(Dataset):

    def __init__(self, seq_refs: List[str], seq_alts: List[str], bio_feats: np.ndarray, labels: np.ndarray):
        assert len(seq_refs) == len(seq_alts), "SEQ_REF and SEQ_ALT length mismatch"
        assert bio_feats.shape[0] == len(seq_refs), "bio_feats rows != number of sequences"
        assert labels.shape[0] == len(seq_refs), "labels length != number of sequences"

        self.seq_refs = list(seq_refs)
        self.seq_alts = list(seq_alts)
        self.bio = torch.tensor(bio_feats, dtype=torch.float32)
        self.label = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.seq_refs)

    def __getitem__(self, idx: int):
        return idx, self.seq_refs[idx], self.seq_alts[idx], self.bio[idx], self.label[idx]


def make_train_subset(
    train_ds: Dataset,
    max_items: int = 5000,
    min_pos_ratio: float = 0.1,
    seed: int = 0,
    pos_label: int = 1,
    neg_label: int = -1,
):

    rng = np.random.default_rng(seed)
    labels = np.array([int(train_ds[i][4]) for i in range(len(train_ds))], dtype=int)

    pos_idxs = np.where(labels == pos_label)[0]
    neg_idxs = np.where(labels == neg_label)[0]

    if len(pos_idxs) == 0:
        raise ValueError("No positive samples (label==pos_label) in dataset subset builder.")

    min_pos = max(1, int(max_items * float(min_pos_ratio)))
    n_pos = min(min_pos, len(pos_idxs))
    n_neg = max_items - n_pos

    chosen_pos = rng.choice(pos_idxs, size=n_pos, replace=False)
    replace_neg = len(neg_idxs) < n_neg
    chosen_neg = rng.choice(neg_idxs, size=n_neg, replace=replace_neg) if n_neg > 0 else np.array([], dtype=int)

    idxs = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idxs)
    return Subset(train_ds, idxs.tolist())



def clean_sequence(seq, max_len: int = 512) -> str:
    if not isinstance(seq, str):
        return "A" * 10
    s = seq.upper().replace(" ", "").replace(",", "")
    s = "".join([c if c in "ATCGN" else "A" for c in s])
    s = s[:max_len]
    if len(s) == 0:
        s = "N" * 10
    return s


def validate_sequences(df: pd.DataFrame, column_name: str, max_len: int = 512):
    bad_idx = []
    for i, seq in enumerate(df[column_name].tolist()):
        if not isinstance(seq, str):
            bad_idx.append(i)
            continue
        s = seq.strip().upper()
        if len(s) == 0:
            bad_idx.append(i)
        elif len(s) > max_len:
            bad_idx.append(i)
        elif " " in s:
            bad_idx.append(i)
        elif not all(c in "ATCGN" for c in s):
            bad_idx.append(i)
    return bad_idx



_VR: Optional[VietorisRipsPersistence] = None
_PI: Optional[PersistenceImage] = None


def init_gtda_objects(cfg):
    global _VR, _PI
    hom_dims = getattr(cfg, "tda_hom_dims", (0, 1))
    _VR = VietorisRipsPersistence(
        homology_dimensions=hom_dims,
        metric="euclidean",
        n_jobs=int(getattr(cfg, "tda_n_jobs", 1)),
    )
    _PI = PersistenceImage(
        n_bins=int(getattr(cfg, "tda_pi_bins", 32)),
        sigma=float(getattr(cfg, "tda_pi_sigma", 0.10)),
    )


@torch.no_grad()
def _subsample_points_np(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if X.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]


@torch.no_grad()
def pure_tda_feature_from_batch(cfg, h_tda: torch.Tensor, epoch: int, batch_idx: int) -> torch.Tensor:
    assert _VR is not None and _PI is not None, "Call init_gtda_objects(cfg) before using pure TDA."
    X = h_tda.detach().float().cpu().numpy()
    P = int(getattr(cfg, "tda_subsample", 256))
    seed = int(getattr(cfg, "seed", 0)) + int(epoch) * 10_000 + int(batch_idx)

    X = _subsample_points_np(X, P, seed=seed).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(X, axis=1).mean())
    if scale > 1e-6:
        X = X / scale

    dgms = _VR.fit_transform([X])
    imgs = _PI.fit_transform(dgms)
    feat = imgs.reshape(-1).astype(np.float32)
    return torch.from_numpy(feat)



def _load_numeric_tsv(path: str, label_path=None) -> tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "labels" not in df.columns:
        if label_path:
            labels_np = np.load(label_path)
        else:
            raise ValueError(f"{path} must contain a 'labels' column")
    else:
        labels_np = df["labels"].to_numpy()
    labels_np = np.asarray(labels_np, dtype=int)
    labels_np = np.where(labels_np == 0, -1, labels_np).astype(int)


    drop_cols = [c for c in ["FAMILY_ID", "CHROM", "POS", "REF", "ALT", "allele_1", "allele_2", "labels"] if c in df.columns]
    X = (
        df.drop(columns=drop_cols, errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    print("*"*100)
    print("The shape of dataset")
    print(X.shape)
    return X, labels_np


def _load_seq_tsv(path: str, max_len: int = 512) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "SEQ_REF" not in df.columns or "SEQ_ALT" not in df.columns:
        raise ValueError(f"{path} must contain SEQ_REF and SEQ_ALT columns")

    seq_ref = df["SEQ_REF"].apply(lambda s: clean_sequence(s, max_len=max_len)).tolist()
    seq_alt = df["SEQ_ALT"].apply(lambda s: clean_sequence(s, max_len=max_len)).tolist()

    bad_ref = validate_sequences(pd.DataFrame({"REF": seq_ref}), "REF", max_len=max_len)
    bad_alt = validate_sequences(pd.DataFrame({"ALT": seq_alt}), "ALT", max_len=max_len)
    if len(bad_ref) or len(bad_alt):
        rank0_print(f"[WARN] {path}: bad_ref={len(bad_ref)} bad_alt={len(bad_alt)} (they were cleaned)")
    print("Print shape in _load_seq_tsv ",len(seq_ref))
    return seq_ref, seq_alt





def get_data_all(
    numeric_train: str,
    seq_train: str,
    label_train: str,
    numeric_test: Optional[str] = None,
    seq_test: Optional[str] = None,
    label_test: Optional[str] = None,
    max_len: int = 512,

    max_train_items: Optional[int] = None,
    max_test_items: Optional[int] = None,
    reduce_seed: int = 42,
    stratify_reduce: bool = True,
):

    Xtr, ytr = _load_numeric_tsv(numeric_train, label_train)
    rtr, atr = _load_seq_tsv(seq_train, max_len=max_len)

    if Xtr.shape[0] != len(rtr):
        raise ValueError(f"Train mismatch: numeric rows={Xtr.shape[0]} vs seq rows={len(rtr)}")

    if max_train_items is not None:
        Xtr, ytr, rtr, atr = _reduce_split(
            Xtr, ytr, rtr, atr,
            max_items=int(max_train_items),
            seed=int(reduce_seed),
            stratify=bool(stratify_reduce),
        )

    train_ds = VariantsDatasetRefAlt(rtr, atr, Xtr, ytr)

    test_ds = None
    if numeric_test is not None and seq_test is not None:
        Xte, yte = _load_numeric_tsv(numeric_test, label_test)
        rte, ate = _load_seq_tsv(seq_test, max_len=max_len)

        if Xte.shape[0] != len(rte):
            raise ValueError(f"Test mismatch: numeric rows={Xte.shape[0]} vs seq rows={len(rte)}")

        if max_test_items is not None:
            Xte, yte, rte, ate = _reduce_split(
                Xte, yte, rte, ate,
                max_items=int(max_test_items),
                seed=int(reduce_seed) + 1,  
                stratify=bool(stratify_reduce),
            )

        test_ds = VariantsDatasetRefAlt(rte, ate, Xte, yte)

    return train_ds, test_ds, ytr
