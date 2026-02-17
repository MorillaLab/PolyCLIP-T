from __future__ import annotations

from libs import *
from config import Config
from loss import collate_fn_refalt

import os
import numpy as np
import torch
import torch.nn.functional as F


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0

def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


@torch.no_grad()
def _gather_variable_1d_cpu(t: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return t

    assert t.dim() == 1, "Expected 1D tensor"
    lengths = torch.tensor([t.numel()], dtype=torch.long, device="cpu")
    lengths_list = [torch.zeros_like(lengths) for _ in range(_world())]
    dist.all_gather(lengths_list, lengths)
    lens = [int(x.item()) for x in lengths_list]
    max_len = max(lens) if lens else 0

    padded = t.contiguous()
    if padded.numel() < max_len:
        pad = torch.zeros(max_len - padded.numel(), dtype=padded.dtype, device="cpu")
        padded = torch.cat([padded, pad], dim=0)

    gathered = [torch.zeros(max_len, dtype=padded.dtype, device="cpu") for _ in range(_world())]
    dist.all_gather(gathered, padded)

    if _rank() != 0:
        return torch.empty(0, dtype=padded.dtype, device="cpu")

    out = []
    for gi, L in zip(gathered, lens):
        if L > 0:
            out.append(gi[:L])
    return torch.cat(out, dim=0) if out else torch.empty(0, dtype=padded.dtype, device="cpu")


@torch.no_grad()
def _gather_variable_2d_cpu(x: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return x

    assert x.dim() == 2, "Expected 2D tensor (N,D)"
    x = x.contiguous()
    n, d = x.shape

    lengths = torch.tensor([n], dtype=torch.long, device="cpu")
    lengths_list = [torch.zeros_like(lengths) for _ in range(_world())]
    dist.all_gather(lengths_list, lengths)
    lens = [int(v.item()) for v in lengths_list]
    max_len = max(lens) if lens else 0

    padded = x
    if n < max_len:
        pad = torch.zeros((max_len - n, d), dtype=x.dtype, device="cpu")
        padded = torch.cat([x, pad], dim=0)

    gathered = [torch.zeros((max_len, d), dtype=x.dtype, device="cpu") for _ in range(_world())]
    dist.all_gather(gathered, padded)

    if _rank() != 0:
        return torch.empty((0, d), dtype=x.dtype, device="cpu")

    out = []
    for gi, L in zip(gathered, lens):
        if L > 0:
            out.append(gi[:L])
    return torch.cat(out, dim=0) if out else torch.empty((0, d), dtype=x.dtype, device="cpu")



def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def _encode_batch(cfg: Config, model: nn.Module, seq_refs, seq_alts, bios, rep: str):
    m = _unwrap_model(model)
    bios = bios.to(cfg.device)

    if rep == "fused":
        return m.encode_fused(seq_refs, seq_alts, bios)

    z_seq, z_bio, h_tda, *_ = m.encode_views(seq_refs, seq_alts, bios)
    if rep == "z_seq":
        return z_seq
    if rep == "z_bio":
        return z_bio
    if rep == "h_tda":
        return h_tda
    raise ValueError(f"Unknown rep='{rep}'")



@torch.no_grad()
def encode_dataset(cfg: Config, model: nn.Module, dataset: Dataset, rep: str = "z_bio", sampler=None):

    m = _unwrap_model(model)
    m.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        collate_fn=collate_fn_refalt,
        pin_memory=False,
        drop_last=False,
    )

    idxs_all, labs_all, emb_all = [], [], []

    for idxs, seq_refs, seq_alts, bios, label in loader:
        E = _encode_batch(cfg, m, seq_refs, seq_alts, bios, rep=rep)
        idxs_all.append(torch.as_tensor(idxs, dtype=torch.long).cpu())
        labs_all.append(label.detach().long().cpu())
        emb_all.append(E.detach().float().cpu())

    idxs_all = torch.cat(idxs_all, dim=0) if idxs_all else torch.empty(0, dtype=torch.long)
    labs_all = torch.cat(labs_all, dim=0) if labs_all else torch.empty(0, dtype=torch.long)
    emb_all  = torch.cat(emb_all,  dim=0) if emb_all  else torch.empty(0, dtype=torch.float32)
    return idxs_all, emb_all, labs_all



@torch.no_grad()
def encode_dataset_global(cfg: Config, model: nn.Module, dataset: Dataset, rep: str = "z_bio"):

    if _is_dist():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=_world(),
            rank=_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    idxs, E, y = encode_dataset(cfg, model, dataset, rep=rep, sampler=sampler)

    if not _is_dist():
        return idxs, E, y

    idxs_g = _gather_variable_1d_cpu(idxs.cpu().long())
    y_g    = _gather_variable_1d_cpu(y.cpu().long())
    E_g    = _gather_variable_2d_cpu(E.cpu().float())

    if _rank() != 0:
        d = E.shape[1] if E.dim() == 2 else 0
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty((0, d), dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )

    if idxs_g.numel() > 0:
        order = torch.argsort(idxs_g)
        idxs_g = idxs_g[order]
        y_g    = y_g[order]
        E_g    = E_g[order]

    return idxs_g, E_g, y_g


@torch.no_grad()
def global_retrieval_metrics(
    cfg: Config,
    model: nn.Module,
    dataset: Dataset,
    rep_a: str = "z_seq",
    rep_b: str = "z_bio",
    temperature: float | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
):

    m = _unwrap_model(model)
    m.eval()
    tau = float(temperature if temperature is not None else getattr(cfg, "temperature", 0.07))

    idx_a, A, _ = encode_dataset_global(cfg, m, dataset, rep=rep_a)
    idx_b, B, _ = encode_dataset_global(cfg, m, dataset, rep=rep_b)

    if _is_dist() and _rank() != 0:
        return {}

    if A.numel() == 0 or B.numel() == 0:
        return {}

    if idx_a.numel() != idx_b.numel() or not torch.equal(idx_a, idx_b):
        b_map = {int(idx_b[i].item()): i for i in range(idx_b.numel())}
        keep = []
        b_rows = []
        for i in range(idx_a.numel()):
            k = int(idx_a[i].item())
            if k in b_map:
                keep.append(i)
                b_rows.append(b_map[k])
        if len(keep) == 0:
            return {}
        keep = torch.tensor(keep, dtype=torch.long)
        b_rows = torch.tensor(b_rows, dtype=torch.long)
        A = A[keep]
        B = B[b_rows]
        idx_a = idx_a[keep]  

    A = F.normalize(A.float(), dim=1)
    B = F.normalize(B.float(), dim=1)
    N = A.size(0)

    logits = (A @ B.T) / tau 
    gt = torch.arange(N)

    order = torch.argsort(logits, dim=1, descending=True)  # (N,N)
    pos = (order == gt[:, None]).nonzero(as_tuple=False)
    ranks = torch.empty(N, dtype=torch.long)
    ranks[pos[:, 0]] = pos[:, 1]
    ranks1 = (ranks + 1).cpu().numpy()

    def pack(prefix: str, ranks1: np.ndarray):
        out = {}
        for k in ks:
            out[f"{prefix}/R@{k}"] = float(np.mean(ranks1 <= int(k)))
        out[f"{prefix}/median_rank"] = float(np.median(ranks1))
        out[f"{prefix}/mean_rank"] = float(np.mean(ranks1))
        return out

    out = {}
    out.update(pack("i2t", ranks1))

    order_t = torch.argsort(logits.T, dim=1, descending=True)
    pos_t = (order_t == gt[:, None]).nonzero(as_tuple=False)
    ranks_t = torch.empty(N, dtype=torch.long)
    ranks_t[pos_t[:, 0]] = pos_t[:, 1]
    ranks1_t = (ranks_t + 1).cpu().numpy()
    out.update(pack("t2i", ranks1_t))

    out["n"] = int(N)
    out["tau"] = float(tau)
    out["rep_a"] = rep_a
    out["rep_b"] = rep_b
    return out




@torch.no_grad()
def knn_purity(
    emb: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    unlabeled_val: int = -1,
    chunk: int = 2048,
):

    emb = F.normalize(emb, dim=1)
    labels = labels.view(-1).long()
    N = emb.size(0)

    mask = labels != int(unlabeled_val)
    idx_l = torch.where(mask)[0]
    if idx_l.numel() == 0:
        return {"knn_purity": None, "n_labeled": 0}

    purity_sum = 0.0
    count = 0

    for start in range(0, idx_l.numel(), chunk):
        ii = idx_l[start:start + chunk]
        Q = emb[ii]                 # (C,D)
        sims = Q @ emb.T            # (C,N)
        sims[torch.arange(sims.size(0)), ii] = -1e9

        nn = sims.topk(k=min(k, N - 1), dim=1).indices
        same = (labels[nn] == labels[ii].unsqueeze(1)).float()
        purity_sum += same.mean(dim=1).sum().item()
        count += ii.numel()

    return {"knn_purity": purity_sum / max(1, count), "n_labeled": int(count)}



@torch.no_grad()
def task_eval_variant_ranking_from_loader(
    cfg: Config,
    model: nn.Module,
    dataset: Dataset,
    rep: str = "fused",
    ks: tuple[int, ...] = (10, 50, 100, 500),
    score_mode: str = "cosine_to_pos_centroid",
):
    """
    Uses the 'label' returned by collate_fn_refalt.
    Any label == -1 is converted to 0.
    DDP-safe: computes on rank0 using global gathered embeddings+labels.
    """
    m = _unwrap_model(model)
    m.eval()

    _idxs, E, y = encode_dataset_global(cfg, m, dataset, rep=rep)

    if _is_dist() and _rank() != 0:
        return {}

    if E.numel() == 0:
        return {}

    y = y.clone().cpu()
    y[y == -1] = 0
    y = y.numpy().astype(np.int64)

    E = F.normalize(E.float(), dim=1).cpu()

    if score_mode == "cosine_to_pos_centroid":
        mask_pos = (y == 1)
        if mask_pos.sum() == 0:
            out = {"rep": rep, "score_mode": score_mode, "auroc": None, "auprc": None}
            for k in ks:
                out[f"precision@{k}"] = None
            out.update({"n": int(len(y)), "n_pos": 0, "n_neg": int((y == 0).sum())})
            return out

        centroid = E[torch.from_numpy(mask_pos)].mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, dim=1)
        scores = (E @ centroid.T).squeeze(1).numpy()
    elif score_mode == "embedding_norm":
        scores = torch.norm(E, dim=1).numpy()
    else:
        raise ValueError(f"Unknown score_mode='{score_mode}'")

    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(np.unique(y)) < 2:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(y, scores))
        auprc = float(average_precision_score(y, scores))

    out = {
        "rep": rep,
        "score_mode": score_mode,
        "auroc": auroc,
        "auprc": auprc,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }

    order = np.argsort(-scores)
    for k in ks:
        k_eff = min(int(k), len(order))
        out[f"precision@{k}"] = float((y[order[:k_eff]] == 1).mean()) if k_eff > 0 else None

    return out



def _reduce_2d(emb: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    if method == "umap":
        try:
            import umap  # umap-learn
            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
            return reducer.fit_transform(emb)
        except Exception:
            pass

    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(emb)


def _cluster_labels(emb: np.ndarray, method: str = "hdbscan", seed: int = 42) -> np.ndarray:
    if method == "hdbscan":
        try:
            import hdbscan
            cl = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean")
            return cl.fit_predict(emb)
        except Exception:
            pass

    from sklearn.cluster import KMeans
    k = 12
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(emb)


def _save_scatter(path: str, xy: np.ndarray, labels: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6))

    labels = np.asarray(labels)
    xy = np.asarray(xy)

    uniq = np.unique(labels)
    if uniq.size <= 1 or uniq.size > 50:
        plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8)
    else:
        for lab in uniq:
            m = labels == lab
            plt.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.85, label=str(lab))
        plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@torch.no_grad()
def alignment_uniformity(
    cfg: Config,
    model: nn.Module,
    dataset: Dataset,
    rep_a: str = "z_seq",
    rep_b: str = "z_bio",
    t: float = 2.0,
    max_pairs: int = 20000,   # subsample for speed
    seed: int = 42,
):

    m = _unwrap_model(model)
    m.eval()

    idx_a, A, _y = encode_dataset_global(cfg, m, dataset, rep=rep_a)
    idx_b, B, _y2 = encode_dataset_global(cfg, m, dataset, rep=rep_b)

    if _is_dist() and _rank() != 0:
        return {}
    if A.numel() == 0 or B.numel() == 0:
        return {}

    A = F.normalize(A.float(), dim=1)
    B = F.normalize(B.float(), dim=1)

    if not torch.equal(idx_a, idx_b):
        idx_b_np = idx_b.cpu().numpy()
        pos = {int(k): i for i, k in enumerate(idx_b_np)}
        keep = []
        rows = []
        for i, k in enumerate(idx_a.cpu().numpy()):
            j = pos.get(int(k), None)
            if j is not None:
                keep.append(i)
                rows.append(j)
        A = A[torch.tensor(keep)]
        B = B[torch.tensor(rows)]

    N = A.size(0)
    if N == 0:
        return {}

    align = torch.mean(torch.sum((A - B) ** 2, dim=1)).item()

    g = torch.Generator().manual_seed(int(seed))
    M = min(int(max_pairs), N * (N - 1) // 2 if N > 1 else 0)
    if M <= 0:
        uni_a = uni_b = None
    else:
        i = torch.randint(0, N, (M,), generator=g)
        j = torch.randint(0, N, (M,), generator=g)
        neq = (i != j)
        i, j = i[neq], j[neq]
        d2_a = 2.0 - 2.0 * torch.sum(A[i] * A[j], dim=1)
        d2_b = 2.0 - 2.0 * torch.sum(B[i] * B[j], dim=1)
        uni_a = torch.log(torch.mean(torch.exp(-t * d2_a))).item()
        uni_b = torch.log(torch.mean(torch.exp(-t * d2_b))).item()

    return {
        "rep_a": rep_a,
        "rep_b": rep_b,
        "alignment": float(align),
        "uniformity_a": None if uni_a is None else float(uni_a),
        "uniformity_b": None if uni_b is None else float(uni_b),
        "t": float(t),
        "n": int(N),
        "max_pairs_used": int(M if M > 0 else 0),
    }



def _save_hist(path: str, values: np.ndarray, title: str, bins: int = 100):

    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(v, bins=int(bins))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_embedding_histograms(
    out_dir: str,
    split_name: str,
    rep: str,
    E: torch.Tensor,
    bins: int = 120,
    max_points_values: int = 2_000_000,
):

    E = E.detach().float().cpu()
    if E.numel() == 0:
        return

    norms = torch.norm(E, dim=1).numpy()
    _save_hist(
        os.path.join(out_dir, f"{split_name}__{rep}__hist_norms.png"),
        norms,
        title=f"{split_name} {rep} | ||emb||",
        bins=bins,
    )

    vals = E.reshape(-1).numpy()
    if vals.size > int(max_points_values):
        idx = np.linspace(0, vals.size - 1, int(max_points_values)).astype(np.int64)
        vals = vals[idx]

    _save_hist(
        os.path.join(out_dir, f"{split_name}__{rep}__hist_values.png"),
        vals,
        title=f"{split_name} {rep} | emb values (flattened)",
        bins=bins,
    )



@torch.no_grad()
def run_all_results(
    cfg: Config,
    model: nn.Module,
    train_ds: Dataset | None = None,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    out_dir: str | None = None,
    epoch: int | None = None,
    reps: Tuple[str, ...] = ("z_seq", "z_bio", "h_tda", "fused"),
    make_plots: bool = True,
    reduce_method: str = "umap",
    cluster_method: str = "hdbscan",
    seed: int = 42,
) -> Dict[str, Any]:
    rank = _rank()
    is_dist = _is_dist()
    m = _unwrap_model(model)
    m.eval()

    metrics: Dict[str, Any] = {}


    if val_ds is not None:
        r = global_retrieval_metrics(cfg, m, val_ds, rep_a="z_seq", rep_b="z_bio", ks=(1, 5, 10))
        if (not is_dist) or rank == 0:
            metrics.update({f"val/retrieval/{k}": v for k, v in r.items()})

    if test_ds is not None:
        r = global_retrieval_metrics(cfg, m, test_ds, rep_a="z_seq", rep_b="z_bio", ks=(1, 5, 10))
        if (not is_dist) or rank == 0:
            metrics.update({f"test/retrieval/{k}": v for k, v in r.items()})

    def _encode_for_reporting(ds: Dataset, rep: str):

        if is_dist:
            return encode_dataset_global(cfg, m, ds, rep=rep)
        return encode_dataset(cfg, m, ds, rep=rep)

    def _do_embed_split(split_name: str, ds: Dataset):
        for rep in reps:
            _idxs, E, y = _encode_for_reporting(ds, rep=rep)

            if is_dist and rank != 0:
                continue

            p = knn_purity(
                E, y,
                k=int(getattr(cfg, "knn_k", 10)),
                unlabeled_val=int(getattr(cfg, "unlabeled_val", -1)),
            )
            metrics.update({f"{split_name}/{rep}/{k}": v for k, v in p.items()})

            if out_dir is not None and rank == 0:
                os.makedirs(out_dir, exist_ok=True)

                np.save(os.path.join(out_dir, f"{split_name}__{rep}__emb.npy"), E.numpy())
                np.save(os.path.join(out_dir, f"{split_name}__{rep}__labels.npy"), y.numpy())

                save_embedding_histograms(
                    out_dir=out_dir,
                    split_name=split_name,
                    rep=rep,
                    E=E,
                    bins=int(getattr(cfg, "hist_bins", 120)),
                    max_points_values=int(getattr(cfg, "hist_max_points_values", 2_000_000)),
                )

                if make_plots:
                    xy = _reduce_2d(E.numpy(), method=reduce_method, seed=seed)
                    np.save(os.path.join(out_dir, f"{split_name}__{rep}__xy.npy"), xy)

                    cl = _cluster_labels(xy, method=cluster_method, seed=seed)
                    np.save(os.path.join(out_dir, f"{split_name}__{rep}__cluster.npy"), cl)

                    _save_scatter(
                        os.path.join(out_dir, f"{split_name}__{rep}__scatter_labels.png"),
                        xy,
                        y.numpy(),
                        title=f"{split_name} {rep} (labels)",
                    )
                    _save_scatter(
                        os.path.join(out_dir, f"{split_name}__{rep}__scatter_cluster.png"),
                        xy,
                        cl,
                        title=f"{split_name} {rep} (clusters)",
                    )

    if train_ds is not None:
        _do_embed_split("train", train_ds)
    if val_ds is not None:
        _do_embed_split("val", val_ds)
    if test_ds is not None:
        _do_embed_split("test", test_ds)


    if val_ds is not None:
        te = task_eval_variant_ranking_from_loader(
            cfg, m, val_ds,
            rep="fused",
            ks=(10, 50, 100, 500),
            score_mode="cosine_to_pos_centroid",
        )
        if (not is_dist) or rank == 0:
            metrics.update({f"val/task/{k}": v for k, v in te.items()})

        au = alignment_uniformity(cfg, m, val_ds, rep_a="z_seq", rep_b="z_bio", t=2.0)
        if (not is_dist) or rank == 0:
            metrics.update({f"val/au/{k}": v for k, v in au.items()})

    if test_ds is not None:
        te = task_eval_variant_ranking_from_loader(
            cfg, m, test_ds,
            rep="fused",
            ks=(10, 50, 100, 500),
            score_mode="cosine_to_pos_centroid",
        )
        if (not is_dist) or rank == 0:
            metrics.update({f"test/task/{k}": v for k, v in te.items()})

        au = alignment_uniformity(cfg, m, test_ds, rep_a="z_seq", rep_b="z_bio", t=2.0)
        if (not is_dist) or rank == 0:
            metrics.update({f"test/au/{k}": v for k, v in au.items()})

    if out_dir is not None and rank == 0:
        payload = {
            "time": time.time(),
            "epoch": epoch,
            "metrics": metrics,
        }
        path = os.path.join(out_dir, "metrics.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    return metrics
