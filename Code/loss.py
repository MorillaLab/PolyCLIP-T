from libs import *
from typing import Optional, Dict, List

# Ontology
def split_terms_safe(x):
    if x is None:
        return []
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none", "na", "."}:
        return []
    for sep in [";", ","]:
        x = x.replace(sep, "|")
    return [t.strip() for t in x.split("|") if t.strip()]


def ontology_terms_from_sample(
    onto_sample: dict,
    keys=("GO_BP", "GO_MF", "GO_CC", "KEGG", "HPO"),
):
    terms = {}
    for k in keys:
        terms[k] = set(split_terms_safe(onto_sample.get(k, "")))
    return terms


def weighted_jaccard_ontology(
    onto_i: dict,
    onto_j: dict,
    weights: Optional[Dict] = None,
):
    if weights is None:
        weights = {
            "GO_BP": 1.0,
            "GO_MF": 0.5,
            "GO_CC": 0.5,
            "KEGG": 1.0,
            "HPO": 1.0,
        }

    ti = ontology_terms_from_sample(onto_i, keys=weights.keys())
    tj = ontology_terms_from_sample(onto_j, keys=weights.keys())

    num = 0.0
    den = 0.0

    for k, w in weights.items():
        ai = ti[k]
        aj = tj[k]

        if len(ai) == 0 and len(aj) == 0:
            s = 0.0
        elif len(ai) == 0 or len(aj) == 0:
            s = 0.0
        else:
            s = len(ai & aj) / max(1, len(ai | aj))

        num += w * s
        den += w

    return num / max(den, 1e-8)

def build_ontology_target(
    ontology_batch: list[dict],
    device: torch.device,
    alpha: float = 0.5,
    weights: Optional[Dict] = None,
    zero_diag: bool = False,
):
    """
    Returns T in R^{B x B}, row-normalized.
    
    alpha:
      - 0.0 -> purely identity target
      - 1.0 -> purely ontology similarity target
    """
    B = len(ontology_batch)
    S = torch.zeros((B, B), dtype=torch.float32, device=device)

    for i in range(B):
        for j in range(B):
            S[i, j] = weighted_jaccard_ontology(
                ontology_batch[i],
                ontology_batch[j],
                weights=weights,
            )

    I = torch.eye(B, dtype=torch.float32, device=device)

    if zero_diag:
        S.fill_diagonal_(0.0)

    T = (1.0 - alpha) * I + alpha * S
    T = T / T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return T
#end ontology



def l2norm(x, dim=1, eps=1e-8):
    norm = torch.norm(x, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm


@torch.no_grad()
def _encode_fused_from_batch(teacher, batch, device):
    seq_refs = batch.get("seq_refs", None)
    seq_alts = batch.get("seq_alts", None)
    texts = batch.get("texts", None)

    bios = batch["bios"].to(device)

    ref_emb = batch.get("ref_emb", None)
    alt_emb = batch.get("alt_emb", None)
    text_emb = batch.get("text_emb", None)

    if ref_emb is not None:
        ref_emb = ref_emb.to(device)

    if alt_emb is not None:
        alt_emb = alt_emb.to(device)

    if text_emb is not None:
        text_emb = text_emb.to(device)

    h = teacher.encode_fused(
        seq_ref=seq_refs,
        seq_alt=seq_alts,
        bio_feats=bios,
        texts=texts,
        ref_emb=ref_emb,
        alt_emb=alt_emb,
        text_emb=text_emb,
    )
    return h

def collate_fn_refalt_ontology(batch):
    idxs = [b["idx"] for b in batch]

    out = {
        "idxs": idxs,
        "bios": torch.stack([b["bio"] for b in batch], dim=0),
    }

    # ---------------------------------------------------------
    # Raw sequence fields
    # ---------------------------------------------------------
    if "seq_ref" in batch[0]:
        out["seq_refs"] = [b["seq_ref"] for b in batch]

    if "seq_alt" in batch[0]:
        out["seq_alts"] = [b["seq_alt"] for b in batch]

    # ---------------------------------------------------------
    # Raw text field
    # ---------------------------------------------------------
    if "text" in batch[0]:
        out["texts"] = [b.get("text", "") for b in batch]

    # ---------------------------------------------------------
    # Offline sequence embeddings
    # ---------------------------------------------------------
    if "ref_emb" in batch[0]:
        out["ref_emb"] = torch.stack(
            [
                b["ref_emb"] if torch.is_tensor(b["ref_emb"]) else torch.tensor(b["ref_emb"], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        )

    if "alt_emb" in batch[0]:
        out["alt_emb"] = torch.stack(
            [
                b["alt_emb"] if torch.is_tensor(b["alt_emb"]) else torch.tensor(b["alt_emb"], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        )

    # ---------------------------------------------------------
    # Offline text embeddings
    # ---------------------------------------------------------
    if "text_emb" in batch[0]:
        out["text_emb"] = torch.stack(
            [
                b["text_emb"] if torch.is_tensor(b["text_emb"]) else torch.tensor(b["text_emb"], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        )

    # ---------------------------------------------------------
    # Optional label
    # ---------------------------------------------------------
    if "label" in batch[0]:
        out["labels"] = torch.stack(
            [
                b["label"] if torch.is_tensor(b["label"]) else torch.tensor(b["label"])
                for b in batch
            ],
            dim=0,
        )

    # ---------------------------------------------------------
    # Optional ontology / oncology / meta
    # ---------------------------------------------------------
    if "ontology" in batch[0]:
        out["ontology"] = [b.get("ontology", {}) for b in batch]

    if "oncology" in batch[0]:
        out["oncology"] = [b.get("oncology", {}) for b in batch]

    if "meta" in batch[0]:
        out["meta"] = [b.get("meta", {}) for b in batch]

    return out


@torch.no_grad()
def _pairwise_sqeuclidean(X: torch.Tensor):
    XX = (X**2).sum(1, keepdim=True)
    D2 = XX + XX.T - 2 * (X @ X.T)
    return torch.clamp(D2, min=0)

@torch.no_grad()
def knn_affinity(H: torch.Tensor, k: int = 15, sigma: Optional[float] = None):
    D2 = _pairwise_sqeuclidean(H)
    if sigma is None:
        mask = ~torch.eye(D2.size(0), dtype=torch.bool, device=H.device)
        sigma = torch.sqrt(torch.median(D2[mask]) + 1e-8).item()
    K = torch.exp(-D2 / (2 * (sigma**2) + 1e-8))
    N = K.size(0)
    K.fill_diagonal_(0.0)
    idx = torch.topk(K, k=k, dim=1).indices
    A = torch.zeros_like(K)
    rows = torch.arange(N, device=H.device).unsqueeze(1).expand_as(idx)
    A[rows, idx] = K[rows, idx]
    A = torch.maximum(A, A.T)
    return A



def tda_pairwise_regularizer(
    H_current: torch.Tensor,
    D_target: torch.Tensor,
    k: int = 15,
    t: int = 3,
    pair_per_row: int = 64,
    robust: bool = False,
):
    # H_current = F.normalize(H_current, p=2, dim=1)
    D_curr = diffusion_distance(H_current, k=k, t=t)
    D_curr = torch.clamp(D_curr, 0, 1e6)

    if robust:
        D_curr = normalize_dist(D_curr)
        D_tgt = normalize_dist(D_target)
    else:
        D_curr = D_curr / (D_curr.max() + 1e-8)
        D_tgt = D_target / (D_target.max() + 1e-8)

    return F.mse_loss(D_curr.to(torch.float16), D_tgt.to(torch.float16))



@torch.no_grad()
def diffusion_distance(H: torch.Tensor, k: int = 15, t: int = 3):
    A = knn_affinity(H, k=k)
    deg = A.sum(1, keepdim=True) + 1e-8
    P = A / deg
    Pt = P.clone()
    for _ in range(t - 1):
        Pt = Pt @ P
    rsq = (Pt**2).sum(1, keepdim=True)
    D = rsq + rsq.T - 2 * (Pt @ Pt.T)
    return torch.clamp(D, min=0)

def normalize_dist(D: torch.Tensor, eps=1e-8):
    mask = ~torch.eye(D.size(0), dtype=torch.bool, device=D.device)
    med = torch.median(D[mask])
    return D / (med + eps)



def clip_like_loss(z_seq: torch.Tensor, z_bio: torch.Tensor, temperature: float = 0.5):
    a = l2norm(z_seq, 1); b = l2norm(z_bio, 1)
    logits = (a @ b.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


# def clip_like_loss_with_ontology(
#     z_seq: torch.Tensor,
#     z_bio: torch.Tensor,
#     ontology_batch: Optional[List[dict]] = None,
#     temperature: float = 0.07,
#     alpha_onto: float = 0.5,
#     onto_weights: Optional[Dict] = None,
#     zero_diag: bool = True,
# ):
#     a = l2norm(z_seq, 1)
#     b = l2norm(z_bio, 1)

#     logits = (a @ b.T) / temperature
#     B = a.size(0)
#     device = a.device

#     if ontology_batch is None:
#         targets = torch.eye(B, dtype=torch.float32, device=device)
#     else:
#         targets = build_ontology_target(
#             ontology_batch=ontology_batch,
#             device=device,
#             alpha=alpha_onto,
#             weights=onto_weights,
#             zero_diag=zero_diag,
#         )

#     loss_ab = soft_cross_entropy(logits, targets)
#     loss_ba = soft_cross_entropy(logits.T, targets.T)

#     return 0.5 * (loss_ab + loss_ba)


def clip_like_loss_with_ontology(
    z_seq: torch.Tensor,
    z_bio: torch.Tensor,
    ontology_batch: Optional[List[dict]] = None,
    temperature: float = 0.07,
    alpha_onto: float = 0.5,
    onto_weights: Optional[Dict] = None,
    zero_diag: bool = True,
):
    a = l2norm(z_seq, 1)
    b = l2norm(z_bio, 1)

    logits = (a @ b.T) / temperature
    device = a.device
    B = a.size(0)

    if ontology_batch is None:
        targets_sb = torch.eye(B, dtype=torch.float32, device=device)
    else:
        targets_sb = build_ontology_target(
            ontology_batch=ontology_batch,
            device=device,
            alpha=alpha_onto,
            weights=onto_weights,
            zero_diag=zero_diag,
        )

    # --- SB (seq → bio) ---
    loss_sb = soft_cross_entropy(logits, targets_sb)

    # --- BS (bio → seq) ---
    targets_bs = targets_sb.T
    targets_bs = targets_bs / targets_bs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    loss_bs = soft_cross_entropy(logits.T, targets_bs)

    return 0.5 * (loss_sb + loss_bs)

def build_variant_bio_ontology_target(
    ontology_batch: Optional[List[dict]],
    variant_ids: torch.Tensor,
    bio_group_ids: torch.Tensor,
    device: torch.device,
    w_pair: float = 1.0,
    w_variant: float = 1.0,
    w_bio: float = 0.5,
    w_onto: float = 0.3,
    onto_weights: Optional[Dict] = None,
    zero_diag_onto: bool = True,
) -> torch.Tensor:
    """
    Build a soft target matrix T in R^{B x B}, row-normalized.

    Components:
      - exact pair identity
      - same variant
      - same bio group
      - ontology similarity

    Returns:
      T: [B, B], each row sums to 1
    """
    B = variant_ids.shape[0]

    # Exact pair
    I = torch.eye(B, dtype=torch.float32, device=device)

    # Same variant mask
    variant_ids = variant_ids.to(device)
    V = (variant_ids[:, None] == variant_ids[None, :]).float()

    # Same bio-group mask
    bio_group_ids = bio_group_ids.to(device)
    Bmask = (bio_group_ids[:, None] == bio_group_ids[None, :]).float()

    # Ontology similarity
    S = torch.zeros((B, B), dtype=torch.float32, device=device)

    if ontology_batch is not None and w_onto > 0:
        for i in range(B):
            for j in range(B):
                S[i, j] = weighted_jaccard_ontology(
                    ontology_batch[i],
                    ontology_batch[j],
                    weights=onto_weights,
                )
        if zero_diag_onto:
            S.fill_diagonal_(0.0)

    # Weighted sum
    T = (
        w_pair * I
        + w_variant * V
        + w_bio * Bmask
        + w_onto * S
    )

    # Safety: if a row is all-zero, fallback to identity
    row_sums = T.sum(dim=1, keepdim=True)
    zero_rows = (row_sums.squeeze(1) <= 0)

    if zero_rows.any():
        T[zero_rows] = I[zero_rows]
        row_sums = T.sum(dim=1, keepdim=True)

    T = T / row_sums.clamp_min(1e-8)
    return T


def clip_like_loss_with_variant_bio_ontology(
    z_seq: torch.Tensor,
    z_bio: torch.Tensor,
    ontology_batch: Optional[List[dict]],
    variant_ids: torch.Tensor,
    bio_group_ids: torch.Tensor,
    temperature: float = 0.1,
    w_pair: float = 1.0,
    w_variant: float = 1.0,
    w_bio: float = 0.5,
    w_onto: float = 0.3,
    onto_weights: Optional[Dict] = None,
    zero_diag_onto: bool = True,
):
    """
    Symmetric soft-target contrastive loss combining:
      - exact pair
      - same variant
      - same bio group
      - ontology similarity

    Args:
      z_seq: [B, D]
      z_bio: [B, D]
      ontology_batch: list of ontology dicts, length B
      variant_ids: [B]
      bio_group_ids: [B]
      temperature: scalar > 0
      w_pair, w_variant, w_bio, w_onto: weights for target construction
      onto_weights: optional weights passed to weighted_jaccard_ontology
      zero_diag_onto: if True, removes ontology self-similarity before mixing

    Returns:
      total_loss, loss_ab, loss_ba, targets, logits
    """
    device = z_seq.device

    a = l2norm(z_seq, dim=1)
    b = l2norm(z_bio, dim=1)

    logits = (a @ b.T) / temperature

    targets = build_variant_bio_ontology_target(
        ontology_batch=ontology_batch,
        variant_ids=variant_ids,
        bio_group_ids=bio_group_ids,
        device=device,
        w_pair=w_pair,
        w_variant=w_variant,
        w_bio=w_bio,
        w_onto=w_onto,
        onto_weights=onto_weights,
        zero_diag_onto=zero_diag_onto,
    )

    loss_ab = soft_cross_entropy(logits, targets)
    loss_ba = soft_cross_entropy(logits.T, targets.T)
    loss = 0.5 * (loss_ab + loss_ba)

    return loss, loss_ab, loss_ba, targets, logits

def clip_like_loss_supervised(z_seq: torch.Tensor, z_bio: torch.Tensor, labels: np.ndarray, temperature=0.07):
    a = l2norm(z_seq, 1)
    b = l2norm(z_bio, 1)
    logits = (a @ b.T) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(a.device)
    mask = mask * (1 - torch.eye(mask.size(0), device=a.device))

    def one_direction_loss(logits, mask):
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

    return 0.5 * (one_direction_loss(logits, mask) + one_direction_loss(logits.T, mask))

def sample_pairs(N: int, per_row: int = 64, device="cpu"):
    i = torch.arange(N, device=device).repeat_interleave(per_row)
    j = torch.randint(0, N, (N * per_row,), device=device)
    keep = i != j
    return i[keep], j[keep]


def build_soft_target_from_scores(
    score_mat: torch.Tensor,
    target_temp: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert a score matrix into a row-stochastic soft target matrix.

    Smaller target_temp -> sharper targets.
    """
    score_mat = score_mat / max(target_temp, eps)
    score_mat = score_mat - score_mat.max(dim=1, keepdim=True).values
    return torch.softmax(score_mat, dim=1)


def build_ontology_target_v2(
    ontology_batch: list[dict],
    device: torch.device,
    alpha_onto: float = 0.5,
    weights: Optional[Dict] = None,
    zero_diag: bool = True,
    target_temp: float = 0.25,
) -> torch.Tensor:
    """
    Build soft ontology-guided targets.

    alpha_onto:
      0.0 -> pure identity target
      1.0 -> pure ontology-based target
    """
    B = len(ontology_batch)
    S = torch.zeros((B, B), dtype=torch.float32, device=device)

    for i in range(B):
        for j in range(B):
            S[i, j] = weighted_jaccard_ontology(
                ontology_batch[i],
                ontology_batch[j],
                weights=weights,
            )

    if zero_diag:
        S.fill_diagonal_(0.0)

    I = torch.eye(B, dtype=torch.float32, device=device)

    # score matrix before softmax
    score = (1.0 - alpha_onto) * I + alpha_onto * S

    T = build_soft_target_from_scores(
        score_mat=score,
        target_temp=target_temp,
    )
    return T


def clip_like_loss_with_ontology_v2(
    z_seq: torch.Tensor,
    z_bio: torch.Tensor,
    ontology_batch: Optional[List[dict]] = None,
    temperature: float = 0.07,
    alpha_onto: float = 0.5,
    onto_weights: Optional[Dict] = None,
    zero_diag: bool = True,
    target_temp: float = 0.25,
) -> torch.Tensor:
    """
    Symmetric CLIP-like loss with ontology-guided soft targets.

    If ontology_batch is None:
      reduces to standard CLIP with identity targets.
    """
    a = F.normalize(z_seq, dim=1)
    b = F.normalize(z_bio, dim=1)

    logits = (a @ b.T) / temperature
    device = a.device
    B = a.size(0)

    if ontology_batch is None:
        targets_ab = torch.eye(B, dtype=torch.float32, device=device)
    else:
        targets_ab = build_ontology_target_v2(
            ontology_batch=ontology_batch,
            device=device,
            alpha_onto=alpha_onto,
            weights=onto_weights,
            zero_diag=zero_diag,
            target_temp=target_temp,
        )

    targets_ba = targets_ab.T
    targets_ba = targets_ba / targets_ba.sum(dim=1, keepdim=True).clamp_min(1e-8)

    loss_ab = soft_cross_entropy(logits, targets_ab)
    loss_ba = soft_cross_entropy(logits.T, targets_ba)

    return 0.5 * (loss_ab + loss_ba)



def hierarchical_build_and_save_diffusion_matrix(
    model,
    dataset,
    cfg,
    out_path="D_target.npy",
    block_size=5000,
    save_dtype="float16",
):
    """
    Build a hierarchical diffusion-distance matrix and save it as a memmap.

    Supports:
      - raw seq_ref / seq_alt
      - raw text
      - offline ref_emb / alt_emb
      - offline text_emb
    """
    device = cfg.device
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    teacher = model.teacher if hasattr(model, "teacher") else model
    teacher.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_refalt_ontology,
        pin_memory=False,
    )

    # ---------------------------------------------------------
    # STEP 1 - Encode fused embeddings for all samples
    # ---------------------------------------------------------
    print("Encoding fused embeddings for all variants...", flush=True)
    h_all = []

    with torch.no_grad():
        for batch in loader:
            h = _encode_fused_from_batch(teacher, batch, device)
            h_all.append(h.detach().cpu())

    H_all = torch.cat(h_all, dim=0)
    N = H_all.shape[0]
    print(f"Encoded {N:,} variants total", flush=True)

    # ---------------------------------------------------------
    # STEP 2 - Initialize disk-backed matrix
    # ---------------------------------------------------------
    D_mem = np.memmap(
        out_path,
        dtype=save_dtype,
        mode="w+",
        shape=(N, N),
    )
    print(f"Initialized memmap file {out_path} with shape {D_mem.shape}", flush=True)

    # ---------------------------------------------------------
    # STEP 3 - Intra-block diffusion distances
    # ---------------------------------------------------------
    print("Computing intra-block diffusion distances...", flush=True)
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        H_i = H_all[i:i_end].to(device)

        if H_i.shape[0] == 1:
            D_mem[i:i_end, i:i_end] = np.zeros((1, 1), dtype=save_dtype)
            print(f"Intra-block diffusion {i:,}-{i_end:,} computed (singleton)", flush=True)
            continue

        k_i = min(cfg.knn_k, H_i.shape[0] - 1)
        A_i = knn_affinity(H_i, k=k_i)
        deg_i = A_i.sum(dim=1, keepdim=True) + 1e-8
        P_i = A_i / deg_i

        Pti = P_i.clone()
        for _ in range(max(int(cfg.diff_t) - 1, 0)):
            Pti = Pti @ P_i

        rsq_i = (Pti ** 2).sum(dim=1, keepdim=True)
        D_intra = (rsq_i + rsq_i.T - 2.0 * (Pti @ Pti.T)).detach().cpu().numpy()
        D_intra = np.maximum(D_intra, 0.0).astype(save_dtype, copy=False)

        D_mem[i:i_end, i:i_end] = D_intra
        print(f"Intra-block diffusion {i:,}-{i_end:,} computed", flush=True)

    # ---------------------------------------------------------
    # STEP 4 - Block centroids + centroid diffusion
    # ---------------------------------------------------------
    print("Computing inter-block diffusion at centroid level...", flush=True)
    block_centroids = []
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        block_centroids.append(H_all[i:i_end].mean(dim=0, keepdim=True))

    C = torch.cat(block_centroids, dim=0).to(device)
    n_blocks = C.shape[0]

    if n_blocks > 1:
        k_c = min(cfg.knn_k, n_blocks - 1)
        A_c = knn_affinity(C, k=k_c)
        deg_c = A_c.sum(dim=1, keepdim=True) + 1e-8
        P_c = A_c / deg_c

        Ptc = P_c.clone()
        for _ in range(max(int(cfg.diff_t) - 1, 0)):
            Ptc = Ptc @ P_c

        rsq_c = (Ptc ** 2).sum(dim=1, keepdim=True)
        D_centroid = (rsq_c + rsq_c.T - 2.0 * (Ptc @ Ptc.T)).detach().cpu().numpy()
        D_centroid = np.maximum(D_centroid, 0.0).astype(save_dtype, copy=False)
    else:
        D_centroid = np.zeros((1, 1), dtype=save_dtype)

    # ---------------------------------------------------------
    # STEP 5 - Fill inter-block regions with centroid distances
    # ---------------------------------------------------------
    print("Propagating inter-block diffusion distances...", flush=True)
    for bi in range(n_blocks):
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, N)

        for bj in range(bi + 1, n_blocks):
            j_start = bj * block_size
            j_end = min((bj + 1) * block_size, N)

            dist = D_centroid[bi, bj]
            D_mem[i_start:i_end, j_start:j_end] = dist
            D_mem[j_start:j_end, i_start:i_end] = dist

    D_mem.flush()
    del D_mem

    print(f"Hierarchical diffusion matrix saved to {out_path}", flush=True)
    return out_path

def hierarchical_build_and_save_diffusion_matrix_distributed(
    model,
    dataset,
    cfg,
    out_path="D_target.npy",
    block_size=5000,
    save_dtype="float16",
    rank=0,
    world_size=1,
):
    """
    Distributed hierarchical diffusion-distance builder.

    Supports:
      - raw seq_ref / seq_alt
      - raw text
      - offline ref_emb / alt_emb
      - offline text_emb
    """
    import gc
    import torch
    import numpy as np
    import os
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    device = cfg.device
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    teacher = model.teacher if hasattr(model, "teacher") else model
    teacher.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_refalt_ontology,
        pin_memory=False,
    )

    try:
        # ---------------------------------------------------------
        # STEP 1 - Build the same global embedding matrix on all ranks
        # ---------------------------------------------------------
        if rank == 0:
            print("[diffusion] Encoding fused embeddings on all ranks...", flush=True)

        h_all = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                h = _encode_fused_from_batch(teacher, batch, device)
                h_all.append(h.detach().cpu())

                # free per-batch tensors early
                del h, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        H_all = torch.cat(h_all, dim=0)
        N = H_all.shape[0]

        # h_all list no longer needed after concat
        del h_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if rank == 0:
            print(f"[diffusion] Encoded {N:,} variants total", flush=True)

        # ---------------------------------------------------------
        # STEP 2 - Create/open shared memmap
        # ---------------------------------------------------------
        if rank == 0:
            D_mem = np.memmap(out_path, dtype=save_dtype, mode="w+", shape=(N, N))
            D_mem.flush()
            del D_mem

        dist.barrier()

        D_mem = np.memmap(out_path, dtype=save_dtype, mode="r+", shape=(N, N))
        n_blocks = (N + block_size - 1) // block_size

        # ---------------------------------------------------------
        # STEP 3 - Precompute block centroids on every rank
        # ---------------------------------------------------------
        block_centroids = []
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            block_centroids.append(H_all[i:i_end].mean(dim=0, keepdim=True))

        C = torch.cat(block_centroids, dim=0).to(device)

        # block_centroids list no longer needed
        del block_centroids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if n_blocks > 1:
            k_c = min(cfg.knn_k, n_blocks - 1)
            A_c = knn_affinity(C, k=k_c)
            deg_c = A_c.sum(dim=1, keepdim=True) + 1e-8
            P_c = A_c / deg_c

            Ptc = P_c.clone()
            for _ in range(max(int(cfg.diff_t) - 1, 0)):
                Ptc = Ptc @ P_c

            rsq_c = (Ptc ** 2).sum(dim=1, keepdim=True)
            D_centroid = (rsq_c + rsq_c.T - 2.0 * (Ptc @ Ptc.T)).detach().cpu().numpy()
            D_centroid = np.maximum(D_centroid, 0.0).astype(save_dtype, copy=False)

            del A_c, deg_c, P_c, Ptc, rsq_c
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            D_centroid = np.zeros((1, 1), dtype=save_dtype)

        # C no longer needed after D_centroid has been built
        del C
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # STEP 4 - Shard block-rows across ranks
        # ---------------------------------------------------------
        my_block_rows = list(range(rank, n_blocks, world_size))
        print(f"[rank {rank}] responsible for block rows: {my_block_rows}", flush=True)

        for bi in my_block_rows:
            i_start = bi * block_size
            i_end = min((bi + 1) * block_size, N)
            H_i = H_all[i_start:i_end].to(device)

            # Intra-block diffusion
            if H_i.shape[0] == 1:
                D_mem[i_start:i_end, i_start:i_end] = np.zeros((1, 1), dtype=save_dtype)
            else:
                k_i = min(cfg.knn_k, H_i.shape[0] - 1)
                A_i = knn_affinity(H_i, k=k_i)
                deg_i = A_i.sum(dim=1, keepdim=True) + 1e-8
                P_i = A_i / deg_i

                Pti = P_i.clone()
                for _ in range(max(int(cfg.diff_t) - 1, 0)):
                    Pti = Pti @ P_i

                rsq_i = (Pti ** 2).sum(dim=1, keepdim=True)
                D_intra = (rsq_i + rsq_i.T - 2.0 * (Pti @ Pti.T)).detach().cpu().numpy()
                D_intra = np.maximum(D_intra, 0.0).astype(save_dtype, copy=False)
                D_mem[i_start:i_end, i_start:i_end] = D_intra

                del A_i, deg_i, P_i, Pti, rsq_i, D_intra

            # Inter-block diffusion from centroid topology
            for bj in range(bi + 1, n_blocks):
                j_start = bj * block_size
                j_end = min((bj + 1) * block_size, N)

                coarse_dist = D_centroid[bi, bj]
                D_mem[i_start:i_end, j_start:j_end] = coarse_dist
                D_mem[j_start:j_end, i_start:i_end] = coarse_dist

            D_mem.flush()
            print(f"[rank {rank}] finished block row {bi} ({i_start}:{i_end})", flush=True)

            del H_i
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dist.barrier()
        D_mem.flush()

        if rank == 0:
            print(f"[diffusion] Distributed hierarchical diffusion matrix saved to {out_path}", flush=True)

        return out_path

    finally:
        # ---------------------------------------------------------
        # FINAL CLEANUP
        # ---------------------------------------------------------
        try:
            if "D_mem" in locals():
                D_mem.flush()
                del D_mem
        except Exception:
            pass

        for name in [
            "loader",
            "teacher",
            "H_all",
            "D_centroid",
            "my_block_rows",
            "dataset",
            "model",
        ]:
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

# def hierarchical_build_and_save_diffusion_matrix_distributed(
#     model,
#     dataset,
#     cfg,
#     out_path="D_target.npy",
#     block_size=5000,
#     save_dtype="float16",
#     rank=0,
#     world_size=1,
# ):
#     """
#     Distributed hierarchical diffusion-distance builder.

#     Supports:
#       - raw seq_ref / seq_alt
#       - raw text
#       - offline ref_emb / alt_emb
#       - offline text_emb
#     """
#     device = cfg.device
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

#     teacher = model.teacher if hasattr(model, "teacher") else model
#     teacher.eval()

#     loader = DataLoader(
#         dataset,
#         batch_size=cfg.batch_size,
#         shuffle=False,
#         num_workers=cfg.num_workers,
#         collate_fn=collate_fn_refalt_ontology,
#         pin_memory=False,
#     )

#     # ---------------------------------------------------------
#     # STEP 1 - Build the same global embedding matrix on all ranks
#     # ---------------------------------------------------------
#     if rank == 0:
#         print("[diffusion] Encoding fused embeddings on all ranks...", flush=True)

#     h_all = []
#     with torch.no_grad():
#         for batch in loader:
#             h = _encode_fused_from_batch(teacher, batch, device)
#             h_all.append(h.detach().cpu())

#     H_all = torch.cat(h_all, dim=0)
#     N = H_all.shape[0]

#     if rank == 0:
#         print(f"[diffusion] Encoded {N:,} variants total", flush=True)

#     # ---------------------------------------------------------
#     # STEP 2 - Create/open shared memmap
#     # ---------------------------------------------------------
#     if rank == 0:
#         D_mem = np.memmap(out_path, dtype=save_dtype, mode="w+", shape=(N, N))
#         D_mem.flush()
#         del D_mem

#     dist.barrier()

#     D_mem = np.memmap(out_path, dtype=save_dtype, mode="r+", shape=(N, N))
#     n_blocks = (N + block_size - 1) // block_size

#     # ---------------------------------------------------------
#     # STEP 3 - Precompute block centroids on every rank
#     # ---------------------------------------------------------
#     block_centroids = []
#     for i in range(0, N, block_size):
#         i_end = min(i + block_size, N)
#         block_centroids.append(H_all[i:i_end].mean(dim=0, keepdim=True))

#     C = torch.cat(block_centroids, dim=0).to(device)

#     if n_blocks > 1:
#         k_c = min(cfg.knn_k, n_blocks - 1)
#         A_c = knn_affinity(C, k=k_c)
#         deg_c = A_c.sum(dim=1, keepdim=True) + 1e-8
#         P_c = A_c / deg_c

#         Ptc = P_c.clone()
#         for _ in range(max(int(cfg.diff_t) - 1, 0)):
#             Ptc = Ptc @ P_c

#         rsq_c = (Ptc ** 2).sum(dim=1, keepdim=True)
#         D_centroid = (rsq_c + rsq_c.T - 2.0 * (Ptc @ Ptc.T)).detach().cpu().numpy()
#         D_centroid = np.maximum(D_centroid, 0.0).astype(save_dtype, copy=False)
#     else:
#         D_centroid = np.zeros((1, 1), dtype=save_dtype)

#     # ---------------------------------------------------------
#     # STEP 4 - Shard block-rows across ranks
#     # ---------------------------------------------------------
#     my_block_rows = list(range(rank, n_blocks, world_size))
#     print(f"[rank {rank}] responsible for block rows: {my_block_rows}", flush=True)

#     for bi in my_block_rows:
#         i_start = bi * block_size
#         i_end = min((bi + 1) * block_size, N)
#         H_i = H_all[i_start:i_end].to(device)

#         # Intra-block diffusion
#         if H_i.shape[0] == 1:
#             D_mem[i_start:i_end, i_start:i_end] = np.zeros((1, 1), dtype=save_dtype)
#         else:
#             k_i = min(cfg.knn_k, H_i.shape[0] - 1)
#             A_i = knn_affinity(H_i, k=k_i)
#             deg_i = A_i.sum(dim=1, keepdim=True) + 1e-8
#             P_i = A_i / deg_i

#             Pti = P_i.clone()
#             for _ in range(max(int(cfg.diff_t) - 1, 0)):
#                 Pti = Pti @ P_i

#             rsq_i = (Pti ** 2).sum(dim=1, keepdim=True)
#             D_intra = (rsq_i + rsq_i.T - 2.0 * (Pti @ Pti.T)).detach().cpu().numpy()
#             D_intra = np.maximum(D_intra, 0.0).astype(save_dtype, copy=False)
#             D_mem[i_start:i_end, i_start:i_end] = D_intra

#         # Inter-block diffusion from centroid topology
#         for bj in range(bi + 1, n_blocks):
#             j_start = bj * block_size
#             j_end = min((bj + 1) * block_size, N)

#             coarse_dist = D_centroid[bi, bj]
#             D_mem[i_start:i_end, j_start:j_end] = coarse_dist
#             D_mem[j_start:j_end, i_start:i_end] = coarse_dist

#         D_mem.flush()
#         print(f"[rank {rank}] finished block row {bi} ({i_start}:{i_end})", flush=True)

#     dist.barrier()
#     D_mem.flush()
#     del D_mem

#     if rank == 0:
#         print(f"[diffusion] Distributed hierarchical diffusion matrix saved to {out_path}", flush=True)

#     return out_path
