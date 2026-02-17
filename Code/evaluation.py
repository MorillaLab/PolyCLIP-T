from __future__ import annotations

from libs import *
from config import Config
from models import TCL_TDA_Model_RefAlt
from loss import collate_fn_refalt
from utils import rank0_print, get_data_all, set_seed


# ============================================================
# Encoding helpers
# ============================================================
@torch.no_grad()
def encode_dataset(
    cfg: Config,
    model: TCL_TDA_Model_RefAlt,
    dataset: Dataset,
    rep: str = "z_bio",          # "z_seq" | "z_bio" | "h_tda" | "fused"
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    """
    Returns:
      idxs_all: (N,) int64
      emb_all:  (N,D) float32 (cpu)
      labels_all: (N,) int64 (cpu)
    Notes:
      - labels are whatever your dataset returns as the last item in __getitem__ (currently label)
      - rep:
          z_seq/z_bio from encode_views
          h_tda from encode_views
          fused from encode_fused
    """
    model.eval()

    bs = int(batch_size if batch_size is not None else cfg.batch_size)
    nw = int(num_workers if num_workers is not None else getattr(cfg, "num_workers", 0))

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn_refalt,
        pin_memory=False,
    )

    idxs_list, emb_list, lab_list = [], [], []

    for idxs, seq_refs, seq_alts, bios, label in loader:
        bios = bios.to(cfg.device)

        if rep == "fused":
            H = model.encode_fused(seq_refs, seq_alts, bios)          # (B,shared_dim)
            E = H
        else:
            z_seq, z_bio, h_tda, *_ = model.encode_views(seq_refs, seq_alts, bios)
            if rep == "z_seq":
                E = z_seq
            elif rep == "z_bio":
                E = z_bio
            elif rep == "h_tda":
                E = h_tda
            else:
                raise ValueError(f"Unknown rep='{rep}'. Choose: z_seq, z_bio, h_tda, fused")

        idxs_list.append(torch.tensor(idxs, dtype=torch.long))
        emb_list.append(E.detach().float().cpu())
        lab_list.append(label.detach().long().cpu())

    idxs_all = torch.cat(idxs_list, dim=0)
    emb_all = torch.cat(emb_list, dim=0)
    labels_all = torch.cat(lab_list, dim=0)
    return idxs_all, emb_all, labels_all


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def clip_retrieval_metrics(
    cfg: Config,
    model: TCL_TDA_Model_RefAlt,
    dataset: Dataset,
    temperature: float | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    """
    Computes CLIP retrieval accuracy on the dataset:
      - seq -> bio : does the correct paired sample (same index within batch) rank top-1/top-5?
      - bio -> seq : symmetric

    This is a strict CLIP sanity check (diagonal alignment).
    """
    model.eval()
    bs = int(batch_size if batch_size is not None else cfg.batch_size)
    nw = int(num_workers if num_workers is not None else getattr(cfg, "num_workers", 0))
    tau = float(temperature if temperature is not None else getattr(cfg, "temperature", 0.07))

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn_refalt,
        pin_memory=False,
    )

    top1_i2t = 0
    top5_i2t = 0
    top1_t2i = 0
    top5_t2i = 0
    total = 0

    for _, seq_refs, seq_alts, bios, _label in loader:
        bios = bios.to(cfg.device)
        z_seq, z_bio, *_ = model.encode_views(seq_refs, seq_alts, bios)  # (B,D)

        logits = (F.normalize(z_seq, dim=1) @ F.normalize(z_bio, dim=1).T) / tau  # (B,B)
        B = logits.size(0)
        gt = torch.arange(B, device=logits.device)

        # seq -> bio
        pred1 = logits.argmax(dim=1)
        top1_i2t += (pred1 == gt).sum().item()

        top5 = logits.topk(k=min(5, B), dim=1).indices
        top5_i2t += (top5 == gt[:, None]).any(dim=1).sum().item()

        # bio -> seq
        logits_t = logits.T
        pred1_t = logits_t.argmax(dim=1)
        top1_t2i += (pred1_t == gt).sum().item()

        top5_t = logits_t.topk(k=min(5, B), dim=1).indices
        top5_t2i += (top5_t == gt[:, None]).any(dim=1).sum().item()

        total += B

    return {
        "clip_i2t_top1": top1_i2t / max(1, total),
        "clip_i2t_top5": top5_i2t / max(1, total),
        "clip_t2i_top1": top1_t2i / max(1, total),
        "clip_t2i_top5": top5_t2i / max(1, total),
        "n": int(total),
    }


@torch.no_grad()
def label_knn_purity(
    emb: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    unlabeled_val: int = -1,
    chunk: int = 2048,
):
    """
    Simple label-purity diagnostic:
      For each labeled sample i, look at its k nearest neighbors (cosine),
      report fraction with same label.

    emb: (N,D) cpu float32
    labels: (N,) cpu int64
    """
    emb = F.normalize(emb, dim=1)
    labels = labels.view(-1).long()
    N = emb.size(0)

    labeled_mask = labels != int(unlabeled_val)
    idx_l = torch.where(labeled_mask)[0]
    if idx_l.numel() == 0:
        return {"knn_purity": None, "n_labeled": 0}

    purity_sum = 0.0
    count = 0

    # brute-force cosine in chunks (avoids NxN full matrix)
    for start in range(0, idx_l.numel(), chunk):
        ii = idx_l[start:start + chunk]  # indices in [0..N)
        Q = emb[ii]                      # (C,D)
        sims = Q @ emb.T                 # (C,N)

        # exclude self from neighbors
        sims[torch.arange(sims.size(0)), ii] = -1e9

        nn = sims.topk(k=min(k, N - 1), dim=1).indices  # (C,k)
        same = (labels[nn] == labels[ii].unsqueeze(1)).float()  # (C,k)
        purity_sum += same.mean(dim=1).sum().item()
        count += ii.numel()

    return {"knn_purity": purity_sum / max(1, count), "n_labeled": int(count)}


# ============================================================
# Main evaluation entrypoint
# ============================================================
def load_checkpoint_model(cfg: Config, ckpt_path: str) -> TCL_TDA_Model_RefAlt:
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model = TCL_TDA_Model_RefAlt(cfg).to(cfg.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt or latest_checkpoint.pt")
    parser.add_argument("--rep", type=str, default="z_bio", choices=["z_seq", "z_bio", "h_tda", "fused"])
    parser.add_argument("--test_family", type=str, default="ROG")
    parser.add_argument("--numeric", type=str, default="../verif_data/ML_NUMERIC.tsv")
    parser.add_argument("--seq", type=str, default="../verif_data/ML_INPUT_WITH_SEQ.tsv")
    parser.add_argument("--out_dir", type=str, default="./eval_out")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    cfg = Config()
    cfg.numeric_path = args.numeric
    cfg.seq_path = args.seq
    set_seed(int(getattr(cfg, "seed", 42)))

    train_ds, test_ds, all_label = get_data_all(
        numeric=cfg.numeric_path,
        seq=cfg.seq_path,
        test_family=args.test_family
    )

    model = load_checkpoint_model(cfg, args.ckpt)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- CLIP retrieval sanity (only meaningful for z_seq/z_bio)
    if args.rep in ("z_seq", "z_bio"):
        m_train = clip_retrieval_metrics(cfg, model, train_ds)
        m_test = clip_retrieval_metrics(cfg, model, test_ds)
        rank0_print("[CLIP RETRIEVAL][TRAIN]", m_train)
        rank0_print("[CLIP RETRIEVAL][TEST ]", m_test)
        with open(os.path.join(args.out_dir, "clip_retrieval.json"), "w") as f:
            json.dump({"train": m_train, "test": m_test}, f, indent=2)

    # --- Encode embeddings and compute label purity
    idx_tr, emb_tr, lab_tr = encode_dataset(cfg, model, train_ds, rep=args.rep)
    idx_te, emb_te, lab_te = encode_dataset(cfg, model, test_ds, rep=args.rep)

    unl = int(getattr(cfg, "unlabeled_val", -1))
    purity_tr = label_knn_purity(emb_tr, lab_tr, k=args.k, unlabeled_val=unl)
    purity_te = label_knn_purity(emb_te, lab_te, k=args.k, unlabeled_val=unl)

    rank0_print(f"[KNN PURITY][TRAIN rep={args.rep} k={args.k}]", purity_tr)
    rank0_print(f"[KNN PURITY][TEST  rep={args.rep} k={args.k}]", purity_te)

    with open(os.path.join(args.out_dir, "knn_purity.json"), "w") as f:
        json.dump({"train": purity_tr, "test": purity_te}, f, indent=2)

    # --- Save embeddings (optional but useful for UMAP/HDBSCAN outside)
    torch.save(
        {"idx": idx_tr, "emb": emb_tr, "label": lab_tr},
        os.path.join(args.out_dir, f"train_{args.rep}_emb.pt")
    )
    torch.save(
        {"idx": idx_te, "emb": emb_te, "label": lab_te},
        os.path.join(args.out_dir, f"test_{args.rep}_emb.pt")
    )

    rank0_print("Saved embeddings to:", args.out_dir)


if __name__ == "__main__":
    main()
