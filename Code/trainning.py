# ========================= FILE: trainning.py =========================
from libs import *
from config import *
from model import *
from loss import *
from utils import *
import time


# ---------------------------------------------------------------------
# Distributed setup / cleanup
# ---------------------------------------------------------------------

def setup_distributed():
    hostname = socket.gethostname()
    pid = os.getpid()
    print(f"[{hostname} | pid={pid}] Initializing distributed training...", flush=True)

    try:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        import datetime

        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(hours=3),
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        print(
            f"[{hostname} | pid={pid}] Rank {rank}/{world_size} initialized "
            f"| local_rank={local_rank} | backend={backend}",
            flush=True,
        )
        return rank, world_size

    except Exception as e:
        print(f"[{hostname} | pid={pid}] ERROR in setup_distributed: {e}", flush=True)
        traceback.print_exc()
        raise

# def cleanup_distributed():
#     try:
#         if dist.is_available() and dist.is_initialized():
#             dist.barrier()
#             dist.destroy_process_group()
#             print("Process group destroyed", flush=True)
#     except Exception as e:
#         print(f"WARNING: Error destroying process group: {e}", flush=True)

def cleanup_distributed():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()   
            print("Process group destroyed", flush=True)
    except Exception as e:
        print(f"WARNING: Error destroying process group: {e}", flush=True)


# =========================================================
# Build model / optimizer / EMA
# =========================================================
# def build_model_optimizer_ema(cfg: Config):
#     model = TCL_TDA_Model_RefAlt(cfg).to(cfg.device)

#     if torch.cuda.is_available():
#         model = DDP(model, device_ids=[cfg.device.index], output_device=cfg.device.index)
#     else:
#         model = DDP(model)

#     ema = EMAWrapper(model.module, decay=cfg.ema_decay).to(cfg.device)

#     head_params = (
#         list(model.module.bio.parameters())
#         + list(model.module.fusion.parameters())
#         + list(model.module.seq_head.parameters())
#         + list(model.module.bio_head.parameters())
#         + list(model.module.mutation_encoder.parameters())
#     )

#     if getattr(cfg, "use_text", False):
#         head_params += list(model.module.text_head.parameters())

#     params = [{"params": head_params, "lr": cfg.lr_heads}]

#     if (not getattr(cfg, "seq_offline_embeddings", False)) and getattr(cfg, "dna_finetune", False):
#         params.append(
#             {
#                 "params": model.module.dna.parameters(),
#                 "lr": cfg.lr_dnabert,
#             }
#         )

#     if (
#         getattr(cfg, "use_text", False)
#         and (not getattr(cfg, "text_offline_embeddings", False))
#         and getattr(cfg, "text_finetune", False)
#     ):
#         params.append(
#             {
#                 "params": model.module.text_encoder.parameters(),
#                 "lr": cfg.lr_text,
#             }
#         )

#     optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
#     return model, optimizer, ema

def build_model_optimizer_ema(cfg: Config):
    model = TCL_TDA_Model_RefAlt(cfg).to(cfg.device)

    if torch.cuda.is_available():
        model = DDP(model, device_ids=[cfg.device.index], output_device=cfg.device.index)
    else:
        model = DDP(model)

    base_model = unwrap_model(model)
    ema = EMAWrapper(base_model, decay=cfg.ema_decay).to(cfg.device)

    head_params = (
        list(base_model.bio.parameters())
        + list(base_model.fusion.parameters())
        + list(base_model.seq_head.parameters())
        + list(base_model.bio_head.parameters())
        + list(base_model.mutation_encoder.parameters())
    )

    if getattr(cfg, "use_text", False):
        head_params += list(base_model.text_head.parameters())

    params = [{"params": head_params, "lr": cfg.lr_heads}]

    if (not getattr(cfg, "seq_offline_embeddings", False)) and getattr(cfg, "dna_finetune", False):
        params.append(
            {
                "params": base_model.dna.parameters(),
                "lr": cfg.lr_dnabert,
            }
        )

    if (
        getattr(cfg, "use_text", False)
        and (not getattr(cfg, "text_offline_embeddings", False))
        and getattr(cfg, "text_finetune", False)
    ):
        params.append(
            {
                "params": base_model.text_encoder.parameters(),
                "lr": cfg.lr_text,
            }
        )

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    return model, optimizer, ema


# =========================================================
# Loader
# =========================================================
def build_loader(dataset, cfg: Config, rank: int, world_size: int, shuffle: bool):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=cfg.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_refalt_ontology,
        pin_memory=False,
    )

    return loader, sampler





# =========================================================
# Logging / checkpoints
# =========================================================
def build_log_file(log_path: str, rank: int):
    if rank != 0:
        return

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "epoch",
                    "split",
                    "loss",
                    "contrast",
                    "tda",
                    "L_sb",
                    "L_st",
                    "L_bt",
                    "r1_seq2bio",
                    "r1_bio2seq",
                    "r5_seq2bio",
                    "r5_bio2seq",
                    "r10_seq2bio",
                    "r10_bio2seq",
                    "r1_seq2bio_multi",
                    "r1_bio2seq_multi",
                    "r5_seq2bio_multi",
                    "r5_bio2seq_multi",
                    "r10_seq2bio_multi",
                    "r10_bio2seq_multi",
                ]
            )


def append_metrics(log_path: str, epoch: int, split: str, metrics: Optional[dict], rank: int):
    if rank != 0 or metrics is None:
        return

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                split,
                metrics.get("loss", ""),
                metrics.get("contrast", ""),
                metrics.get("tda", ""),
                metrics.get("L_sb", ""),
                metrics.get("L_st", ""),
                metrics.get("L_bt", ""),
                metrics.get("r1_seq2bio", ""),
                metrics.get("r1_bio2seq", ""),
                metrics.get("r5_seq2bio", ""),
                metrics.get("r5_bio2seq", ""),
                metrics.get("r10_seq2bio", ""),
                metrics.get("r10_bio2seq", ""),
                metrics.get("r1_seq2bio_multi", ""),
                metrics.get("r1_bio2seq_multi", ""),
                metrics.get("r5_seq2bio_multi", ""),
                metrics.get("r5_bio2seq_multi", ""),
                metrics.get("r10_seq2bio_multi", ""),
                metrics.get("r10_bio2seq_multi", ""),
            ]
        )


# def save_checkpoint(
#     path: str,
#     epoch: int,
#     model,
#     teacher,
#     optimizer,
#     best_loss: float,
#     rank: int,
#     metrics: Optional[dict] = None,
# ):
#     if rank != 0:
#         return

#     ckpt = {
#         "epoch": epoch,
#         "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
#         "teacher_state_dict": teacher.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "best_loss": best_loss,
#         "metrics": metrics,
#     }
#     torch.save(ckpt, path)

def save_checkpoint(
    path: str,
    epoch: int,
    model,
    teacher,
    optimizer,
    best_loss: float,
    rank: int,
    metrics: Optional[dict] = None,
):
    if rank != 0:
        return

    ckpt = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "teacher_state_dict": unwrap_model(teacher).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "metrics": metrics,
    }
    torch.save(ckpt, path)


# def load_checkpoint(path: str, model, teacher, optimizer, device, rank: int):
#     ckpt = torch.load(path, map_location=device)

#     if hasattr(model, "module"):
#         model.module.load_state_dict(ckpt["model_state_dict"])
#     else:
#         model.load_state_dict(ckpt["model_state_dict"])

#     teacher.load_state_dict(ckpt["teacher_state_dict"])
#     optimizer.load_state_dict(ckpt["optimizer_state_dict"])

#     start_epoch = int(ckpt.get("epoch", 0))
#     best_loss = float(ckpt.get("best_loss", float("inf")))

#     if rank == 0:
#         print(
#             f"Loaded checkpoint from {path} | epoch={start_epoch} | best_loss={best_loss:.4f}",
#             flush=True,
#         )

#     return start_epoch, best_loss

def load_checkpoint(path: str, model, teacher, optimizer, device, rank: int):
    ckpt = torch.load(path, map_location=device)

    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    unwrap_model(teacher).load_state_dict(ckpt["teacher_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))

    if rank == 0:
        print(
            f"Loaded checkpoint from {path} | epoch={start_epoch} | best_loss={best_loss:.4f}",
            flush=True,
        )

    return start_epoch, best_loss


# =========================================================
# Metrics
# =========================================================
@torch.no_grad()
def recall_at_k_bidirectional(z_seq: torch.Tensor, z_bio: torch.Tensor, ks=(1, 5, 10)):
    z_seq = l2norm(z_seq, dim=1)
    z_bio = l2norm(z_bio, dim=1)

    sim = z_seq @ z_bio.T
    bsz = sim.size(0)
    gt = torch.arange(bsz, device=sim.device)

    metrics = {}
    for k in ks:
        k_eff = min(k, bsz)

        topk_seq = sim.topk(k_eff, dim=1).indices
        hit_seq = (topk_seq == gt.unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"r{k}_seq2bio"] = hit_seq.item()

        topk_bio = sim.T.topk(k_eff, dim=1).indices
        hit_bio = (topk_bio == gt.unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"r{k}_bio2seq"] = hit_bio.item()

    return metrics


@torch.no_grad()
def recall_at_k_bidirectional_multipositive(
    z_seq: torch.Tensor,
    z_bio: torch.Tensor,
    ontology_batch=None,
    ks=(1, 5, 10),
    sim_threshold: float = 0.5,
):
    z_seq = l2norm(z_seq, dim=1)
    z_bio = l2norm(z_bio, dim=1)

    sim = z_seq @ z_bio.T
    B = sim.size(0)
    device = sim.device

    pos_mask = torch.eye(B, dtype=torch.bool, device=device)

    def _to_term_set(x):
        if x is None:
            return set()

        if isinstance(x, set):
            return {str(v).strip() for v in x if str(v).strip()}

        if isinstance(x, (list, tuple)):
            out = set()
            for v in x:
                s = str(v).strip()
                if s:
                    out.add(s)
            return out

        s = str(x).strip()
        if not s:
            return set()

        if "," in s or ";" in s or "|" in s:
            tokens = re.split(r"[,;|]", s)
            return {t.strip() for t in tokens if t.strip()}

        return {s}

    if ontology_batch is not None:
        onto_sets = [_to_term_set(x) for x in ontology_batch]

        for i in range(B):
            si = onto_sets[i]
            for j in range(i + 1, B):
                sj = onto_sets[j]

                if len(si) == 0 or len(sj) == 0:
                    continue

                inter = len(si & sj)
                union = len(si | sj)
                score = inter / max(union, 1)

                if score >= sim_threshold:
                    pos_mask[i, j] = True
                    pos_mask[j, i] = True

    pos_mask.fill_diagonal_(True)

    metrics = {}

    for k in ks:
        k_eff = min(k, B)

        topk_seq = sim.topk(k_eff, dim=1).indices
        seq_hits = []
        for i in range(B):
            seq_hits.append(pos_mask[i, topk_seq[i]].any().float())
        metrics[f"r{k}_seq2bio_multi"] = torch.stack(seq_hits).mean().item()

        topk_bio = sim.T.topk(k_eff, dim=1).indices
        bio_hits = []
        for i in range(B):
            bio_hits.append(pos_mask[i, topk_bio[i]].any().float())
        metrics[f"r{k}_bio2seq_multi"] = torch.stack(bio_hits).mean().item()

    return metrics


# =========================================================
# Helpers for optional modalities
# =========================================================
def move_optional_tensor_to_device(x, device):
    if x is None:
        return None
    return x.to(device)


# def compute_contrastive_losses(cfg, z_seq, z_bio, z_text=None, ontology=None):
#     """
#     Returns:
#       L_contrast, L_sb, L_st, L_bt
#     """
#     L_sb = clip_like_loss_with_ontology(
#         z_seq,
#         z_bio,
#         ontology_batch=ontology,
#         temperature=cfg.temperature,
#         alpha_onto=getattr(cfg, "alpha_onto", 0.0),
#     )

#     L_st = None
#     L_bt = None

#     if getattr(cfg, "use_text", False):
#         if z_text is None:
#             raise ValueError("cfg.use_text=True but z_text is None.")

#         L_st = clip_like_loss_with_ontology(
#             z_seq,
#             z_text,
#             ontology_batch=ontology,
#             temperature=cfg.temperature,
#             alpha_onto=getattr(cfg, "alpha_onto", 0.0),
#         )

#         L_bt = clip_like_loss_with_ontology(
#             z_bio,
#             z_text,
#             ontology_batch=ontology,
#             temperature=cfg.temperature,
#             alpha_onto=getattr(cfg, "alpha_onto", 0.0),
#         )

#         L_contrast = (
#             getattr(cfg, "lambda_sb", 1.0) * L_sb
#             + getattr(cfg, "lambda_st", 1.0) * L_st
#             + getattr(cfg, "lambda_bt", 1.0) * L_bt
#         )
#     else:
#         L_contrast = L_sb

#     return L_contrast, L_sb, L_st, L_bt

def compute_contrastive_losses(cfg, z_seq, z_bio, z_text=None, ontology=None):
    """
    Returns:
      L_contrast, L_sb, L_st, L_bt
    """
    L_sb = clip_like_loss_with_ontology_v2(
        z_seq,
        z_bio,
        ontology_batch=ontology,
        temperature=cfg.temperature,
        alpha_onto=getattr(cfg, "alpha_onto", 0.0),
        onto_weights=getattr(cfg, "onto_weights", None),
        zero_diag=getattr(cfg, "zero_diag_onto", True),
        target_temp=getattr(cfg, "target_temp", 0.25),
    )

    L_st = None
    L_bt = None

    if getattr(cfg, "use_text", False):
        if z_text is None:
            raise ValueError("cfg.use_text=True but z_text is None.")

        L_st = clip_like_loss_with_ontology_v2(
            z_seq,
            z_text,
            ontology_batch=ontology,
            temperature=cfg.temperature,
            alpha_onto=getattr(cfg, "alpha_onto", 0.0),
            onto_weights=getattr(cfg, "onto_weights", None),
            zero_diag=getattr(cfg, "zero_diag_onto", True),
            target_temp=getattr(cfg, "target_temp", 0.25),
        )

        L_bt = clip_like_loss_with_ontology_v2(
            z_bio,
            z_text,
            ontology_batch=ontology,
            temperature=cfg.temperature,
            alpha_onto=getattr(cfg, "alpha_onto", 0.0),
            onto_weights=getattr(cfg, "onto_weights", None),
            zero_diag=getattr(cfg, "zero_diag_onto", True),
            target_temp=getattr(cfg, "target_temp", 0.25),
        )

        L_contrast = (
            getattr(cfg, "lambda_sb", 1.0) * L_sb
            + getattr(cfg, "lambda_st", 1.0) * L_st
            + getattr(cfg, "lambda_bt", 1.0) * L_bt
        )
    else:
        L_contrast = L_sb

    return L_contrast, L_sb, L_st, L_bt


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

 


# def forward_model_from_batch(cfg, model, batch):
#     m = unwrap_model(model)
#     bios = batch["bios"].to(cfg.device)
#     ref_emb = move_optional_tensor_to_device(batch.get("ref_emb", None), cfg.device)
#     alt_emb = move_optional_tensor_to_device(batch.get("alt_emb", None), cfg.device)
#     text_emb = move_optional_tensor_to_device(batch.get("text_emb", None), cfg.device)

#     outputs = m(                          # works for DDP and plain nn.Module
#         seq_ref=batch.get("seq_refs", None),
#         seq_alt=batch.get("seq_alts", None),
#         bio_feats=bios,
#         texts=batch.get("texts", None),
#         ref_emb=ref_emb,
#         alt_emb=alt_emb,
#         text_emb=text_emb,
#     )
#     return outputs, bios

# def forward_model_from_batch(cfg, model, batch):
#     bios = batch["bios"].to(cfg.device)
#     ref_emb = move_optional_tensor_to_device(batch.get("ref_emb", None), cfg.device)
#     alt_emb = move_optional_tensor_to_device(batch.get("alt_emb", None), cfg.device)
#     text_emb = move_optional_tensor_to_device(batch.get("text_emb", None), cfg.device)

#     base_model = unwrap_model(model)

#     outputs = base_model(
#         seq_ref=batch.get("seq_refs", None),
#         seq_alt=batch.get("seq_alts", None),
#         bio_feats=bios,
#         texts=batch.get("texts", None),
#         ref_emb=ref_emb,
#         alt_emb=alt_emb,
#         text_emb=text_emb,
#     )
#     return outputs, bios

def move_to_device(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def forward_model_from_batch(cfg, model, batch):
    bios = batch["bios"].to(cfg.device)
    seq_refs = move_to_device(batch.get("seq_refs", None), cfg.device)
    seq_alts = move_to_device(batch.get("seq_alts", None), cfg.device)
    texts = move_to_device(batch.get("texts", None), cfg.device)
    ref_emb = move_to_device(batch.get("ref_emb", None), cfg.device)
    alt_emb = move_to_device(batch.get("alt_emb", None), cfg.device)
    text_emb = move_to_device(batch.get("text_emb", None), cfg.device)

    base_model = unwrap_model(model)

    outputs = base_model(
        seq_ref=seq_refs,
        seq_alt=seq_alts,
        bio_feats=bios,
        texts=texts,
        ref_emb=ref_emb,
        alt_emb=alt_emb,
        text_emb=text_emb,
    )
    return outputs, bios


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    cfg,
    model,
    ema,
    optimizer,
    loader,
    diffusion_path: str,
    N: int,
):
    model.train()

    stats = {
        "loss": 0.0,
        "contrast": 0.0,
        "tda": 0.0,
        "L_sb": 0.0,
        "L_st": 0.0,
        "L_bt": 0.0,
        "n_steps": 0.0,
        "n_text_steps": 0.0,
    }

    if cfg.lambda_tda > 0:
        D_target = np.memmap(
            diffusion_path,
            dtype="float16",
            mode="r",
            shape=(N, N),
        )

    for step, batch in enumerate(loader):
        t0 = time.time()
        idxs = batch["idxs"]
        bs = batch["bios"].shape[0]
        ontology = batch.get("ontology", None)
        t1 = time.time()

        (
            z_seq,
            z_bio,
            z_text,
            h_tda,
            h_mut,
            h_bio,
            h_text,
            h_ref,
            h_alt,
        ), _ = forward_model_from_batch(cfg, model, batch)
        t2 = time.time()

        L_contrast, L_sb, L_st, L_bt = compute_contrastive_losses(
            cfg=cfg,
            z_seq=z_seq,
            z_bio=z_bio,
            z_text=z_text,
            ontology=ontology,
        )

        # idxs_np = np.asarray(idxs, dtype=int)
        # D_sub = D_target[np.ix_(idxs_np, idxs_np)].astype(np.float32)
        # D_target_batch = torch.from_numpy(D_sub).to(cfg.device)
        # t3 = time.time()

        # L_tda = tda_pairwise_regularizer(
        #     h_tda,
        #     D_target_batch,
        #     k=cfg.knn_k,
        #     t=cfg.diff_t,
        # )
        # use_tda = (cfg.lambda_tda > 0) and (step % getattr(cfg, "tda_every", 1) == 0)
        use_tda = (cfg.lambda_tda > 0)
        t3 = time.time()
        if use_tda:
            idxs_np = np.asarray(idxs, dtype=int)

            tda_bs = min(len(idxs_np), getattr(cfg, "tda_batch_size", len(idxs_np)))

            if tda_bs < len(idxs_np):
                sub_idx = np.random.choice(len(idxs_np), size=tda_bs, replace=False)
                idxs_tda = idxs_np[sub_idx]
                h_tda_sub = h_tda[sub_idx]
            else:
                idxs_tda = idxs_np
                h_tda_sub = h_tda

            D_sub = D_target[np.ix_(idxs_tda, idxs_tda)].astype(np.float32)
            D_target_batch = torch.from_numpy(D_sub).to(cfg.device)

            k_eff = min(cfg.knn_k, max(1, tda_bs - 1))
            L_tda = tda_pairwise_regularizer(
                h_tda_sub,
                D_target_batch,
                k=k_eff,
                t=cfg.diff_t,
            )
            # print("TDA loss")
            # print(L_tda)
        else:
            L_tda = torch.tensor(0.0, device=cfg.device)

        
        optimizer.zero_grad()
        loss = L_contrast + cfg.lambda_tda * L_tda

        if use_tda:
            del idxs_np, D_sub, D_target_batch
        # else:
        #     del bios

        if torch.isnan(loss):
            print(
                "NaN loss detected "
                f"| total={loss.item() if torch.isfinite(loss) else 'nan'} "
                f"| L_sb={L_sb.item() if L_sb is not None else 'NA'} "
                f"| L_st={L_st.item() if L_st is not None else 'NA'} "
                f"| L_bt={L_bt.item() if L_bt is not None else 'NA'} "
                f"| L_tda={L_tda.item() if L_tda is not None else 'NA'}",
                flush=True,
            )
            continue

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=getattr(cfg, "grad_clip", 1.0),   # add grad_clip=1.0 to your Config
        )
        optimizer.step()
        # ema.update(model.module)
        ema.update(unwrap_model(model))
        t4 = time.time()
        if step % 10 == 0 and dist.get_rank() == 0:
            
            print(f"step={step} | batch_time={t1-t0:.3f}s", flush=True)
            print(
                f"step={step} | "
                f"batch_load={t1-t0:.3f}s | "
                f"forward={t2-t1:.3f}s | "
                f"diff_read={t3-t2:.3f}s | "
                f"backward_opt={t4-t3:.3f}s | "
                f"total={t4-t0:.3f}s",
                flush=True,
            )

        stats["loss"] += loss.item()
        stats["contrast"] += L_contrast.item()
        stats["tda"] += (cfg.lambda_tda * L_tda).item()
        stats["L_sb"] += L_sb.item()

        if L_st is not None:
            stats["L_st"] += L_st.item()
            stats["n_text_steps"] += 1.0

        if L_bt is not None:
            stats["L_bt"] += L_bt.item()

        stats["n_steps"] += 1.0
        if step % 2 == 0 and dist.get_rank() == 0:
            avg_loss = stats["loss"] / max(stats["n_steps"], 1.0)
            avg_contrast = stats["contrast"] / max(stats["n_steps"], 1.0)
            avg_tda = stats["tda"] / max(stats["n_steps"], 1.0)
            avg_L_sb = stats["L_sb"] / max(stats["n_steps"], 1.0)

            if stats["n_text_steps"] > 0:
                avg_L_st = stats["L_st"] / stats["n_text_steps"]
                avg_L_bt = stats["L_bt"] / stats["n_text_steps"]
            else:
                avg_L_st = 0.0
                avg_L_bt = 0.0

            print(
                f"Step {step:03d} | "
                f"train_loss={avg_loss:.4f} | "
                f"contrast={avg_contrast:.4f} | "
                f"tda={avg_tda:.4f} | "
                f"L_sb={avg_L_sb:.4f} | "
                f"L_st={avg_L_st:.4f} | "
                f"L_bt={avg_L_bt:.4f}",
                flush=True,
            )

    if cfg.lambda_tda > 0:
        del D_target

    tensor_stats = torch.tensor(
        [
            stats["loss"],
            stats["contrast"],
            stats["tda"],
            stats["L_sb"],
            stats["L_st"],
            stats["L_bt"],
            stats["n_steps"],
            stats["n_text_steps"],
        ],
        device=cfg.device,
        dtype=torch.float32,
    )

    dist.all_reduce(tensor_stats, op=dist.ReduceOp.SUM)

    (
        total_loss,
        total_contrast,
        total_tda,
        total_L_sb,
        total_L_st,
        total_L_bt,
        total_steps,
        total_text_steps,
    ) = tensor_stats.tolist()

    total_steps = max(total_steps, 1.0)
    total_text_steps = max(total_text_steps, 1.0)

    metrics = {
        "loss": total_loss / total_steps,
        "contrast": total_contrast / total_steps,
        "tda": total_tda / total_steps,
        "L_sb": total_L_sb / total_steps,
    }

    if getattr(cfg, "use_text", False):
        metrics["L_st"] = total_L_st / total_text_steps
        metrics["L_bt"] = total_L_bt / total_text_steps
    else:
        metrics["L_st"] = 0.0
        metrics["L_bt"] = 0.0
    # if step % 2 == 0 and dist.get_rank() == 0:
    #     print(
    #         f"Step {step:03d} | "
    #         f"train_loss={metrics['loss']:.4f} | "
    #         f"contrast={metrics['contrast']:.4f} | "
    #         f"tda={metrics['tda']:.4f} | "
    #         f"L_sb={metrics.get('L_sb', 0.0):.4f} | "
    #         f"L_st={metrics.get('L_st', 0.0):.4f} | "
    #         f"L_bt={metrics.get('L_bt', 0.0):.4f}",
    #         flush=True,
    #         )



    return metrics


# =========================================================
# Evaluate
# =========================================================
@torch.no_grad()
def evaluate_one_epoch(
    cfg,
    model,
    loader,
    split_name: str = "val",
):
    """
    Validation / evaluation without diffusion matrix.

    We evaluate only:
      - contrastive loss
      - retrieval metrics

    """
    model.eval()

    stats = {
        "loss": 0.0,
        "contrast": 0.0,
        "tda": 0.0,          # kept for logging compatibility
        "L_sb": 0.0,
        "L_st": 0.0,
        "L_bt": 0.0,
        "n_steps": 0.0,
        "n_text_steps": 0.0,
    }

    recalls_sum = {}
    recalls_count = 0.0

    for batch in loader:
        ontology = batch.get("ontology", None)

        (
            z_seq,
            z_bio,
            z_text,
            h_tda,
            h_mut,
            h_bio,
            h_text,
            h_ref,
            h_alt,
        ), bios = forward_model_from_batch(cfg, model, batch)

        L_contrast, L_sb, L_st, L_bt = compute_contrastive_losses(
            cfg=cfg,
            z_seq=z_seq,
            z_bio=z_bio,
            z_text=z_text,
            ontology=ontology,
        )

        # Validation loss = contrastive loss only
        loss = L_contrast

        strict = recall_at_k_bidirectional(z_seq, z_bio)

        multi = recall_at_k_bidirectional_multipositive(
            z_seq,
            z_bio,
            ontology_batch=ontology,
            sim_threshold=getattr(cfg, "eval_onto_threshold", 0.5),
        )

        stats["loss"] += loss.item()
        stats["contrast"] += L_contrast.item()
        stats["tda"] += 0.0   # no validation TDA
        stats["L_sb"] += L_sb.item()

        if L_st is not None:
            stats["L_st"] += L_st.item()
            stats["n_text_steps"] += 1.0

        if L_bt is not None:
            stats["L_bt"] += L_bt.item()

        stats["n_steps"] += 1.0
        recalls_count += 1.0

        for key, value in {**strict, **multi}.items():
            recalls_sum[key] = recalls_sum.get(key, 0.0) + float(value)

        del bios

    tensor_stats = torch.tensor(
        [
            stats["loss"],
            stats["contrast"],
            stats["tda"],
            stats["L_sb"],
            stats["L_st"],
            stats["L_bt"],
            stats["n_steps"],
            stats["n_text_steps"],
            recalls_count,
        ],
        device=cfg.device,
        dtype=torch.float32,
    )

    dist.all_reduce(tensor_stats, op=dist.ReduceOp.SUM)

    (
        total_loss,
        total_contrast,
        total_tda,
        total_L_sb,
        total_L_st,
        total_L_bt,
        total_steps,
        total_text_steps,
        total_recalls_count,
    ) = tensor_stats.tolist()

    total_steps = max(total_steps, 1.0)
    total_text_steps = max(total_text_steps, 1.0)
    total_recalls_count = max(total_recalls_count, 1.0)

    metrics = {
        "loss": total_loss / total_steps,
        "contrast": total_contrast / total_steps,
        "tda": 0.0,  # no TDA in validation
        "L_sb": total_L_sb / total_steps,
    }

    if getattr(cfg, "use_text", False):
        metrics["L_st"] = total_L_st / total_text_steps
        metrics["L_bt"] = total_L_bt / total_text_steps
    else:
        metrics["L_st"] = 0.0
        metrics["L_bt"] = 0.0

    recall_keys = [
        "r1_seq2bio",
        "r1_bio2seq",
        "r5_seq2bio",
        "r5_bio2seq",
        "r10_seq2bio",
        "r10_bio2seq",
        "r1_seq2bio_multi",
        "r1_bio2seq_multi",
        "r5_seq2bio_multi",
        "r5_bio2seq_multi",
        "r10_seq2bio_multi",
        "r10_bio2seq_multi",
    ]

    recall_tensor = torch.tensor(
        [recalls_sum.get(k, 0.0) for k in recall_keys],
        device=cfg.device,
        dtype=torch.float32,
    )
    dist.all_reduce(recall_tensor, op=dist.ReduceOp.SUM)

    recall_vals = recall_tensor.tolist()
    for k, v in zip(recall_keys, recall_vals):
        metrics[k] = v / total_recalls_count

    return metrics


def build_diffusion_if_needed_distributed(
    model,
    dataset,
    cfg,
    out_path,
    rank,
    world_size,
):
    if rank == 0:
        need_build = not os.path.exists(out_path)
        print(f"[rank0] diffusion file={out_path} | need_build={need_build}", flush=True)
    else:
        need_build = None

    obj = [need_build]
    dist.broadcast_object_list(obj, src=0)
    need_build = obj[0]

    if need_build:
        print(f"[rank {rank}] building diffusion matrix: {out_path}", flush=True)
        hierarchical_build_and_save_diffusion_matrix_distributed(
            model=model,
            dataset=dataset,
            cfg=cfg,
            out_path=out_path,
            block_size=getattr(cfg, "diff_block_size", 5000),
            save_dtype="float16",
            rank=rank,
            world_size=world_size,
        )

    dist.barrier()

    if rank == 0:
        print(f"[rank0] diffusion ready: {out_path}", flush=True)

# =========================================================
# Main training driver
# =========================================================
def train_refalt_distributed_clean(
    cfg,
    train_ds,
    N_train,
    val_ds=None,
    N_val=None,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        if rank == 0:
            print("\n" + "=" * 70, flush=True)
            print("DISTRIBUTED REFALT TRAINING WITH EMA AND TDA", flush=True)
            print("=" * 70, flush=True)
            print(f"Config: {cfg}", flush=True)
            print("=" * 70 + "\n", flush=True)
            print(f"seq_offline_embeddings={getattr(cfg, 'seq_offline_embeddings', False)}", flush=True)
            print(f"text_offline_embeddings={getattr(cfg, 'text_offline_embeddings', False)}", flush=True)
            print(f"use_text={getattr(cfg, 'use_text', False)}", flush=True)
            

        model, optimizer, ema = build_model_optimizer_ema(cfg)

        train_loader, train_sampler = build_loader(
            train_ds,
            cfg,
            rank,
            world_size,
            shuffle=True,
        )

        val_loader = None
        if val_ds is not None:
            if rank == 0:
                print(f"[rank0] starting validation diffusion build for N={N_val}", flush=True)
            val_loader, _ = build_loader(
                val_ds,
                cfg,
                rank,
                world_size,
                shuffle=False,
            )

        os.makedirs(cfg.save_dir, exist_ok=True)

        best_path = os.path.join(cfg.save_dir, "best_model.pt")
        # best_train_loss_path = os.path.join(cfg.save_dir, "best_train_loss_model.pt")
        latest_path = os.path.join(cfg.save_dir, "latest_checkpoint.pt")
        log_path = os.path.join(cfg.save_dir, "training_log.csv")

        build_log_file(log_path, rank)

        best_loss = float("inf")
        start_epoch = 1


        if rank == 0:
            has_ckpt = os.path.exists(latest_path)
        else:
            has_ckpt = False
        obj = [has_ckpt]
        dist.broadcast_object_list(obj, src=0)
        if obj[0]:
            start_epoch, best_loss = load_checkpoint(
                latest_path, model, ema.teacher, optimizer, cfg.device, rank,
            )
            start_epoch += 1

        train_diffusion_path = getattr(cfg, "train_diffusion_file", "D_target_train.npy")
        # val_diffusion_path = getattr(cfg, "val_diffusion_file", "D_target_val.npy")

        # rank = dist.get_rank()
        # world_size = dist.get_world_size()

        if cfg.lambda_tda > 0:
            build_diffusion_if_needed_distributed(
                model=ema,
                dataset=train_ds,
                cfg=cfg,
                out_path=train_diffusion_path,
                rank=rank,
                world_size=world_size,
            )
            # dist.barrier()
        # if val_ds is not None:
        #     build_diffusion_if_needed_distributed(
        #         model=ema,
        #         dataset=val_ds,
        #         cfg=cfg,
        #         out_path=val_diffusion_path,
        #         rank=rank,
        #         world_size=world_size,
        #     )
        # dist.barrier()
        if rank == 0:
            print(f"[rank0] N_train={N_train} | train_diffusion_path={train_diffusion_path}", flush=True)
            print(f"[rank0] N_val={N_val}", flush=True)
            print(f"[rank0] val_ds is None? {val_ds is None}", flush=True)
            print(f"[rank0] train diffusion exists? {os.path.exists(train_diffusion_path)}", flush=True)

        for epoch in range(start_epoch, cfg.epochs + 1):
            train_sampler.set_epoch(epoch)

            start = time.time()
            train_metrics = train_one_epoch(
                cfg=cfg,
                model=model,
                ema=ema,
                optimizer=optimizer,
                loader=train_loader,
                diffusion_path=train_diffusion_path,
                N=N_train,
            )
            if rank == 0:
                print("epoch time:", time.time() - start, flush=True)

            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate_one_epoch(
                    cfg=cfg,
                    model=ema.teacher,
                    loader=val_loader,
                    split_name="val",
                )

            if rank == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"contrast={train_metrics['contrast']:.4f} | "
                    f"tda={train_metrics['tda']:.4f} | "
                    f"L_sb={train_metrics.get('L_sb', 0.0):.4f} | "
                    f"L_st={train_metrics.get('L_st', 0.0):.4f} | "
                    f"L_bt={train_metrics.get('L_bt', 0.0):.4f}",
                    flush=True,
                )

                if val_metrics is not None:
                    print(
                        f"val_loss={val_metrics['loss']:.4f} | "
                        f"R@1 seq->bio={val_metrics.get('r1_seq2bio', 0.0):.4f} | "
                        f"R@1 bio->seq={val_metrics.get('r1_bio2seq', 0.0):.4f}",
                        flush=True,
                    )

            append_metrics(log_path, epoch, "train", train_metrics, rank)
            append_metrics(log_path, epoch, "val", val_metrics, rank)

            # metric_for_best = train_metrics["loss"] if val_metrics is not None else val_metrics["loss"]
            metric_for_best = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]

            save_checkpoint(
                latest_path,
                epoch,
                model,
                ema.teacher,
                optimizer,
                best_loss,
                rank,
                metrics=val_metrics if val_metrics is not None else train_metrics,
            )

            if rank == 0 and metric_for_best < best_loss:
                best_loss = metric_for_best
                save_checkpoint(
                    best_path,
                    epoch,
                    model,
                    ema.teacher,
                    optimizer,
                    best_loss,
                    rank,
                    metrics=val_metrics if val_metrics is not None else train_metrics,
                )
                print(f"NEW BEST model saved (loss={best_loss:.4f})", flush=True)



            if cfg.lambda_tda > 0 and epoch % cfg.tda_rebuild_every == 0:
                hierarchical_build_and_save_diffusion_matrix_distributed(
                    model=ema,
                    dataset=train_ds,
                    cfg=cfg,
                    out_path=train_diffusion_path,
                    block_size=getattr(cfg, "diff_block_size", 5000),
                    save_dtype="float16",
                    rank=rank,
                    world_size=world_size,
                )
                dist.barrier()

        if rank == 0:
            print("\n" + "=" * 70, flush=True)
            print("DISTRIBUTED TRAINING COMPLETED", flush=True)
            print("=" * 70 + "\n", flush=True)

        return unwrap_model(model), unwrap_model(ema.teacher)

    except Exception as e:
        print(f"\nERROR in train_refalt_distributed_clean: {e}", flush=True)
        traceback.print_exc()
        raise

    finally:
        gc.collect()
        cleanup_distributed()


def validate_offline_config(cfg):
    if getattr(cfg, "seq_offline_embeddings", False):
        if not getattr(cfg, "train_ref_emb_path", None):
            raise ValueError("seq_offline_embeddings=True but train_ref_emb_path is missing")
        if not getattr(cfg, "train_alt_emb_path", None):
            raise ValueError("seq_offline_embeddings=True but train_alt_emb_path is missing")
        if not getattr(cfg, "val_ref_emb_path", None):
            raise ValueError("seq_offline_embeddings=True but val_ref_emb_path is missing")
        if not getattr(cfg, "val_alt_emb_path", None):
            raise ValueError("seq_offline_embeddings=True but val_alt_emb_path is missing")

    if getattr(cfg, "use_text", False) and getattr(cfg, "text_offline_embeddings", False):
        if not getattr(cfg, "train_text_emb_path", None):
            raise ValueError("text_offline_embeddings=True but train_text_emb_path is missing")
        if not getattr(cfg, "val_text_emb_path", None):
            raise ValueError("text_offline_embeddings=True but val_text_emb_path is missing")



cont_cols = [
    "QUAL",
    "DP_GERM", "VAF_GERM", "DP_SOM", "VAF_SOM", "DP_UNI", "VAF_UNI",
    "IG_AF", "EVS_EA_MAF", "1000G_Global_AF", "gnomAD_Global_AF", "Kaviar_AF", "MAX_AF", "MAX_POP_AF", "GnomAD_MNV_AF",
    "1000G_AFR_AF", "1000G_AMR_AF", "1000G_EAS_AF", "1000G_EUR_AF", "1000G_SAS_AF", "1000G_AA_AF", "1000G_EA_AF",
    "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF", "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF",
    "PhyloP", "PhastCons", "Distance_Grantham",
    "MaxEntScan_alt", "MaxEntScan_diff", "MaxEntScan_ref",
    "DANN_Score", "FATHMM_Non_Coding_Score", "FATHMM_Coding_Score", "Gene_damage_index",
    "N_PATIENTS_WITH_VARIANT", "N_CANCER_WITH_VARIANT", "N_NONCANCER_WITH_VARIANT", "SOMATIC_REPEAT_COUNT",
]

ordinal_cols = [
    "SIFT_score", "PolyPhen_ord", "PolyPhen_score",
    "Consequence_num", "Impact_num", "VariantClass_num", "Biotype_num", "Canonical_num",
    "Origin_num", "CNV_LOH_status_num",
]

binary_cols = [
    "SIFT_bin",
    "HAS_GERM", "HAS_SOM", "N_GERM_ONLY", "N_GERM_AND_SOM",
    "PATHWAY_HIT", "PATH_DNA_REPAIR",
    "LOH_PARTIAL_PATHO", "LOH_PARTIAL_PATHO_SOM",
    "CNV_HIT", "CNV_pathogenic",
    "IN_ROH", "HET_HIGH_HOM", "HAS_SOMATIC_PATIENT", "GERM_PASS",
    "VARIANT_CLASS_SNV", "VARIANT_CLASS_deletion", "VARIANT_CLASS_insertion", "VARIANT_CLASS_substitution",
    "IMPACT_HIGH", "IMPACT_LOW", "IMPACT_MODERATE", "IMPACT_UNKNOWN",
    "BIOTYPE_CTCF_binding_site", "BIOTYPE_NA", "BIOTYPE_TF_binding_site", "BIOTYPE_enhancer",
    "BIOTYPE_lncRNA", "BIOTYPE_miRNA", "BIOTYPE_misc_RNA", "BIOTYPE_open_chromatin_region",
    "BIOTYPE_promoter", "BIOTYPE_promoter_flanking_region", "BIOTYPE_protein_coding",
    "BIOTYPE_snoRNA", "BIOTYPE_transcribed_pseudogene",
    "ORIGIN_germline", "ORIGIN_somatic",
]

unit_cols = [
    "ada_score",
    "rf_score",
]

ordinal_max_map = {
    "SIFT_score": 1,
    "PolyPhen_ord": 3,
    "PolyPhen_score": 1,
    "Consequence_num": 5,
    "Impact_num": 4,
    "VariantClass_num": 4,
    "Biotype_num": 5,
    "Canonical_num": 1,
    "Origin_num": 3,
    "CNV_LOH_status_num": 2,
}
# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        print("I am in")
        rank, world_size = setup_distributed()

        #config for simple model
        cfg = Config()
        validate_offline_config(cfg)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            cfg.device = torch.device(f"cuda:{local_rank}")
        else:
            cfg.device = torch.device("cpu")

        # train_ds, n_train = get_data_all(
        #     seq_tsv=cfg.train_seq,
        #     bio_tsv=cfg.train_bio,
        #     onto_tsv=cfg.train_onto,
        #     text_tsv=getattr(cfg, "train_text", None),
        #     textfields_tsv=getattr(cfg, "train_textfields", None),
        #     labels_path=cfg.train_labels if os.path.exists(cfg.train_labels) else None,
        #     label_col="labels",
        #     include_ontology=True,
        #     include_oncology=True,
        #     cache_dir="dataset",
        #     use_cache=True,
        #     rebuild_cache=True,
        #     max_bio_line=getattr(cfg, "max_bio_line", None),
        #     use_text=getattr(cfg, "use_text", False),
        #     seq_offline_embeddings=getattr(cfg, "seq_offline_embeddings", False),
        #     text_offline_embeddings=getattr(cfg, "text_offline_embeddings", False),
        #     dna_model_name=getattr(cfg, "dna_model_name", None),
        #     text_model_name=getattr(cfg, "text_model_name", None),
        #     seq_pool=getattr(cfg, "pool", "mean"),
        #     text_pool=getattr(cfg, "text_pool", "mean"),
        #     seq_max_len=getattr(cfg, "max_len", 512),
        #     text_max_len=getattr(cfg, "text_max_len", 256),
        #     precompute_batch_size=getattr(cfg, "precompute_batch_size", 32),
        #     embedding_device=str(cfg.device),
        #     precomputed_ref_path=getattr(cfg, "train_ref_emb_path", None),
        #     precomputed_alt_path=getattr(cfg, "train_alt_emb_path", None),
        #     precomputed_text_path=getattr(cfg, "train_text_emb_path", None),
        # )

        # val_ds, n_val = get_data_all(
        #     seq_tsv=cfg.test_seq,
        #     bio_tsv=cfg.test_bio,
        #     onto_tsv=cfg.test_onto,
        #     text_tsv=getattr(cfg, "test_text", None),
        #     textfields_tsv=getattr(cfg, "test_textfields", None),
        #     labels_path=cfg.test_labels if os.path.exists(cfg.test_labels) else None,
        #     label_col="labels",
        #     include_ontology=True,
        #     include_oncology=True,
        #     cache_dir="dataset",
        #     use_cache=True,
        #     rebuild_cache=True,
        #     max_bio_line=getattr(cfg, "max_bio_line_eval", None),
        #     use_text=getattr(cfg, "use_text", False),
        #     seq_offline_embeddings=getattr(cfg, "seq_offline_embeddings", False),
        #     text_offline_embeddings=getattr(cfg, "text_offline_embeddings", False),
        #     dna_model_name=getattr(cfg, "dna_model_name", None),
        #     text_model_name=getattr(cfg, "text_model_name", None),
        #     seq_pool=getattr(cfg, "pool", "mean"),
        #     text_pool=getattr(cfg, "text_pool", "mean"),
        #     seq_max_len=getattr(cfg, "max_len", 512),
        #     text_max_len=getattr(cfg, "text_max_len", 256),
        #     precompute_batch_size=getattr(cfg, "precompute_batch_size", 32),
        #     embedding_device=str(cfg.device),
        #     precomputed_ref_path=getattr(cfg, "val_ref_emb_path", None),
        #     precomputed_alt_path=getattr(cfg, "val_alt_emb_path", None),
        #     precomputed_text_path=getattr(cfg, "val_text_emb_path", None),
        # )

        train_ds, n_train = get_data_all(
                seq_tsv=cfg.train_seq,
                bio_tsv=cfg.train_bio,   
                onto_tsv=cfg.train_onto,
                text_tsv=getattr(cfg, "train_text", None),
                textfields_tsv=getattr(cfg, "train_textfields", None),
                labels_path=cfg.train_labels if os.path.exists(cfg.train_labels) else None,
                label_col="labels",
                include_ontology=True,
                include_oncology=False,
                cache_dir="dataset",
                use_cache=True,
                rebuild_cache=True,
                max_bio_line=getattr(cfg, "max_bio_line", None),
                use_text=True,
                seq_offline_embeddings=getattr(cfg, "seq_offline_embeddings", False),
                text_offline_embeddings=False,
                dna_model_name=getattr(cfg, "dna_model_name", None),
                text_model_name=getattr(cfg, "text_model_name", None),
                seq_pool=getattr(cfg, "pool", "mean"),
                text_pool=getattr(cfg, "text_pool", "mean"),
                seq_max_len=getattr(cfg, "max_len", 512),
                text_max_len=getattr(cfg, "text_max_len", 256),
                precompute_batch_size=getattr(cfg, "precompute_batch_size", 32),
                embedding_device=str(cfg.device),
                precomputed_ref_path=getattr(cfg, "train_ref_emb_path", None),
                precomputed_alt_path=getattr(cfg, "train_alt_emb_path", None),
                precomputed_text_path=getattr(cfg, "train_text_emb_path", None),
                cont_cols=cont_cols,
                ordinal_cols=ordinal_cols,
                binary_cols=binary_cols,
                unit_cols=unit_cols,
                ordinal_max_map=ordinal_max_map,
                corr_threshold=None,
            )                       
        val_ds, n_val = get_data_all(
            seq_tsv=cfg.test_seq,
            bio_tsv=cfg.test_bio,
            onto_tsv=cfg.test_onto,
            text_tsv=getattr(cfg, "test_text", None),
            textfields_tsv=getattr(cfg, "test_textfields", None),
            labels_path=cfg.test_labels if os.path.exists(cfg.test_labels) else None,
            label_col="labels",
            include_ontology=True,
            include_oncology=False,
            cache_dir="dataset",
            use_cache=True,
            rebuild_cache=True,
            max_bio_line=getattr(cfg, "max_bio_line", None),
            use_text=True,
            seq_offline_embeddings=getattr(cfg, "seq_offline_embeddings", False),
            text_offline_embeddings=False,
            dna_model_name=getattr(cfg, "dna_model_name", None),
            text_model_name=getattr(cfg, "text_model_name", None),
            seq_pool=getattr(cfg, "pool", "mean"),
            text_pool=getattr(cfg, "text_pool", "mean"),
            seq_max_len=getattr(cfg, "max_len", 512),
            text_max_len=getattr(cfg, "text_max_len", 256),
            precompute_batch_size=getattr(cfg, "precompute_batch_size", 32),
            embedding_device=str(cfg.device),
            precomputed_ref_path=getattr(cfg, "val_ref_emb_path", None),
            precomputed_alt_path=getattr(cfg, "val_alt_emb_path", None),
            precomputed_text_path=getattr(cfg, "val_text_emb_path", None),
            cont_cols=cont_cols,
            ordinal_cols=ordinal_cols,
            binary_cols=binary_cols,
            unit_cols=unit_cols,
            ordinal_max_map=ordinal_max_map,
            corr_threshold=None,
        )



        print(f"train n = {n_train}")
        print(f"val n   = {n_val}")
        print(f"val_ds is None? {val_ds is None}")
        cfg.bio_dim = train_ds.bio_in_dim

        print("Training bio")
        print("*" * 50)
        print(cfg.bio_dim)

        print("First 50 bio columns:")
        print(train_ds.bio_feature_cols[:50])

        train_refalt_distributed_clean(
            cfg=cfg,
            train_ds=train_ds,
            N_train=n_train,
            val_ds=val_ds,
            N_val=n_val,
        )

        for var in ["cfg", "train_ds", "val_ds", "n_train", "n_val"]:
            if var in locals():
                del locals()[var]

        gc.collect()

    except Exception as e:
        print("\nERROR in main:", e)
        print("Cleaning up RAM...")

        for var in ["cfg", "train_ds", "val_ds", "n_train", "n_val"]:
            if var in locals():
                try:
                    del locals()[var]
                except Exception:
                    pass

        gc.collect()
        raise