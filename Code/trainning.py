from __future__ import annotations

from libs import *
from config import Config
from models import TCL_TDA_Model_RefAlt
from loss import (
    collate_fn_refalt,
    clip_like_loss,
    clip_soft_sirius_loss,
)
from utils import (
    setup_distributed, cleanup_distributed,
    init_gtda_objects, pure_tda_feature_from_batch, linear_warmup,
    check_nans, rank0_print, set_seed,
    get_data_all, make_train_subset,
)
from samplers import DistributedLabeledCollisionBatchSampler
from results import run_all_results



def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_one_run_distributed(
    cfg: Config,
    train_ds: Dataset,
    all_label_train: np.ndarray,
    test_ds: Optional[Dataset],
    run_name: str,
    mode: str,
):

    assert mode in {"clip_only", "labelled_clip", "clip_tda_labelled"}

    rank, world_size = setup_distributed()
    print("World size is ", world_size)

    run_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    latest_path = os.path.join(run_dir, "latest_checkpoint.pt")
    log_path = os.path.join(run_dir, "training_log.csv")
    cfg_path = os.path.join(run_dir, "config_used.json")

    best_loss_path = os.path.join(run_dir, "best_loss.pt")

    eval_every = int(getattr(cfg, "eval_every", 1))

    best_metric_key = str(getattr(cfg, "best_metric_key", "val/task/auprc"))
    best_metric_mode = str(getattr(cfg, "best_metric_mode", "max"))  # "max" or "min"
    best_metric_val = -float("inf") if best_metric_mode == "max" else float("inf")

    save_best_history = bool(getattr(cfg, "save_best_history", True))

    metric_tag = best_metric_key.replace("/", "__")
    best_metric_path = os.path.join(run_dir, f"best_metric__{metric_tag}.pt")

    if rank == 0:
        rank0_print("\n" + "=" * 80)
        rank0_print(f"RUN: {run_name}")
        rank0_print(f"mode = {mode}")
        rank0_print(f"device={cfg.device} | world_size={world_size}")
        rank0_print(f"best_metric_key={best_metric_key} | best_metric_mode={best_metric_mode}")
        rank0_print(f"best_loss_path={best_loss_path}")
        rank0_print(f"best_metric_path={best_metric_path}")
        rank0_print("=" * 80 + "\n")

        try:
            save_json(cfg_path, cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg))
        except Exception:
            save_json(cfg_path, {"error": "could not serialize cfg"})

        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "batch",
                    "loss_total", "loss_clip", "loss_tda", "lambda_tda",
                    "time_sec"
                ])


    model = TCL_TDA_Model_RefAlt(cfg).to(cfg.device)
    model = DDP(model)


    params = [
        {"params": list(model.module.bio.parameters()), "lr": cfg.lr_heads},
        {"params": list(model.module.fusion.parameters()), "lr": cfg.lr_heads},
        {"params": list(model.module.seq_head.parameters()), "lr": cfg.lr_heads},
        {"params": list(model.module.bio_head.parameters()), "lr": cfg.lr_heads},
    ]
    if mode == "clip_tda_labelled":
        params.append({"params": list(model.module.tda_pure_head.parameters()), "lr": cfg.lr_heads})
    if cfg.dna_finetune:
        params.append({"params": list(model.module.dna.parameters()), "lr": cfg.lr_dnabert})

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)


    alpha_soft = float(getattr(cfg, "alpha_soft", 0.25))
    unlabeled_val = int(getattr(cfg, "unlabeled_val", -1))
    temperature = float(getattr(cfg, "temperature", 0.07))

    tda_warmup_epochs = int(getattr(cfg, "tda_warmup_epochs", 8))
    lambda_tda_max = float(getattr(cfg, "lambda_pure_tda", 2.0))

    if mode == "clip_tda_labelled":
        init_gtda_objects(cfg)


    steps_per_epoch = int(getattr(cfg, "steps_per_epoch", 1000))

    sampler = DistributedLabeledCollisionBatchSampler(
        labels=all_label_train,
        batch_size=int(cfg.batch_size),
        labeled_per_batch=int(getattr(cfg, "labeled_per_batch", 32)),
        m_per_class=int(getattr(cfg, "m_per_class", 2)),
        unlabeled_val=unlabeled_val,
        seed=int(getattr(cfg, "seed", 42)),
        drop_last=True,
        steps_per_epoch=steps_per_epoch,
        dataset_size=len(train_ds),
    )

    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        collate_fn=collate_fn_refalt,
        pin_memory=False,
    )


    start_epoch = 1
    best_loss = float("inf")

    if os.path.exists(latest_path):
        if rank == 0:
            ckpt = torch.load(latest_path, map_location=cfg.device)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = float(ckpt.get("best_loss", float("inf")))
            best_metric_val = float(ckpt.get("best_metric_val", best_metric_val))

            model.module.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])

            rank0_print(
                f"[RESUME] {latest_path} start_epoch={start_epoch} "
                f"best_loss={best_loss:.6f} best_metric_val={best_metric_val}"
            )

        dist.barrier()
        for p in model.module.parameters():
            dist.broadcast(p.data, src=0)
        dist.barrier()


    val_subset = make_train_subset(
        train_ds,
        max_items=int(getattr(cfg, "val_subset_size", 2000)),
        seed=int(getattr(cfg, "seed", 42)),
        min_pos_ratio=float(getattr(cfg, "min_pos_ratio_val", 0.1)),
    )

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        sampler.set_epoch(epoch)
        model.train()

        t0 = time.time()
        epoch_loss = 0.0
        n_steps = 0

        lambda_tda = (
            linear_warmup(epoch, tda_warmup_epochs, lambda_tda_max)
            if mode == "clip_tda_labelled" else 0.0
        )

        for batch_idx, (idxs, seq_refs, seq_alts, bios, label) in enumerate(loader):
            bios = bios.to(cfg.device)
            check_nans(bios, "bios")

            labels = (
                label.to(cfg.device).long()
                if isinstance(label, torch.Tensor)
                else torch.tensor(label, device=cfg.device, dtype=torch.long)
            )

            z_seq, z_bio, h_tda, *_ = model.module.encode_views(seq_refs, seq_alts, bios)

            if mode == "clip_only":
                L_clip = clip_like_loss(z_seq, z_bio, temperature=temperature)
            else:
                L_clip = clip_soft_sirius_loss(
                    z_seq, z_bio,
                    labels=labels,
                    temperature=temperature,
                    alpha=alpha_soft,
                    unlabeled_val=unlabeled_val,
                )

            if mode == "clip_tda_labelled":
                tda_feat = pure_tda_feature_from_batch(cfg, h_tda, epoch=epoch, batch_idx=batch_idx).to(cfg.device)
                z_tda = model.module.tda_pure_head(tda_feat)  # (1, shared_dim)

                z_seq_pool = F.normalize(z_seq.mean(dim=0, keepdim=True), dim=1)
                z_bio_pool = F.normalize(z_bio.mean(dim=0, keepdim=True), dim=1)


                L_tda = (1.0 - (z_seq_pool * z_tda).sum()) + (1.0 - (z_bio_pool * z_tda).sum())
            else:
                L_tda = torch.zeros((), device=cfg.device)

            loss = L_clip + (lambda_tda * L_tda)

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    rank0_print(f"[SKIP] NaN/Inf loss at epoch={epoch} batch={batch_idx}")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_steps += 1

            if rank == 0 and batch_idx % int(getattr(cfg, "log_every", 10)) == 0:
                rank0_print(
                    f"Epoch {epoch:03d} | Batch {batch_idx:04d} | "
                    f"loss={loss.item():.4f} | clip={L_clip.item():.4f} | "
                    f"tda={(L_tda.item() if mode=='clip_tda_labelled' else 0.0):.4f} | "
                    f"lambda_tda={lambda_tda:.3f}"
                )
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        epoch, batch_idx,
                        float(loss.item()),
                        float(L_clip.item()),
                        float(L_tda.item()) if mode == "clip_tda_labelled" else 0.0,
                        float(lambda_tda),
                        float(time.time() - t0),
                    ])

        dist.barrier()
        avg_epoch_loss = epoch_loss / max(1, n_steps)


        do_eval = (epoch % eval_every == 0) or (epoch == start_epoch)
        metrics = {}

        if do_eval:
            model.eval()
            with torch.no_grad():
                out_dir = os.path.join(run_dir, "results", f"epoch_{epoch:03d}")
                os.makedirs(out_dir, exist_ok=True)

                metrics = run_all_results(
                    cfg=cfg,
                    model=model,
                    train_ds=None,
                    val_ds=val_subset,
                    test_ds=test_ds,
                    out_dir=out_dir,
                    epoch=epoch,
                )
            model.train()
        if rank == 0:
            ckpt_latest = {
                "epoch": int(epoch),
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": float(best_loss),
                "best_metric_key": best_metric_key,
                "best_metric_mode": best_metric_mode,
                "best_metric_val": float(best_metric_val),
                "run_name": run_name,
                "mode": mode,
            }
            torch.save(ckpt_latest, latest_path)

            rank0_print(f"\nEpoch {epoch:03d} | AvgLoss={avg_epoch_loss:.6f} | BestLoss={best_loss:.6f}\n")

            

            improved_loss = (avg_epoch_loss < best_loss)
            if improved_loss:
                best_loss = float(avg_epoch_loss)
                ckpt_best_loss = dict(ckpt_latest)
                ckpt_best_loss["best_loss"] = float(best_loss)
                ckpt_best_loss["best_metric_val"] = float(best_metric_val)

                torch.save(ckpt_best_loss, best_loss_path)
                rank0_print(f"[BEST_LOSS] loss={best_loss:.6f} saved: {best_loss_path}")

                if save_best_history:
                    hist = os.path.join(run_dir, f"best_loss__epoch_{epoch:03d}__loss_{best_loss:.6f}.pt")
                    torch.save(ckpt_best_loss, hist)

            if do_eval:
                mval = metrics.get(best_metric_key, None)
                improved_metric = False

                if mval is not None and isinstance(mval, (int, float)) and np.isfinite(mval):
                    mval = float(mval)
                    if best_metric_mode == "max":
                        improved_metric = (mval > best_metric_val)
                    else:
                        improved_metric = (mval < best_metric_val)

                    if improved_metric:
                        best_metric_val = float(mval)
                        ckpt_best_metric = dict(ckpt_latest)
                        ckpt_best_metric["best_loss"] = float(best_loss)
                        ckpt_best_metric["best_metric_val"] = float(best_metric_val)

                        torch.save(ckpt_best_metric, best_metric_path)
                        rank0_print(f"[BEST_METRIC] {best_metric_key}={best_metric_val:.6f} saved: {best_metric_path}")

                        if save_best_history:
                            # hist = os.path.join(
                            #     run_dir,
                            #     f"best_metric__{metric_tag}__epoch_{epoch:03d}__val_{best_metric_val:.6f}.pt"
                            # )
                            torch.save(ckpt_best_metric, hist)
                else:
                    rank0_print(f"[BEST_METRIC] metric missing/non-numeric: {best_metric_key} -> skipped")

        dist.barrier()

    if rank == 0:
        rank0_print("\n" + "=" * 80)
        rank0_print(f"RUN COMPLETED: {run_name}")
        rank0_print("=" * 80 + "\n")

    cleanup_distributed()
    return model.module



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "clip_only", "labelled_clip", "clip_tda_labelled"],
        help="Which experiment to run. Use 'all' to run sequentially in one job.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Output folder name under cfg.save_dir. Default is mode.",
    )
    args = parser.parse_args()

    cfg = Config()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        cfg.device = "cpu"

    set_seed(int(getattr(cfg, "seed", 42)))

    train_ds, test_ds, all_label_train = get_data_all(
        numeric_train=cfg.numeric_train_path,
        seq_train=cfg.seq_train_path,
        label_train=cfg.label_train_path,
        numeric_test=cfg.numeric_test_path,
        seq_test=cfg.seq_test_path,
        label_test=cfg.label_test_path,
        max_len=int(getattr(cfg, "max_len", 512)),
        max_train_items=getattr(cfg, "max_train_items", None),
        max_test_items=getattr(cfg, "max_test_items", 5000),
        reduce_seed=int(getattr(cfg, "reduce_seed", 42)),
        stratify_reduce=bool(getattr(cfg, "stratify_reduce", True)),
    )


    if args.mode == "all":
        for mode in ["clip_only", "labelled_clip", "clip_tda_labelled"]:
            rn = mode
            train_one_run_distributed(
                cfg=cfg,
                train_ds=train_ds,
                all_label_train=all_label_train,
                test_ds=test_ds,
                run_name=rn,
                mode=mode,
            )
    else:
        rn = args.run_name if args.run_name is not None else args.mode
        train_one_run_distributed(
            cfg=cfg,
            train_ds=train_ds,
            all_label_train=all_label_train,
            test_ds=test_ds,
            run_name=rn,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
