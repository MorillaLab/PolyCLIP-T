#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline sequence embeddings + lightweight fusion training pipeline.

This file implements a practical 3-stage workflow:

Stage A: Precompute REF/ALT sequence embeddings once with a transformer backbone
Stage B: Train a lightweight fusion/projection/classification model on saved embeddings + bio features
Stage C: Optional scaffold for later backbone fine-tuning on a selected subset

Designed for large genomic variant datasets where running the sequence encoder
inside every training step is too expensive.

Expected input tables
---------------------
Sequence TSV must contain at least:
    ROW_ID, REF_SEQ, ALT_SEQ

Bio TSV should contain at least:
    ROW_ID, <numeric features...>, [LABEL]

Metadata alignment rule
-----------------------
ROW_ID must be stable and unique across sequence/bio/label files.
This is critical. Never rely on implicit row order after filtering.

Example usage
-------------
1) Precompute embeddings
python offline_embeddings_pipeline.py precompute \
    --seq-tsv train_seq.tsv \
    --out-dir PRECOMP/train \
    --model-name zhihan1996/DNABERT-2-117M \
    --batch-size 32 \
    --max-length 256

2) Train lightweight model
python offline_embeddings_pipeline.py train \
    --emb-dir PRECOMP/train \
    --bio-tsv train_bio.tsv \
    --val-emb-dir PRECOMP/val \
    --val-bio-tsv val_bio.tsv \
    --label-col LABEL \
    --epochs 20 \
    --batch-size 512 \
    --lr 1e-3
"""

from __future__ import annotations

import os
import gc
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def masked_mean_pool(last_hidden_state: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    x = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return x.sum(dim=1) / denom


def infer_device(explicit_device: Optional[str] = None) -> str:
    if explicit_device:
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Sequence precompute dataset
# =========================================================

class SeqOnlyDataset(Dataset):
    """Loads only ROW_ID / REF_SEQ / ALT_SEQ for offline embedding extraction."""

    def __init__(self, seq_tsv: str,
                 row_id_col: str = "ROW_ID",
                 ref_col: str = "REF_SEQ",
                 alt_col: str = "ALT_SEQ") -> None:
        self.df = pd.read_csv(seq_tsv, sep="\t")

        missing = [c for c in [row_id_col, ref_col, alt_col] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in seq_tsv: {missing}")

        self.row_ids = self.df[row_id_col].to_numpy(dtype=np.int64)
        self.refs = self.df[ref_col].astype(str).tolist()
        self.alts = self.df[alt_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "row_id": int(self.row_ids[idx]),
            "ref": self.refs[idx],
            "alt": self.alts[idx],
        }


def collate_seq(batch: List[Dict]) -> Dict[str, List]:
    return {
        "row_id": [b["row_id"] for b in batch],
        "ref": [b["ref"] for b in batch],
        "alt": [b["alt"] for b in batch],
    }


# =========================================================
# Encoder wrapper
# =========================================================

class SequenceEmbedder:
    """Transformer-based sequence embedder with masked mean pooling."""

    def __init__(self,
                 model_name: str,
                 device: Optional[str] = None,
                 max_length: int = 256,
                 cache_dir: Optional[str] = None,
                 trust_remote_code: bool = True,
                 use_amp: bool = True) -> None:
        self.model_name = model_name
        self.device = infer_device(device)
        self.max_length = max_length
        self.use_amp = use_amp and self.device.startswith("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(**tok)
        else:
            out = self.model(**tok)

        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = masked_mean_pool(hidden, tok["attention_mask"])
        return pooled.detach().cpu()

    @torch.no_grad()
    def embedding_dim(self) -> int:
        test = self.encode_texts(["ACGTACGTACGT"])
        return int(test.shape[1])


# =========================================================
# Stage A: precompute embeddings to memmap
# =========================================================

def precompute_embeddings(seq_tsv: str,
                          out_dir: str,
                          model_name: str,
                          row_id_col: str = "ROW_ID",
                          ref_col: str = "REF_SEQ",
                          alt_col: str = "ALT_SEQ",
                          batch_size: int = 32,
                          num_workers: int = 4,
                          max_length: int = 256,
                          dtype: str = "float16",
                          device: Optional[str] = None,
                          cache_dir: Optional[str] = None,
                          trust_remote_code: bool = True,
                          use_amp: bool = True) -> None:
    ensure_dir(out_dir)

    ds = SeqOnlyDataset(
        seq_tsv=seq_tsv,
        row_id_col=row_id_col,
        ref_col=ref_col,
        alt_col=alt_col,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_seq,
    )

    embedder = SequenceEmbedder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_amp=use_amp,
    )

    emb_dim = embedder.embedding_dim()
    n = len(ds)
    np_dtype = np.float16 if dtype == "float16" else np.float32

    ref_path = os.path.join(out_dir, "ref_emb.memmap")
    alt_path = os.path.join(out_dir, "alt_emb.memmap")
    meta_path = os.path.join(out_dir, "meta.parquet")
    cfg_path = os.path.join(out_dir, "precompute_config.json")

    ref_mm = np.memmap(ref_path, mode="w+", dtype=np_dtype, shape=(n, emb_dim))
    alt_mm = np.memmap(alt_path, mode="w+", dtype=np_dtype, shape=(n, emb_dim))

    seq_df = pd.read_csv(seq_tsv, sep="\t")
    seq_df = seq_df[[row_id_col, ref_col, alt_col]].copy()
    seq_df["__ROW_POS__"] = np.arange(len(seq_df), dtype=np.int64)

    row_id_to_pos = dict(zip(seq_df[row_id_col].astype(np.int64), seq_df["__ROW_POS__"].astype(np.int64)))

    print(f"[INFO] Precomputing embeddings for n={n}, emb_dim={emb_dim}, device={embedder.device}")

    for step, batch in enumerate(dl, start=1):
        row_ids = batch["row_id"]
        pos = np.array([row_id_to_pos[int(x)] for x in row_ids], dtype=np.int64)

        ref_emb = embedder.encode_texts(batch["ref"]).numpy().astype(np_dtype, copy=False)
        alt_emb = embedder.encode_texts(batch["alt"]).numpy().astype(np_dtype, copy=False)

        ref_mm[pos] = ref_emb
        alt_mm[pos] = alt_emb

        if step % 20 == 0:
            print(f"[INFO] Processed {min(step * batch_size, n)}/{n}")

    ref_mm.flush()
    alt_mm.flush()

    meta_df = seq_df[[row_id_col, ref_col, alt_col, "__ROW_POS__"]].copy()
    meta_df.to_parquet(meta_path, index=False)

    save_json(
        {
            "n_rows": n,
            "emb_dim": emb_dim,
            "dtype": dtype,
            "row_id_col": row_id_col,
            "ref_col": ref_col,
            "alt_col": alt_col,
            "model_name": model_name,
            "max_length": max_length,
            "ref_memmap": os.path.basename(ref_path),
            "alt_memmap": os.path.basename(alt_path),
            "meta_parquet": os.path.basename(meta_path),
        },
        cfg_path,
    )

    print(f"[DONE] Saved offline embeddings to: {out_dir}")


# =========================================================
# Stage B dataset: precomputed embeddings + bio features
# =========================================================

class PrecomputedVariantDataset(Dataset):
    """
    Dataset backed by memmap embeddings and a bio TSV.

    Returns:
        ref_emb, alt_emb, delta_emb, bio, label
    """

    def __init__(self,
                 emb_dir: str,
                 bio_tsv: str,
                 label_col: str = "LABEL",
                 row_id_col: str = "ROW_ID",
                 drop_cols: Optional[List[str]] = None,
                 dtype_embeddings: str = "float16") -> None:
        self.emb_dir = emb_dir
        self.bio_tsv = bio_tsv
        self.label_col = label_col
        self.row_id_col = row_id_col
        self.drop_cols = drop_cols or []

        cfg = load_json(os.path.join(emb_dir, "precompute_config.json"))
        self.n = int(cfg["n_rows"])
        self.emb_dim = int(cfg["emb_dim"])
        self.emb_dtype = np.float16 if dtype_embeddings == "float16" else np.float32

        self.meta = pd.read_parquet(os.path.join(emb_dir, cfg["meta_parquet"]))
        if row_id_col not in self.meta.columns:
            raise ValueError(f"{row_id_col} missing from meta parquet")
        if "__ROW_POS__" not in self.meta.columns:
            raise ValueError("__ROW_POS__ missing from meta parquet")

        bio_df = pd.read_csv(bio_tsv, sep="\t")
        if row_id_col not in bio_df.columns:
            raise ValueError(f"{row_id_col} missing from bio_tsv")

        merged = self.meta[[row_id_col, "__ROW_POS__"]].merge(
            bio_df,
            on=row_id_col,
            how="inner",
            validate="one_to_one",
        )
        merged = merged.sort_values("__ROW_POS__").reset_index(drop=True)

        if len(merged) == 0:
            raise ValueError("No rows after merging embeddings metadata with bio_tsv")

        self.row_pos = merged["__ROW_POS__"].to_numpy(dtype=np.int64)
        self.row_ids = merged[row_id_col].to_numpy(dtype=np.int64)

        if label_col in merged.columns:
            self.labels = merged[label_col].to_numpy()
        else:
            self.labels = np.zeros(len(merged), dtype=np.int64)

        exclude = {row_id_col, "__ROW_POS__", label_col, *self.drop_cols}
        bio_cols = [c for c in merged.columns if c not in exclude]
        if not bio_cols:
            raise ValueError("No biological feature columns left after exclusions")

        self.bio_cols = bio_cols
        self.bio = merged[bio_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        self.ref_mm = np.memmap(
            os.path.join(emb_dir, cfg["ref_memmap"]),
            mode="r",
            dtype=self.emb_dtype,
            shape=(self.n, self.emb_dim),
        )
        self.alt_mm = np.memmap(
            os.path.join(emb_dir, cfg["alt_memmap"]),
            mode="r",
            dtype=self.emb_dtype,
            shape=(self.n, self.emb_dim),
        )

    def __len__(self) -> int:
        return len(self.row_pos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pos = int(self.row_pos[idx])
        ref = np.asarray(self.ref_mm[pos], dtype=np.float32)
        alt = np.asarray(self.alt_mm[pos], dtype=np.float32)
        delta = alt - ref

        label = self.labels[idx]
        if isinstance(label, (np.floating, float)):
            label_tensor = torch.tensor(float(label), dtype=torch.float32)
        elif isinstance(label, (np.integer, int)):
            label_tensor = torch.tensor(int(label), dtype=torch.long)
        else:
            # fallback for string/inconsistent labels; caller should clean upstream if needed
            try:
                label_tensor = torch.tensor(int(label), dtype=torch.long)
            except Exception:
                label_tensor = torch.tensor(0, dtype=torch.long)

        return {
            "ref_emb": torch.from_numpy(ref),
            "alt_emb": torch.from_numpy(alt),
            "delta_emb": torch.from_numpy(delta),
            "bio": torch.from_numpy(self.bio[idx]),
            "label": label_tensor,
        }


# =========================================================
# Lightweight models
# =========================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightFusionModel(nn.Module):
    """
    Sequence side uses [ref_emb, alt_emb, delta_emb].
    Bio side uses numeric features.

    Outputs:
        z_seq   : normalized projected seq representation
        z_bio   : normalized projected bio representation
        z_fused : normalized projected fused representation
        logits  : classification logits
    """

    def __init__(self,
                 seq_dim: int,
                 bio_dim: int,
                 seq_hidden: int = 512,
                 bio_hidden: int = 256,
                 fused_hidden: int = 512,
                 proj_dim: int = 256,
                 n_classes: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.seq_encoder = MLP(in_dim=3 * seq_dim, hidden_dims=[seq_hidden], dropout=dropout)
        self.bio_encoder = MLP(in_dim=bio_dim, hidden_dims=[bio_hidden], dropout=dropout)

        self.seq_proj = nn.Linear(self.seq_encoder.out_dim, proj_dim)
        self.bio_proj = nn.Linear(self.bio_encoder.out_dim, proj_dim)

        self.fused = MLP(
            in_dim=self.seq_encoder.out_dim + self.bio_encoder.out_dim,
            hidden_dims=[fused_hidden],
            dropout=dropout,
        )
        self.fused_proj = nn.Linear(self.fused.out_dim, proj_dim)
        self.classifier = nn.Linear(self.fused.out_dim, n_classes)

    def forward(self,
                ref_emb: torch.Tensor,
                alt_emb: torch.Tensor,
                delta_emb: torch.Tensor,
                bio: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_x = torch.cat([ref_emb, alt_emb, delta_emb], dim=-1)
        h_seq = self.seq_encoder(seq_x)
        h_bio = self.bio_encoder(bio)

        z_seq = F.normalize(self.seq_proj(h_seq), dim=-1)
        z_bio = F.normalize(self.bio_proj(h_bio), dim=-1)

        h_fused = self.fused(torch.cat([h_seq, h_bio], dim=-1))
        z_fused = F.normalize(self.fused_proj(h_fused), dim=-1)
        logits = self.classifier(h_fused)

        return {
            "z_seq": z_seq,
            "z_bio": z_bio,
            "z_fused": z_fused,
            "logits": logits,
        }


# =========================================================
# Losses
# =========================================================

def clip_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE between two aligned views."""
    logits = (z1 @ z2.T) / tau
    targets = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_12 + loss_21)


def classification_loss(logits: torch.Tensor,
                        labels: torch.Tensor,
                        n_classes: int = 2) -> torch.Tensor:
    if labels.dtype in (torch.float16, torch.float32, torch.float64) and n_classes == 1:
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
    return F.cross_entropy(logits, labels.long())


# =========================================================
# Metrics
# =========================================================

@torch.no_grad()
def multiclass_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


# =========================================================
# Training / evaluation
# =========================================================

@dataclass
class TrainConfig:
    train_emb_dir: str
    train_bio_tsv: str
    val_emb_dir: Optional[str] = None
    val_bio_tsv: Optional[str] = None
    label_col: str = "LABEL"
    row_id_col: str = "ROW_ID"
    batch_size: int = 512
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    seq_hidden: int = 512
    bio_hidden: int = 256
    fused_hidden: int = 512
    proj_dim: int = 256
    tau: float = 0.07
    lambda_clip: float = 1.0
    lambda_cls: float = 1.0
    out_dir: str = "runs/offline_fusion"
    seed: int = 42
    device: Optional[str] = None


def build_loaders(cfg: TrainConfig) -> Tuple[PrecomputedVariantDataset, DataLoader, Optional[PrecomputedVariantDataset], Optional[DataLoader]]:
    train_ds = PrecomputedVariantDataset(
        emb_dir=cfg.train_emb_dir,
        bio_tsv=cfg.train_bio_tsv,
        label_col=cfg.label_col,
        row_id_col=cfg.row_id_col,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_ds = None
    val_dl = None
    if cfg.val_emb_dir and cfg.val_bio_tsv:
        val_ds = PrecomputedVariantDataset(
            emb_dir=cfg.val_emb_dir,
            bio_tsv=cfg.val_bio_tsv,
            label_col=cfg.label_col,
            row_id_col=cfg.row_id_col,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_ds, train_dl, val_ds, val_dl


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    tau: float,
                    lambda_clip: float,
                    lambda_cls: float) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_clip = 0.0
    total_cls = 0.0
    total_acc = 0.0
    total_n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            ref_emb=batch["ref_emb"],
            alt_emb=batch["alt_emb"],
            delta_emb=batch["delta_emb"],
            bio=batch["bio"],
        )

        l_clip = clip_loss(out["z_seq"], out["z_bio"], tau=tau)
        l_cls = classification_loss(out["logits"], batch["label"], n_classes=out["logits"].shape[1])
        loss = lambda_clip * l_clip + lambda_cls * l_cls

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = batch["label"].shape[0]
        total_loss += float(loss.item()) * bs
        total_clip += float(l_clip.item()) * bs
        total_cls += float(l_cls.item()) * bs
        total_acc += multiclass_accuracy(out["logits"], batch["label"]) * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "clip_loss": total_clip / max(total_n, 1),
        "cls_loss": total_cls / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str,
             tau: float,
             lambda_clip: float,
             lambda_cls: float) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_clip = 0.0
    total_cls = 0.0
    total_acc = 0.0
    total_n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            ref_emb=batch["ref_emb"],
            alt_emb=batch["alt_emb"],
            delta_emb=batch["delta_emb"],
            bio=batch["bio"],
        )

        l_clip = clip_loss(out["z_seq"], out["z_bio"], tau=tau)
        l_cls = classification_loss(out["logits"], batch["label"], n_classes=out["logits"].shape[1])
        loss = lambda_clip * l_clip + lambda_cls * l_cls

        bs = batch["label"].shape[0]
        total_loss += float(loss.item()) * bs
        total_clip += float(l_clip.item()) * bs
        total_cls += float(l_cls.item()) * bs
        total_acc += multiclass_accuracy(out["logits"], batch["label"]) * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "clip_loss": total_clip / max(total_n, 1),
        "cls_loss": total_cls / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
    }


def run_train(cfg: TrainConfig) -> None:
    ensure_dir(cfg.out_dir)
    save_json(asdict(cfg), os.path.join(cfg.out_dir, "train_config.json"))
    seed_everything(cfg.seed)

    device = infer_device(cfg.device)
    train_ds, train_dl, val_ds, val_dl = build_loaders(cfg)

    model = LightweightFusionModel(
        seq_dim=train_ds.emb_dim,
        bio_dim=train_ds.bio.shape[1],
        seq_hidden=cfg.seq_hidden,
        bio_hidden=cfg.bio_hidden,
        fused_hidden=cfg.fused_hidden,
        proj_dim=cfg.proj_dim,
        n_classes=2,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history: List[Dict] = []
    best_metric = -float("inf")
    best_path = os.path.join(cfg.out_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            tau=cfg.tau,
            lambda_clip=cfg.lambda_clip,
            lambda_cls=cfg.lambda_cls,
        )

        row = {"epoch": epoch, "split": "train", **train_metrics}
        history.append(row)
        print(f"[Epoch {epoch:03d}] TRAIN {train_metrics}")

        score = train_metrics["acc"]
        if val_dl is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_dl,
                device=device,
                tau=cfg.tau,
                lambda_clip=cfg.lambda_clip,
                lambda_cls=cfg.lambda_cls,
            )
            history.append({"epoch": epoch, "split": "val", **val_metrics})
            print(f"[Epoch {epoch:03d}] VAL   {val_metrics}")
            score = val_metrics["acc"]

        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "train_config": asdict(cfg),
                    "seq_dim": train_ds.emb_dim,
                    "bio_dim": int(train_ds.bio.shape[1]),
                    "bio_cols": train_ds.bio_cols,
                    "best_metric": best_metric,
                },
                best_path,
            )
            print(f"[INFO] Saved new best checkpoint -> {best_path}")

        pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, "history.csv"), index=False)

    print(f"[DONE] Training finished. Best metric={best_metric:.6f}")


# =========================================================
# Stage C scaffold: optional later fine-tuning
# =========================================================

def fine_tuning_scaffold() -> None:
    """
    Placeholder scaffold for later end-to-end refinement.

    Typical strategy:
      1. Train the lightweight model to convergence with frozen/offline embeddings.
      2. Select a subset of variants for refinement:
           - top candidates
           - hard negatives
           - mislabeled / borderline cases
           - recurrent variants across families
      3. Reattach the sequence backbone and initialize the fusion head from the
         lightweight model checkpoint.
      4. Fine-tune only the last transformer layers first, then optionally unfreeze more.

    This stage is intentionally not fully implemented here because it depends on
    your exact end-to-end architecture and training codebase.
    """
    print("Use the saved lightweight checkpoint to initialize your refinement stage.")


# =========================================================
# CLI
# =========================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline embeddings + lightweight fusion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("precompute", help="Precompute REF/ALT embeddings")
    p_pre.add_argument("--seq-tsv", required=True)
    p_pre.add_argument("--out-dir", required=True)
    p_pre.add_argument("--model-name", required=True)
    p_pre.add_argument("--row-id-col", default="ROW_ID")
    p_pre.add_argument("--ref-col", default="REF_SEQ")
    p_pre.add_argument("--alt-col", default="ALT_SEQ")
    p_pre.add_argument("--batch-size", type=int, default=32)
    p_pre.add_argument("--num-workers", type=int, default=4)
    p_pre.add_argument("--max-length", type=int, default=256)
    p_pre.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p_pre.add_argument("--device", default=None)
    p_pre.add_argument("--cache-dir", default=None)
    p_pre.add_argument("--no-trust-remote-code", action="store_true")
    p_pre.add_argument("--no-amp", action="store_true")

    p_train = sub.add_parser("train", help="Train lightweight fusion model")
    p_train.add_argument("--emb-dir", required=True)
    p_train.add_argument("--bio-tsv", required=True)
    p_train.add_argument("--val-emb-dir", default=None)
    p_train.add_argument("--val-bio-tsv", default=None)
    p_train.add_argument("--label-col", default="LABEL")
    p_train.add_argument("--row-id-col", default="ROW_ID")
    p_train.add_argument("--batch-size", type=int, default=512)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--seq-hidden", type=int, default=512)
    p_train.add_argument("--bio-hidden", type=int, default=256)
    p_train.add_argument("--fused-hidden", type=int, default=512)
    p_train.add_argument("--proj-dim", type=int, default=256)
    p_train.add_argument("--tau", type=float, default=0.07)
    p_train.add_argument("--lambda-clip", type=float, default=1.0)
    p_train.add_argument("--lambda-cls", type=float, default=1.0)
    p_train.add_argument("--out-dir", default="runs/offline_fusion")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", default=None)

    p_ft = sub.add_parser("finetune_scaffold", help="Print guidance for later end-to-end refinement")
    _ = p_ft

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "precompute":
        precompute_embeddings(
            seq_tsv=args.seq_tsv,
            out_dir=args.out_dir,
            model_name=args.model_name,
            row_id_col=args.row_id_col,
            ref_col=args.ref_col,
            alt_col=args.alt_col,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            dtype=args.dtype,
            device=args.device,
            cache_dir=args.cache_dir,
            trust_remote_code=not args.no_trust_remote_code,
            use_amp=not args.no_amp,
        )
        return

    if args.command == "train":
        cfg = TrainConfig(
            train_emb_dir=args.emb_dir,
            train_bio_tsv=args.bio_tsv,
            val_emb_dir=args.val_emb_dir,
            val_bio_tsv=args.val_bio_tsv,
            label_col=args.label_col,
            row_id_col=args.row_id_col,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seq_hidden=args.seq_hidden,
            bio_hidden=args.bio_hidden,
            fused_hidden=args.fused_hidden,
            proj_dim=args.proj_dim,
            tau=args.tau,
            lambda_clip=args.lambda_clip,
            lambda_cls=args.lambda_cls,
            out_dir=args.out_dir,
            seed=args.seed,
            device=args.device,
        )
        run_train(cfg)
        return

    if args.command == "finetune_scaffold":
        fine_tuning_scaffold()
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
