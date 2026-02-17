# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:


    save_dir: str = "./PolyClip-T"

    numeric_train_path: str = "../clip_ready/X_train.tsv"
    seq_train_path: str = "../clip_ready/train_sequences.tsv"
    label_train_path: str = "../clip_ready/y_train.npy"


    numeric_test_path: Optional[str] = "../clip_ready/X_val.tsv"
    seq_test_path: Optional[str] = "../clip_ready/val_sequences.tsv"
    label_test_path: str = "../clip_ready/y_val.npy"


    eval_every: int = 1
    val_subset_size: int = 2000
    min_pos_ratio_val: float = 0.10

    best_metric_key: str = "val/task/auprc"     
    best_metric_mode: str = "max"              
    save_best_history: bool = True             

    eval_test_every: int = 1  

    max_train_items = None        
    max_test_items  = 5000        
    reduce_seed     = 42
    stratify_reduce = True        

    device: str = os.getenv("POLYGENY_DEVICE", "cuda")
    seed: int = 42



    batch_size: int = 1024
    num_workers: int = 2
    steps_per_epoch: int = 1000
    log_every: int = 10


    dna_model_name: str = "zhihan1996/DNABERT-2-117M"
    dna_finetune: bool = False

    lr_dnabert: float = 1e-5
    max_len: int = 512
    pool: str = "mean"          # "mean" | "cls"


   
    bio_dim: int = 84#58 #256

    bio_hidden: int = 512
    bio_dropout: float = 0.1


 
    proj_dim: int = 128         # CLIP projections (z_seq, z_bio)
    shared_dim: int = 128       # fused / TDA representation



    epochs: int = 20

    lr_heads: float = 1e-3
    weight_decay: float = 1e-4



    temperature: float = 0.07

    unlabeled_val: int = -1     # unlabeled marker
    alpha_soft: float = 0.25    # soft-label mass



    labeled_per_batch: int = 32
    m_per_class: int = 2


    tda_warmup_epochs: int = 8
    lambda_pure_tda: float = 2.0

    tda_hom_dims: Tuple[int, ...] = (0, 1)
    tda_n_jobs: int = 1

    tda_pi_bins: int = 32
    tda_pi_sigma: float = 0.10
    tda_subsample: int = 256


    tda_pi_dim: int = 2048



    rep: str = "z_bio"          # "z_seq" | "z_bio" | "h_tda" | "fused"

    eval_knn_k: int = 10

    eval_every: int = 1



    val_subset_size: int = 1500
    min_pos_ratio_val: float = 0.1



    detect_nan: bool = True
    save_every: int = 1
