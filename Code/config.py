import os


class Config:
    def __init__(self):
        # ==================================================
        # Models
        # ==================================================
        self.dna_model_name: str = "zhihan1996/DNABERT-2-117M"
        self.text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

        self.dna_finetune: bool = True
        self.text_finetune: bool = True

        self.use_text: bool = True

        # ==================================================
        # Offline embeddings
        # ==================================================
        self.seq_offline_embeddings: bool = False
        self.text_offline_embeddings: bool = False

        self.precompute_batch_size: int = 32

        self.seq_emb_dim: int = 768
        self.text_emb_dim: int = 384   # MiniLM-L6-v2

        # ==================================================
        # Pooling / tokenization
        # ==================================================
        self.pool: str = "mean"
        self.text_pool: str = "mean"

        self.max_len: int = 512
        self.text_max_len: int = 256

        # ==================================================
        # Bio features
        # ==================================================
        self.bio_dim: int = None
        self.bio_hidden: int = 512#256
        self.bio_dim_out: int = 256

        # ==================================================
        # Mutation / fusion / projection
        # ==================================================
        self.mutation_dim: int = 768
        self.mutation_hidden: int = 1024
        self.dropout: float = 0.1

        self.shared_dim: int = 256#128
        self.proj_dim: int = 256#128

        # ==================================================
        # Training
        # ==================================================
        self.batch_size: int =  128#2048
        self.epochs: int = 1500

        self.lr_heads: float = 5e-4#1e-3#1e-4#1e-4
        self.lr_dnabert: float = 5e-6
        self.lr_text: float = 5e-6
        self.weight_decay: float = 1e-4#1e-4
        self.temperature: float = 0.07

        # ==================================================
        # Contrastive loss weights
        # ==================================================
        self.lambda_sb: float = 1.0
        self.lambda_st: float = 0.8#1.0
        self.lambda_bt: float = 0.8#1.0

        # Ontology target mixing
        self.alpha_onto: float = 0.3
        self.target_temp = 0.25
        self.zero_diag_onto = True
        self.eval_onto_threshold: float = 0.5

        # ==================================================
        # Misc
        # ==================================================
        self.seed: int = 42
        self.num_workers: int = 0
        self.device = "cpu"
        self.save_dir: str = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

        # Optional row limits
        self.max_bio_line = None
        self.max_bio_line_eval = None

        # ==================================================
        # TDA regularizer
        # ==================================================
        self.lambda_tda: float = 0#1e-6
        self.tda_rebuild_every: int = 0
        self.knn_k: int = 0
        self.diff_t: int = 3
        self.diff_block_size: int = 0#80000

        self.tda_batch_size: int = 0
        self.tda_random_subset: bool = True
        self.tda_every = 1

        # Separate diffusion files
        self.train_diffusion_file = os.path.join(self.save_dir, "D_target_train.npy")
        self.val_diffusion_file   = os.path.join(self.save_dir, "D_target_val.npy")

        # ==================================================
        # EMA teacher
        # ==================================================
        self.ema_decay: float = 0.99



        # ==================================================
        # Train data
        # ==================================================
        self.train_seq: str = os.path.expanduser("../../ML_DATA_GENOME/ML_SEQUENCES_TRAIN.tsv")
        self.train_bio: str = os.path.expanduser("../../ML_DATA_GENOME/ML_NUMERIC_TRAIN.tsv")
        self.train_onto: str = os.path.expanduser("../../ML_DATA_GENOME/ML_ONTO_TRAIN.tsv")
        self.train_text: str = os.path.expanduser("../../ML_DATA_GENOME/ML_TEXT_TRAIN.tsv")
        self.train_textfields: str = os.path.expanduser("../../ML_DATA_GENOME/ML_TEXTFIELDS_TRAIN.tsv")
        self.train_labels: str = os.path.expanduser("../../ML_DATA_GENOME/ML_LABELS_TRAIN.npy")

        # ==================================================
        # Validation / test data
        # ==================================================
        self.test_seq: str = os.path.expanduser("../../ML_DATA_GENOME/ML_SEQUENCES_EVAL500.tsv")
        self.test_bio: str = os.path.expanduser("../../ML_DATA_GENOME/ML_NUMERIC_EVAL500.tsv")
        self.test_onto: str = os.path.expanduser("../../ML_DATA_GENOME/ML_ONTO_EVAL500.tsv")
        self.test_text: str = os.path.expanduser("../../ML_DATA_GENOME/ML_TEXT_EVAL500.tsv")
        self.test_textfields: str = os.path.expanduser("../../ML_DATA_GENOME/ML_TEXTFIELDS_EVAL500.tsv")
        self.test_labels: str = os.path.expanduser("../../ML_DATA_GENOME/ML_LABELS_EVAL500.npy")


        # ==================================================
        # Downstream task
        # ==================================================
        self.evaluate_seq: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_SEQUENCES_ALL.tsv")
        self.evaluate_bio: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_NUMERIC_ALL.tsv")
        self.evaluate_onto: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_ONTO_ALL.tsv")
        self.evaluate_text: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_TEXT_ALL.tsv")
        self.evaluate_textfields: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_TEXTFIELDS_ALL.tsv")
        self.evaluate_labels: str = os.path.expanduser("../../ML_DATA_GENOME/Evaluation/ML_LABELS_ALL.npy")

        # ==================================================
        # Embedding output paths
        # ==================================================
        self.embedding_dir: str = "embeddings"
        os.makedirs(self.embedding_dir, exist_ok=True)

        self.train_ref_emb_path: str = os.path.join(self.embedding_dir, "train_ref_emb.pt")
        self.train_alt_emb_path: str = os.path.join(self.embedding_dir, "train_alt_emb.pt")
        self.train_text_emb_path: str = os.path.join(self.embedding_dir, "train_text_emb.pt")

        self.val_ref_emb_path: str = os.path.join(self.embedding_dir, "val_ref_emb.pt")
        self.val_alt_emb_path: str = os.path.join(self.embedding_dir, "val_alt_emb.pt")
        self.val_text_emb_path: str = os.path.join(self.embedding_dir, "val_text_emb.pt")

    def __repr__(self):
        keys = sorted(self.__dict__.keys())
        body = ", ".join(f"{k}={self.__dict__[k]!r}" for k in keys)
        return f"Config({body})"