from __future__ import annotations

from libs import *
from config import Config
from loss import l2norm




class DNABertEncoder(nn.Module):

    def __init__(self, model_name: str, finetune: bool, pool: str = "mean", max_len: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
        conf = BertConfig.from_pretrained(model_name, cache_dir="./tmp")
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, config=conf, attn_implementation="eager"
        )

        self.pool = pool
        self.max_len = max_len

        if not finetune:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Partial fine-tuning: last layer + pooler only
            for name, param in self.backbone.named_parameters():
                if ("encoder.layer.11" in name) or ("pooler" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, seqs: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            seqs, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(next(self.backbone.parameters()).device)

        outputs = self.backbone(**inputs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state  # [B,L,768]

        if self.pool == "mean":
            return hidden.mean(dim=1)
        if self.pool == "cls":
            return hidden[:, 0, :]
        raise ValueError("pool must be 'mean' or 'cls'")


class BioEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """
    Joint representation head used for TDA branch and inference embedding.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class InstanceProjector(nn.Module):
    """
    Projection head for CLIP embeddings (not normalized inside).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PureTDAHead(nn.Module):
    """
    Maps a persistence vector (e.g., PI flattened) to an embedding space.
    Output is normalized.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.normalize(self.net(x), dim=1)



class TCL_TDA_Model_RefAlt(nn.Module):
    """
    Backbone outputs:
      - z_seq: normalized CLIP embedding for sequence view
      - z_bio: normalized CLIP embedding for bio view
      - h_tda: fused representation (NOT normalized) for TDA and inference
      - h_seq, h_bio: raw view embeddings

    This class is used by:
      - CLIP-only
      - Labelled CLIP
      - CLIP + TDA + Labelled
    via different losses in trainning.py (not here).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # DNA encoder: each produces 768
        self.dna = DNABertEncoder(
            model_name=cfg.dna_model_name,
            finetune=cfg.dna_finetune,
            pool=cfg.pool,
            max_len=cfg.max_len,
        )

        # Bio encoder -> 256
        self.bio = BioEncoder(
            in_dim=cfg.bio_dim,
            hidden=cfg.bio_hidden,
            out_dim=256,
            dropout=float(getattr(cfg, "bio_dropout", 0.1)),
        )

        # Projection heads for CLIP
        self.seq_head = InstanceProjector(in_dim=768 * 2, out_dim=cfg.proj_dim)
        self.bio_head = InstanceProjector(in_dim=256, out_dim=cfg.proj_dim)

        # Fused embedding for TDA/inference
        self.fusion = FusionHead(in_dim=768 * 2 + 256, out_dim=cfg.shared_dim)

        # TDA head (PI -> shared_dim) (only used when you compute L_pure_tda)
        self.tda_pure_head = PureTDAHead(
            in_dim=int(getattr(cfg, "tda_pi_dim", 2048)),
            out_dim=int(getattr(cfg, "shared_dim", cfg.shared_dim)),
            hidden=int(getattr(cfg, "tda_pure_hidden", 512)),
            dropout=float(getattr(cfg, "tda_pure_dropout", 0.1)),
        )

    def encode_views(
        self,
        seq_ref: List[str],
        seq_alt: List[str],
        bio_feats: torch.Tensor,
    ):
        """
        Returns:
          z_seq: (B, proj_dim) normalized
          z_bio: (B, proj_dim) normalized
          h_tda: (B, shared_dim) fused (not normalized)
          h_seq: (B, 1536)
          h_bio: (B, 256)
          h_ref: (B, 768)
          h_alt: (B, 768)
        """
        h_ref = self.dna(seq_ref)        # (B,768)
        h_alt = self.dna(seq_alt)        # (B,768)
        h_bio = self.bio(bio_feats)      # (B,256)

        h_seq = torch.cat([h_ref, h_alt], dim=1)                 # (B,1536)
        fused = torch.cat([h_ref, h_alt, h_bio], dim=1)          # (B,1792)
        h_tda = self.fusion(fused)                               # (B,shared_dim)

        z_seq = l2norm(self.seq_head(h_seq), dim=1)              # (B,proj_dim)
        z_bio = l2norm(self.bio_head(h_bio), dim=1)              # (B,proj_dim)

        return z_seq, z_bio, h_tda, h_seq, h_bio, h_ref, h_alt

    @torch.no_grad()
    def encode_fused(self, seq_ref: List[str], seq_alt: List[str], bio_feats: torch.Tensor) -> torch.Tensor:
        """
        Fused embedding for inference/retrieval (no proj, no norm).
        """
        h_ref = self.dna(seq_ref)
        h_alt = self.dna(seq_alt)
        h_bio = self.bio(bio_feats)
        fused = torch.cat([h_ref, h_alt, h_bio], dim=1)
        return self.fusion(fused)  # (B,shared_dim)

    @torch.no_grad()
    def encode_sequence(self, seq_ref: List[str], seq_alt: List[str]) -> torch.Tensor:
        """
        CLIP sequence embedding (normalized).
        """
        h_ref = self.dna(seq_ref)
        h_alt = self.dna(seq_alt)
        h_seq = torch.cat([h_ref, h_alt], dim=1)
        return l2norm(self.seq_head(h_seq), dim=1)

    @torch.no_grad()
    def encode_bio(self, bio_feats: torch.Tensor) -> torch.Tensor:
        """
        CLIP bio embedding (normalized).
        """
        h_bio = self.bio(bio_feats)
        return l2norm(self.bio_head(h_bio), dim=1)



class EMAWrapper(nn.Module):
    """
    Exponential Moving Average teacher.
    Use:
      ema = EMAWrapper(model, decay=0.99)
      ema.update(model)
    """
    def __init__(self, model: nn.Module, decay: float = 0.99):
        super().__init__()
        self.decay = float(decay)

        base = model.module if isinstance(model, DDP) else model
        self.teacher = type(base)(base.cfg)
        self.teacher.load_state_dict(base.state_dict(), strict=True)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        base = model.module if isinstance(model, DDP) else model
        for t_param, s_param in zip(self.teacher.parameters(), base.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def to(self, *args, **kwargs):
        self.teacher.to(*args, **kwargs)
        return self
