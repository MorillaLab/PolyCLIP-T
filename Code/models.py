from libs import *
from config import *
from utils import *
from loss import *


class DNABertEncoder(nn.Module):
    def __init__(self, model_name: str, finetune: bool, pool: str = "mean", max_len: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
        conf = BertConfig.from_pretrained(model_name, cache_dir="./tmp")
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=conf, attn_implementation="eager")
        self.pool = pool
        self.max_len = max_len
        if not finetune:
            # Full freeze — used when dna_finetune=False
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Partial fine-tuning: freeze all except the last block
            for name, param in self.backbone.named_parameters():
                # Unfreeze only the last layer (layer.11 for BERT-base)
                if "encoder.layer.11" in name or "pooler" in name:
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
            return hidden.mean(dim=1)  # [B,768]
        elif self.pool == "cls":
            return hidden[:, 0, :]
        raise ValueError("pool must be 'mean' or 'cls'")


class BioEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FusionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor):
        return self.fc(x)

class InstanceProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, h: torch.Tensor):
        return self.net(h)




class TCL_TDA_Model_RefAlt(nn.Module):
    """
        Two-view model:
        - Sequence view: DNABERT(ref) + DNABERT(alt) concatenated
        - Bio view: MLP(bio_feats)

        Outputs:
        - z_seq: normalized embedding for sequence contrastive learning
        - z_bio: normalized embedding for bio contrastive learning
        - h_tda: fused embedding used ONLY for TDA regularization
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # DNA encoder produces 768-dim embeddings
        self.dna = DNABertEncoder(
            cfg.dna_model_name,
            cfg.dna_finetune,
            cfg.pool,
            cfg.max_len,
        )                      

        # Bio encoder produces 256-dim embeddings
        self.bio = BioEncoder(
            cfg.bio_dim,
            cfg.bio_hidden,
            256,
        )                     

        # Fusion head (joint sequence + bio representation)
        # Input: [h_ref (768), h_alt (768), h_bio (256)] i.e 768*2 + 256
        self.fusion = FusionHead(
            768 * 2 + 256,      # [h_ref, h_alt, h_bio]
            cfg.shared_dim      
        )

        # Fusion head (joint sequence + bio representation)
        # Input: [h_ref (768), h_alt (768), h_bio (256)] → 768*2 + 256
        self.seq_head = InstanceProjector(
            in_dim=768 * 2,
            out_dim=cfg.proj_dim
        )

        # Bio view: 256 → proj_dim
        self.bio_head = InstanceProjector(
            in_dim=256,
            out_dim=cfg.proj_dim
        )

        self.tda_pure_head = PureTDAHead(
            in_dim=getattr(cfg, "tda_pi_dim", 2048),
            out_dim=getattr(cfg, "dim_shared"),
            hidden=getattr(cfg, "tda_pure_hidden", 512),
            dropout=getattr(cfg, "tda_pure_dropout", 0.1)
        )
        self.cross = CrossPredictor(768*2, 256,
                                           hidden=getattr(cfg, "cross_hidden", 512),
                                           dropout=getattr(cfg, "cross_dropout", 0.1)).to(cfg.device)

    def encode_views(
        self,
        seq_ref: List[str],
        seq_alt: List[str],
        bio_feats: torch.Tensor
    ):
        """
        Returns:
        - z_seq: normalized contrastive embedding for the sequence view
        - z_bio: normalized contrastive embedding for the bio view
        - h_tda: fused embedding for TDA regularization
        - h_seq: raw sequence view embedding [h_ref + h_alt]
        - h_bio: raw bio embedding
        - h_ref, h_alt: raw DNA embeddings from DNABERT
        """

        # Encode raw views
        h_ref = self.dna(seq_ref)      # [B, 768]
        h_alt = self.dna(seq_alt)      # [B, 768]
        h_bio = self.bio(bio_feats)    # [B, 256]

        # Sequence view embedding for contrastive learning
        h_seq = torch.cat([h_ref, h_alt], dim=1)   # [B, 768*2]

        # Fused embedding for TDA (joint representation)
        fused = torch.cat([h_ref, h_alt, h_bio], dim=1)  # [B, 768*2 + 256]
        h_tda = self.fusion(fused)                       # [B, shared_dim]

        # Projection heads (CLIP-style, no normalization on raw embeddings)
        z_seq = l2norm(self.seq_head(h_seq), dim=1)      # [B, proj_dim], norm=1
        z_bio = l2norm(self.bio_head(h_bio), dim=1)      # [B, proj_dim], norm=1

        return z_seq, z_bio, h_tda, h_seq, h_bio, h_ref, h_alt

    @torch.no_grad()
    def encode_fused(
        self,
        seq_ref: List[str],
        seq_alt: List[str],
        bio_feats: torch.Tensor
    ):
        """
        Fused embedding for rebuild/inference steps.
        Produces a fused embedding with no projection or normalization.
        """
        h_ref = self.dna(seq_ref)
        h_alt = self.dna(seq_alt)
        h_bio = self.bio(bio_feats)
        fused = torch.cat([h_ref, h_alt, h_bio], dim=1)
        return self.fusion(fused)      # [B, shared_dim]

    @torch.no_grad()
    def encode_sequence(
        self,
        seq_ref: List[str],
        seq_alt: List[str]
    ):

        h_ref = self.dna(seq_ref)
        h_alt = self.dna(seq_alt)
        h_seq = torch.cat([h_ref, h_alt], dim=1)
        return l2norm(self.seq_head(h_seq), dim=1)

    @torch.no_grad()
    def encode_bio(
        self,
        bio_feats: torch.Tensor
    ):
        
        h_bio = self.bio(bio_feats)
        return l2norm(self.bio_head(h_bio), dim=1)

class EMAWrapper(nn.Module):
    def __init__(self, model, decay: float = 0.99):
        super().__init__()
        self.decay = decay

        # unwrap DDP if needed
        base = model.module if isinstance(model, DDP) else model

        # construct a teacher of the SAME class as the base model
        self.teacher = type(base)(base.cfg)
        self.teacher.load_state_dict(base.state_dict(), strict=True)

        # EMA teacher is inference-only
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        base = model.module if isinstance(model, DDP) else model
        for t_param, s_param in zip(self.teacher.parameters(), base.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def to(self, *args, **kwargs):
        self.teacher.to(*args, **kwargs)
        return self


class CrossPredictor(nn.Module):
    """
    Predict the embedding of one view from the other:
      seq -> bio, bio -> seq
    We operate on h_seq and h_bio (NOT z_seq/z_bio) to avoid interfering with CLIP normalization.
    """
    def __init__(self, dim_seq: int, dim_bio: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.seq_to_bio = nn.Sequential(
            nn.Linear(dim_seq, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_bio),
        )
        self.bio_to_seq = nn.Sequential(
            nn.Linear(dim_bio, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_seq),
        )

    def forward(self, h_seq, h_bio):
        pred_bio = self.seq_to_bio(h_seq)
        pred_seq = self.bio_to_seq(h_bio)
        return pred_seq, pred_bio

class PureTDAHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1,F)
        return F.normalize(self.net(x), dim=1)  # (1,out_dim)

def cross_view_loss(h_seq, h_bio, cross_module: CrossPredictor):
    """
    L_cross = ||bio_hat - h_bio||^2 + ||seq_hat - h_seq||^2
    """
    pred_seq, pred_bio = cross_module(h_seq, h_bio)
    loss1 = F.mse_loss(pred_bio, h_bio)
    loss2 = F.mse_loss(pred_seq, h_seq)
    return loss1 + loss2


# def cross_view_loss(h_seq, h_bio, cross_module):
#     pred_seq, pred_bio = cross_module(h_seq, h_bio)

#     pred_seq = F.normalize(pred_seq, dim=1)
#     pred_bio = F.normalize(pred_bio, dim=1)

#     tgt_seq = F.normalize(h_seq.detach(), dim=1)
#     tgt_bio = F.normalize(h_bio.detach(), dim=1)

#     loss_seq = 2.0 - 2.0 * (pred_seq * tgt_seq).sum(dim=1).mean()
#     loss_bio = 2.0 - 2.0 * (pred_bio * tgt_bio).sum(dim=1).mean()
#     return loss_seq + loss_bio

