from libs import *
from config import *
from utils import *
from loss import *
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig


def load_hf_backbone(
    model_name: str,
    cache_dir: str = "./tmp",
    trust_remote_code: bool = False,
    config=None,
):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if rank == 0:
        backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            config=config,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        dist.barrier()
        backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            config=config,
            local_files_only=True,
        )
    return backbone


def load_dnabert_backbone(model_name: str, cache_dir: str, conf):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    print("x" * 100)
    print("Rank ", rank)

    if rank == 0:
        backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            config=conf,
            attn_implementation="eager",
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        dist.barrier()
        backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            config=conf,
            attn_implementation="eager",
            local_files_only=True,
        )

    return backbone


class BaseHFEncoder(nn.Module):
    """
    Generic HF encoder with mean/cls pooling.
    Can be used for text models.
    """
    def __init__(
        self,
        model_name: str,
        finetune: bool,
        pool: str = "mean",
        max_len: int = 256,
        trust_remote_code: bool = False,
        cache_dir: str = "./tmp",
    ):
        super().__init__()
        self.model_name = model_name
        self.pool = pool
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        self.backbone = load_hf_backbone(
            model_name=model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            config=None,
        )

        hidden_size = getattr(self.backbone.config, "hidden_size", 768)
        self.out_dim = hidden_size

        if not finetune:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**inputs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

        if self.pool == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = hidden * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            return hidden.sum(dim=1) / denom
        elif self.pool == "cls":
            return hidden[:, 0, :]
        else:
            raise ValueError("pool must be 'mean' or 'cls'")


# class DNABertEncoder(nn.Module):
#     def __init__(self, model_name: str, finetune: bool, pool: str = "mean", max_len: int = 512):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             cache_dir="./tmp",
#             trust_remote_code=True,
#         )
#         conf = BertConfig.from_pretrained(model_name, cache_dir="./tmp")
#         self.backbone = load_dnabert_backbone(model_name, "./tmp", conf)

#         self.pool = pool
#         self.max_len = max_len
#         self.out_dim = getattr(self.backbone.config, "hidden_size", 768)

#         if not finetune:
#             self.backbone.eval()
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         else:
#             for name, param in self.backbone.named_parameters():
#                 if "encoder.layer.11" in name or "pooler" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False

#     def forward(self, seqs: List[str]) -> torch.Tensor:
#         device = next(self.backbone.parameters()).device
#         inputs = self.tokenizer(
#             seqs,
#             padding=True,
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         ).to(device)

#         outputs = self.backbone(**inputs)
#         hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

#         if self.pool == "mean":
#             mask = inputs["attention_mask"].unsqueeze(-1).float()
#             hidden = hidden * mask
#             denom = mask.sum(dim=1).clamp_min(1.0)
#             return hidden.sum(dim=1) / denom
#         elif self.pool == "cls":
#             return hidden[:, 0, :]
#         raise ValueError("pool must be 'mean' or 'cls'")


class DNABertEncoder(nn.Module):
    def __init__(self, model_name: str, finetune: bool, pool: str = "mean", max_len: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./tmp",
            trust_remote_code=True,
        )
        conf = BertConfig.from_pretrained(model_name, cache_dir="./tmp")
        self.backbone = load_dnabert_backbone(model_name, "./tmp", conf)

        self.pool = pool
        self.max_len = max_len
        self.out_dim = getattr(self.backbone.config, "hidden_size", 768)

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Freeze everything first
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze only last encoder block
            for name, param in self.backbone.named_parameters():
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True

            print("\n[DNABERT] Trainable parameters:")
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    print(" ", name)

    def forward(self, seqs: List[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        inputs = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**inputs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

        if self.pool == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = hidden * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            return hidden.sum(dim=1) / denom
        elif self.pool == "cls":
            return hidden[:, 0, :]
        raise ValueError("pool must be 'mean' or 'cls'")


class MutationEncoder(nn.Module):
    """
    Learn a compact mutation representation from REF / ALT embeddings.

    Input features:
      [h_ref, h_alt, h_alt - h_ref, |h_alt - h_ref|, h_alt * h_ref]
    """
    def __init__(
        self,
        dna_dim: int = 768,
        out_dim: int = 768,
        hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = 5 * dna_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h_ref: torch.Tensor, h_alt: torch.Tensor) -> torch.Tensor:
        delta = h_alt - h_ref
        abs_delta = torch.abs(delta)
        interaction = h_alt * h_ref
        x = torch.cat([h_ref, h_alt, delta, abs_delta, interaction], dim=1)
        return self.net(x)


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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.use_text = getattr(cfg, "use_text", False)
        self.seq_offline_embeddings = getattr(cfg, "seq_offline_embeddings", False)
        self.text_offline_embeddings = getattr(cfg, "text_offline_embeddings", False)

        self.bio_dim_out = 256

        # ---------------------------------------------------------
        # Sequence branch
        # ---------------------------------------------------------
        if self.seq_offline_embeddings:
            self.dna_dim = getattr(cfg, "seq_emb_dim", 768)
            self.dna = None
        else:
            self.dna = DNABertEncoder(
                cfg.dna_model_name,
                cfg.dna_finetune,
                cfg.pool,
                cfg.max_len,
            )
            self.dna_dim = self.dna.out_dim

        self.mutation_dim = getattr(cfg, "mutation_dim", self.dna_dim)

        self.mutation_encoder = MutationEncoder(
            dna_dim=self.dna_dim,
            out_dim=self.mutation_dim,
            hidden=getattr(cfg, "mutation_hidden", 1024),
            dropout=getattr(cfg, "dropout", 0.1),
        )

        # ---------------------------------------------------------
        # Bio branch
        # ---------------------------------------------------------
        self.bio = BioEncoder(
            cfg.bio_dim,
            cfg.bio_hidden,
            self.bio_dim_out,
        )

        # ---------------------------------------------------------
        # Text branch
        # ---------------------------------------------------------
        self.text_dim = None
        if self.use_text:
            if self.text_offline_embeddings:
                self.text_dim = getattr(cfg, "text_emb_dim", 768)
                self.text_encoder = None
            else:
                self.text_encoder = BaseHFEncoder(
                    model_name=cfg.text_model_name,
                    finetune=cfg.text_finetune,
                    pool=getattr(cfg, "text_pool", "mean"),
                    max_len=getattr(cfg, "text_max_len", 256),
                    trust_remote_code=getattr(cfg, "text_trust_remote_code", False),
                    cache_dir="./tmp",
                )
                self.text_dim = self.text_encoder.out_dim

            self.text_head = InstanceProjector(
                in_dim=self.text_dim,
                out_dim=cfg.proj_dim,
            )

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------
        self.fused_dim = self.mutation_dim + self.bio_dim_out
        if self.use_text:
            self.fused_dim += self.text_dim

        self.fusion = FusionHead(self.fused_dim, cfg.shared_dim)

        self.seq_head = InstanceProjector(
            in_dim=self.mutation_dim,
            out_dim=cfg.proj_dim,
        )

        self.bio_head = InstanceProjector(
            in_dim=self.bio_dim_out,
            out_dim=cfg.proj_dim,
        )

    def encode_refalt(self, seq_ref, seq_alt, ref_emb=None, alt_emb=None):
        if self.seq_offline_embeddings:
            if ref_emb is None or alt_emb is None:
                raise ValueError("Offline sequence mode requires ref_emb and alt_emb.")
            h_ref = ref_emb
            h_alt = alt_emb
        else:
            h_ref = self.dna(seq_ref)
            h_alt = self.dna(seq_alt)

        h_mut = self.mutation_encoder(h_ref, h_alt)
        return h_mut, h_ref, h_alt

    def encode_text(self, texts=None, text_emb=None):
        if not self.use_text:
            return None

        if self.text_offline_embeddings:
            if text_emb is None:
                raise ValueError("Offline text mode requires text_emb.")
            h_text = text_emb
        else:
            if texts is None:
                raise ValueError("Online text mode requires texts.")
            h_text = self.text_encoder(texts)

        return h_text

    def build_fused_representation(
        self,
        seq_ref=None,
        seq_alt=None,
        bio_feats=None,
        texts=None,
        ref_emb=None,
        alt_emb=None,
        text_emb=None,
    ):
        h_mut, h_ref, h_alt = self.encode_refalt(
            seq_ref=seq_ref,
            seq_alt=seq_alt,
            ref_emb=ref_emb,
            alt_emb=alt_emb,
        )

        h_bio = self.bio(bio_feats)

        fusion_parts = [h_mut, h_bio]

        h_text = None
        if self.use_text:
            h_text = self.encode_text(texts=texts, text_emb=text_emb)
            fusion_parts.append(h_text)

        fused_input = torch.cat(fusion_parts, dim=1)
        h_fused = self.fusion(fused_input)

        return h_fused, h_mut, h_bio, h_text, h_ref, h_alt

    def forward(
        self,
        seq_ref=None,
        seq_alt=None,
        bio_feats=None,
        texts=None,
        ref_emb=None,
        alt_emb=None,
        text_emb=None,
    ):
        h_tda, h_mut, h_bio, h_text, h_ref, h_alt = self.build_fused_representation(
            seq_ref=seq_ref,
            seq_alt=seq_alt,
            bio_feats=bio_feats,
            texts=texts,
            ref_emb=ref_emb,
            alt_emb=alt_emb,
            text_emb=text_emb,
        )

        z_seq = l2norm(self.seq_head(h_mut), dim=1)
        z_bio = l2norm(self.bio_head(h_bio), dim=1)
        z_text = None

        if self.use_text:
            z_text = l2norm(self.text_head(h_text), dim=1)

        return z_seq, z_bio, z_text, h_tda, h_mut, h_bio, h_text, h_ref, h_alt

    @torch.no_grad()
    def encode_fused(
        self,
        seq_ref=None,
        seq_alt=None,
        bio_feats=None,
        texts=None,
        ref_emb=None,
        alt_emb=None,
        text_emb=None,
    ):
        h_tda, _, _, _, _, _ = self.build_fused_representation(
            seq_ref=seq_ref,
            seq_alt=seq_alt,
            bio_feats=bio_feats,
            texts=texts,
            ref_emb=ref_emb,
            alt_emb=alt_emb,
            text_emb=text_emb,
        )
        return h_tda

    @torch.no_grad()
    def encode_sequence(
        self,
        seq_ref=None,
        seq_alt=None,
        ref_emb=None,
        alt_emb=None,
    ):
        h_mut, _, _ = self.encode_refalt(
            seq_ref=seq_ref,
            seq_alt=seq_alt,
            ref_emb=ref_emb,
            alt_emb=alt_emb,
        )
        return l2norm(self.seq_head(h_mut), dim=1)

    @torch.no_grad()
    def encode_bio(self, bio_feats: torch.Tensor) -> torch.Tensor:
        h_bio = self.bio(bio_feats)
        return l2norm(self.bio_head(h_bio), dim=1)

    @torch.no_grad()
    def encode_text_view(self, texts=None, text_emb=None):
        if not self.use_text:
            raise ValueError("Text branch is disabled.")
        h_text = self.encode_text(texts=texts, text_emb=text_emb)
        return l2norm(self.text_head(h_text), dim=1)


class EMAWrapper(nn.Module):
    def __init__(self, model, decay: float = 0.99):
        super().__init__()
        self.decay = decay

        base = model.module if isinstance(model, DDP) else model
        self.teacher = type(base)(base.cfg)
        self.teacher.load_state_dict(base.state_dict(), strict=True)

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

@torch.no_grad()
def precompute_offline_refalt_embeddings(
    seq_refs: List[str],
    seq_alts: List[str],
    model_name: str,
    out_ref_path: str,
    out_alt_path: str,
    batch_size: int = 32,
    pool: str = "mean",
    max_len: int = 512,
    device: str = "cuda",
):
    encoder = DNABertEncoder(
        model_name=model_name,
        finetune=False,
        pool=pool,
        max_len=max_len,
    )

    ref_emb = encode_in_batches_texts(
        encoder=encoder,
        texts=seq_refs,
        batch_size=batch_size,
        device=device,
    )
    alt_emb = encode_in_batches_texts(
        encoder=encoder,
        texts=seq_alts,
        batch_size=batch_size,
        device=device,
    )

    torch.save(ref_emb, out_ref_path)
    torch.save(alt_emb, out_alt_path)

    return ref_emb, alt_emb


@torch.no_grad()
def precompute_offline_text_embeddings(
    texts: List[str],
    model_name: str,
    out_text_path: str,
    batch_size: int = 32,
    pool: str = "mean",
    max_len: int = 256,
    trust_remote_code: bool = False,
    device: str = "cuda",
):
    encoder = BaseHFEncoder(
        model_name=model_name,
        finetune=False,
        pool=pool,
        max_len=max_len,
        trust_remote_code=trust_remote_code,
    )

    text_emb = encode_in_batches_texts(
        encoder=encoder,
        texts=texts,
        batch_size=batch_size,
        device=device,
    )

    torch.save(text_emb, out_text_path)
    return text_emb


@torch.no_grad()
def precompute_all_modalities(
    seq_refs: List[str],
    seq_alts: List[str],
    texts: Optional[List[str]],
    dna_model_name: str,
    text_model_name: Optional[str],
    out_ref_path: str,
    out_alt_path: str,
    out_text_path: Optional[str] = None,
    batch_size: int = 32,
    seq_pool: str = "mean",
    text_pool: str = "mean",
    seq_max_len: int = 512,
    text_max_len: int = 256,
    device: str = "cuda",
):
    ref_emb, alt_emb = precompute_offline_refalt_embeddings(
        seq_refs=seq_refs,
        seq_alts=seq_alts,
        model_name=dna_model_name,
        out_ref_path=out_ref_path,
        out_alt_path=out_alt_path,
        batch_size=batch_size,
        pool=seq_pool,
        max_len=seq_max_len,
        device=device,
    )

    text_emb = None
    if texts is not None and text_model_name is not None and out_text_path is not None:
        text_emb = precompute_offline_text_embeddings(
            texts=texts,
            model_name=text_model_name,
            out_text_path=out_text_path,
            batch_size=batch_size,
            pool=text_pool,
            max_len=text_max_len,
            device=device,
        )

    return ref_emb, alt_emb, text_emb