from __future__ import annotations

from libs import *


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.norm(x, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm



def collate_fn_refalt(batch):
    idxs = [b[0] for b in batch]
    seq_refs = [b[1] for b in batch]
    seq_alts = [b[2] for b in batch]
    bios = torch.stack([b[3] for b in batch], dim=0)     
    label = torch.stack([b[4] for b in batch], dim=0)   
    return idxs, seq_refs, seq_alts, bios, label



def clip_like_loss(z_seq: torch.Tensor, z_bio: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:

    a = F.normalize(z_seq, dim=1)
    b = F.normalize(z_bio, dim=1)
    logits = (a @ b.T) / float(temperature)
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def clip_soft_sirius_loss(
    z_seq: torch.Tensor,
    z_bio: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    alpha: float = 0.2,
    unlabeled_val: int = -1,
) -> torch.Tensor:

    z_seq = F.normalize(z_seq, dim=1)
    z_bio = F.normalize(z_bio, dim=1)
    logits = (z_seq @ z_bio.T) / float(temperature)  # (B,B)
    B = logits.size(0)

    target = torch.eye(B, device=logits.device, dtype=torch.float32)

    labels = labels.view(-1)
    valid = (labels != int(unlabeled_val))

    if valid.any():
        same = (labels[:, None] == labels[None, :]).float()
        same = same * (valid[:, None].float() * valid[None, :].float())
        same = same - torch.eye(B, device=logits.device)  # remove diagonal
        target = target + float(alpha) * same

    target = target / (target.sum(dim=1, keepdim=True) + 1e-12)

    logp = F.log_softmax(logits, dim=1)
    loss_i2t = -(target * logp).sum(dim=1).mean()

    logp_t = F.log_softmax(logits.T, dim=1)
    loss_t2i = -(target.T * logp_t).sum(dim=1).mean()

    return 0.5 * (loss_i2t + loss_t2i)





