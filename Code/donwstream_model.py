from __future__ import annotations

import os
import copy
import numpy as np
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================================================
# Dataset
# =========================================================
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape (N, D)
        y: shape (N,)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} vs {len(y)}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================================================
# Model
# =========================================================
class LogisticRegression(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # linear probe / multinomial logistic regression
        self.model = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.max_epochs * 0.6),
                int(self.max_epochs * 0.8),
            ],
            gamma=0.1,
        )

        return optimizer, scheduler

    def calculate_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, labels = batch
        logits = self(feats)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return loss, acc, preds


# =========================================================
# Utils
# =========================================================
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = EmbeddingDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# =========================================================
# Train / Eval loops
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for feats, labels in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(feats)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for feats, labels in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(feats)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs

        # accumulate for sklearn metrics
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # concat
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    # metrics
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_preds = []
    all_probs = []

    for feats, _ in loader:
        feats = feats.to(device, non_blocking=True)

        logits = model(feats)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    return y_pred, y_prob


# =========================================================
# Full training function
# =========================================================
# def fit_logistic_regression(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: Optional[np.ndarray] = None,
#     y_test: Optional[np.ndarray] = None,
#     batch_size: int = 128,
#     lr: float = 1e-3,
#     weight_decay: float = 1e-4,
#     max_epochs: int = 100,
#     num_workers: int = 0,
#     seed: int = 42,
#     out_ckpt: Optional[str] = None,
# ) -> Dict[str, Any]:
#     seed_everything(seed)
#     device = get_device()

#     if X_train.ndim != 2:
#         raise ValueError(f"X_train must be 2D, got {X_train.shape}")

#     feature_dim = X_train.shape[1]
#     num_classes = int(np.max(y_train)) + 1

#     train_loader = make_dataloader(
#         X_train, y_train,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )
#     val_loader = make_dataloader(
#         X_val, y_val,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     test_loader = None
#     if X_test is not None and y_test is not None:
#         test_loader = make_dataloader(
#             X_test, y_test,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         )

#     model = LogisticRegression(
#         feature_dim=feature_dim,
#         num_classes=num_classes,
#         lr=lr,
#         weight_decay=weight_decay,
#         max_epochs=max_epochs,
#     ).to(device)

#     optimizer, scheduler = model.configure_optimizers()

#     best_val_acc = -1.0
#     best_state_dict = copy.deepcopy(model.state_dict())
#     history = []

#     print(f"Device: {device}")
#     print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
#     if X_test is not None:
#         print(f"Test shape: {X_test.shape}")
#     print(f"Feature dim: {feature_dim}, Num classes: {num_classes}")

#     for epoch in range(1, max_epochs + 1):
#         train_metrics = train_one_epoch(model, train_loader, optimizer, device)
#         val_metrics = evaluate(model, val_loader, device)

#         scheduler.step()

#         row = {
#             "epoch": epoch,
#             "train_loss": train_metrics["loss"],
#             "train_acc": train_metrics["acc"],
#             "val_loss": val_metrics["loss"],
#             "val_acc": val_metrics["acc"],
#             "lr": optimizer.param_groups[0]["lr"],
#         }
#         history.append(row)

#         print(
#             f"Epoch {epoch:03d}/{max_epochs:03d} | "
#             f"train_loss={row['train_loss']:.4f} | "
#             f"train_acc={row['train_acc']:.4f} | "
#             f"val_loss={row['val_loss']:.4f} | "
#             f"val_acc={row['val_acc']:.4f} | "
#             f"lr={row['lr']:.2e}"
#         )

#         if val_metrics["acc"] > best_val_acc:
#             best_val_acc = val_metrics["acc"]
#             best_state_dict = copy.deepcopy(model.state_dict())

#     # restore best model
#     model.load_state_dict(best_state_dict)

#     if out_ckpt is not None:
#         os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
#         torch.save(
#             {
#                 "model_state_dict": model.state_dict(),
#                 "feature_dim": feature_dim,
#                 "num_classes": num_classes,
#                 "lr": lr,
#                 "weight_decay": weight_decay,
#                 "max_epochs": max_epochs,
#                 "best_val_acc": best_val_acc,
#             },
#             out_ckpt,
#         )
#         print(f"Best model saved to: {out_ckpt}")

#     results = {
#         "model": model,
#         "history": history,
#         "best_val_acc": best_val_acc,
#     }

#     # final validation metrics
#     final_val_metrics = evaluate(model, val_loader, device)
#     results["val_metrics"] = final_val_metrics

#     if test_loader is not None:
#         test_metrics = evaluate(model, test_loader, device)
#         y_test_pred, y_test_prob = predict(model, test_loader, device)

#         results["test_metrics"] = test_metrics
#         results["y_test_pred"] = y_test_pred
#         results["y_test_prob"] = y_test_prob

#         print(
#             f"[BEST MODEL] test_loss={test_metrics['loss']:.4f} | "
#             f"test_acc={test_metrics['acc']:.4f}"
#         )

#     return results


# import os
# import copy
# import numpy as np
# import torch
# from typing import Optional, Dict, Any


# def fit_logistic_regression(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: Optional[np.ndarray] = None,
#     y_test: Optional[np.ndarray] = None,
#     batch_size: int = 128,
#     lr: float = 1e-3,
#     weight_decay: float = 1e-4,
#     max_epochs: int = 100,
#     num_workers: int = 0,
#     seed: int = 42,
#     out_ckpt: Optional[str] = None,
#     selection_metric: str = "f1_macro",   # or "acc"
# ) -> Dict[str, Any]:
#     seed_everything(seed)
#     device = get_device()

#     if X_train.ndim != 2:
#         raise ValueError(f"X_train must be 2D, got {X_train.shape}")

#     feature_dim = X_train.shape[1]
#     num_classes = int(np.max(y_train)) + 1

#     train_loader = make_dataloader(
#         X_train, y_train,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )
#     val_loader = make_dataloader(
#         X_val, y_val,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     test_loader = None
#     if X_test is not None and y_test is not None:
#         test_loader = make_dataloader(
#             X_test, y_test,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         )

#     model = LogisticRegression(
#         feature_dim=feature_dim,
#         num_classes=num_classes,
#         lr=lr,
#         weight_decay=weight_decay,
#         max_epochs=max_epochs,
#     ).to(device)

#     optimizer, scheduler = model.configure_optimizers()

#     best_val_score = -float("inf")
#     best_val_metrics = None
#     best_epoch = -1
#     best_state_dict = copy.deepcopy(model.state_dict())
#     history = []

#     print(f"Device: {device}")
#     print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
#     if X_test is not None:
#         print(f"Test shape: {X_test.shape}")
#     print(f"Feature dim: {feature_dim}, Num classes: {num_classes}")
#     print(f"Selection metric: {selection_metric}")

#     for epoch in range(1, max_epochs + 1):
#         train_metrics = train_one_epoch(model, train_loader, optimizer, device)
#         val_metrics = evaluate(model, val_loader, device)

#         scheduler.step()

#         row = {
#             "epoch": epoch,
#             "train_loss": train_metrics["loss"],
#             "train_acc": train_metrics["acc"],
#             "val_loss": val_metrics["loss"],
#             "val_acc": val_metrics["acc"],
#             "val_precision_macro": val_metrics["precision_macro"],
#             "val_recall_macro": val_metrics["recall_macro"],
#             "val_f1_macro": val_metrics["f1_macro"],
#             "val_precision_weighted": val_metrics["precision_weighted"],
#             "val_recall_weighted": val_metrics["recall_weighted"],
#             "val_f1_weighted": val_metrics["f1_weighted"],
#             "lr": optimizer.param_groups[0]["lr"],
#         }
#         history.append(row)

#         print(
#             f"Epoch {epoch:03d}/{max_epochs:03d} | "
#             f"train_loss={row['train_loss']:.4f} | "
#             f"train_acc={row['train_acc']:.4f} | "
#             f"val_loss={row['val_loss']:.4f} | "
#             f"val_acc={row['val_acc']:.4f} | "
#             f"val_f1_macro={row['val_f1_macro']:.4f} | "
#             f"lr={row['lr']:.2e}"
#         )

#         if selection_metric not in val_metrics:
#             raise ValueError(
#                 f"selection_metric='{selection_metric}' not found in val_metrics. "
#                 f"Available keys: {list(val_metrics.keys())}"
#             )

#         current_score = val_metrics[selection_metric]

#         if current_score > best_val_score:
#             best_val_score = current_score
#             best_val_metrics = copy.deepcopy(val_metrics)
#             best_epoch = epoch
#             best_state_dict = copy.deepcopy(model.state_dict())

#     # restore best model
#     model.load_state_dict(best_state_dict)

#     if out_ckpt is not None:
#         ckpt_dir = os.path.dirname(out_ckpt)
#         if ckpt_dir:
#             os.makedirs(ckpt_dir, exist_ok=True)

#         torch.save(
#             {
#                 "model_state_dict": model.state_dict(),
#                 "feature_dim": feature_dim,
#                 "num_classes": num_classes,
#                 "lr": lr,
#                 "weight_decay": weight_decay,
#                 "max_epochs": max_epochs,
#                 "selection_metric": selection_metric,
#                 "best_val_score": best_val_score,
#                 "best_val_metrics": best_val_metrics,
#                 "best_epoch": best_epoch,
#             },
#             out_ckpt,
#         )
#         print(f"Best model saved to: {out_ckpt}")

#     results = {
#         "model": model,
#         "history": history,
#         "best_epoch": best_epoch,
#         "best_val_score": best_val_score,
#         "best_val_metrics": best_val_metrics,
#     }

#     # metrics after restoring best model
#     final_val_metrics = evaluate(model, val_loader, device)
#     results["val_metrics"] = final_val_metrics

#     if test_loader is not None:
#         test_metrics = evaluate(model, test_loader, device)
#         y_test_pred, y_test_prob = predict(model, test_loader, device)

#         results["test_metrics"] = test_metrics
#         results["y_test_pred"] = y_test_pred
#         results["y_test_prob"] = y_test_prob

#         print(
#             f"[BEST MODEL] "
#             f"best_epoch={best_epoch} | "
#             f"best_val_{selection_metric}={best_val_score:.4f} | "
#             f"test_loss={test_metrics['loss']:.4f} | "
#             f"test_acc={test_metrics['acc']:.4f} | "
#             f"test_f1_macro={test_metrics['f1_macro']:.4f}"
#         )

#     return results


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    num_workers: int = 0,
    seed: int = 42,
    out_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = get_device()

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got {X_train.shape}")

    feature_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    train_loader = make_dataloader(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = None
    if X_test is not None and y_test is not None:
        test_loader = make_dataloader(
            X_test, y_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    model = LogisticRegression(
        feature_dim=feature_dim,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    ).to(device)

    optimizer, scheduler = model.configure_optimizers()

    history = []

    print(f"Device: {device}")
    print(f"Train shape: {X_train.shape}")
    if X_test is not None:
        print(f"Test shape: {X_test.shape}")
    print(f"Feature dim: {feature_dim}, Num classes: {num_classes}")

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"lr={row['lr']:.2e}"
        )

    if out_ckpt is not None:
        ckpt_dir = os.path.dirname(out_ckpt)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "num_classes": num_classes,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
            },
            out_ckpt,
        )
        print(f"Model saved to: {out_ckpt}")

    results = {
        "model": model,
        "history": history,
    }

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device)
        y_test_pred, y_test_prob = predict(model, test_loader, device)

        results["test_metrics"] = test_metrics
        results["y_test_pred"] = y_test_pred
        results["y_test_prob"] = y_test_prob

        print(
            f"[FINAL MODEL] "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f} | "
            f"test_f1_macro={test_metrics['f1_macro']:.4f}"
        )

    return results