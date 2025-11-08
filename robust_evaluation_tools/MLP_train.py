import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from typing import Tuple

try:
    from torch.utils.tensorboard import SummaryWriter  # optional
except Exception:  # pragma: no cover
    SummaryWriter = None

from robust_evaluation_tools.robust_MLP import MODEL_DIR


class PatientDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train = DataLoader(PatientDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val = DataLoader(PatientDataset(X_val, y_val), batch_size=batch_size)
    test = DataLoader(PatientDataset(X_test, y_test), batch_size=batch_size)
    return train, val, test


def train_epoch(model: torch.nn.Module, loader: DataLoader, crit: nn.Module, opt: torch.optim.Optimizer,
                device: str = "cpu", neg_weight: float = 10.0) -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).float()
        opt.zero_grad()
        logits = model(xb)
        base_loss = crit(logits, yb)
        weights = torch.where(yb == 0, float(neg_weight), 1.0)
        loss = (base_loss * weights).mean()
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: torch.nn.Module, loader: DataLoader, crit: nn.Module,
               device: str = "cpu", neg_weight: float = 10.0):
    model.eval()
    losses, probs, labels = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.float()
        logits = model(xb)
        base_loss = crit(logits, yb.to(device))
        weights = torch.where(yb == 0, float(neg_weight), 1.0)
        loss = (base_loss * weights).mean().item()
        losses.append(loss * xb.size(0))
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(yb)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    f1 = f1_score(labels, (probs > 0.5).astype(int)) if len(np.unique(labels)) > 1 else float('nan')
    return np.sum(losses) / len(loader.dataset), auc, f1, probs, labels


def fit(model: torch.nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        wd: float = 1e-4,
        patience: int = 10,
        device: str = "cpu",
        neg_weight: float = 10.0,
        run_name: str = None):
    """Entraîne le modèle avec sélection du meilleur état par AUC validation.

    Retourne: (best_state, train_losses, val_losses, best_auc)
    """
    crit = nn.BCEWithLogitsLoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    writer = None
    if run_name and SummaryWriter is not None:
        writer = SummaryWriter(f"{MODEL_DIR}/runs/{run_name}")

    best_auc, best_state, counter = -1.0, None, 0
    tr_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_dl, crit, opt, device=device, neg_weight=neg_weight)
        val_loss, val_auc, _f1, _p, _l = eval_epoch(model, val_dl, crit, device=device, neg_weight=neg_weight)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        sched.step(val_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", tr_loss, ep)
            writer.add_scalar("Loss/val", val_loss, ep)
            if not np.isnan(val_auc):
                writer.add_scalar("AUC/val", val_auc, ep)

        if not np.isnan(val_auc) and val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, tr_losses, val_losses, best_auc

