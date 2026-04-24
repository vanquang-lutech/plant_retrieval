from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE


def plot_metrics(csv_path, out_path):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]

    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], label="train", color="tab:blue")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="val", color="tab:orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]

    if "train_acc" in df.columns:
        ax.plot(df["epoch"], df["train_acc"] * 100, label="train", color="tab:blue")
    if "val_acc" in df.columns:
        ax.plot(df["epoch"], df["val_acc"] * 100, label="val", color="tab:orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return out_path

@torch.no_grad()
def _extract_features(model, loader, device, max_samples: int = 2000):
    was_training = model.training
    model.eval()
    feats, labels = [], []
    total = 0

    for images, lbls in loader:
        images = images.to(device, non_blocking=True)
        f = model.base_model(images)
        feats.append(f.cpu())
        labels.append(lbls)
        total += lbls.size(0)
        if total >= max_samples:
            break

    if was_training:
        model.train()

    feats = torch.cat(feats, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]
    return feats.numpy(), labels.numpy()

def plot_tsne(model, loader, device, out_path, epoch: int, max_samples: int = 2000, idx_to_class: Optional[dict] = None,):

    feats, labels = _extract_features(model, loader, device, max_samples=max_samples)
    if len(feats) < 10:
        return None

    perplexity = min(30, max(5, len(feats) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    emb = tsne.fit_transform(feats)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = np.unique(labels)
    cmap = plt.get_cmap("tab20" if len(unique) <= 20 else "gist_ncar")

    for i, c in enumerate(unique):
        mask = labels == c
        name = idx_to_class[int(c)] if idx_to_class else str(c)
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            s=12, alpha=0.7,
            color=cmap(i / max(len(unique), 1)),
            label=name,
        )

    ax.set_title(f"t-SNE (epoch {epoch})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path