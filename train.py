import argparse
import json
from pathlib import Path

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.config import Config
from src.dataset import PlantDataset
from src.model import OrgansClassifier
from src.trainer import Trainer
from src.utils import seed_everything, resize_pwd


def build_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_pwd(img, cfg.data.padding_color, cfg.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])
    val_test_transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_pwd(img, cfg.data.padding_color, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])
    return train_transform, val_test_transform


def build_dataloaders(cfg, val_ratio: float):
    train_transform, val_test_transform = build_transforms(cfg)

    full_train = PlantDataset(cfg.data.root_dir, transform=train_transform)
    full_val = PlantDataset(cfg.data.root_dir, transform=val_test_transform)

    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(cfg.train.seed)
    indices = torch.randperm(n, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader, full_train.num_classes, full_train.class_to_idx


def build_optimizer(model, cfg):
    trainable = [p for p in model.parameters() if p.requires_grad]
    name = cfg.optim.optimizer.lower()

    if name == "adamw":
        return AdamW(trainable, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if name == "adam":
        return Adam(trainable, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if name == "sgd":
        return SGD(
            trainable,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optim.optimizer}")


def build_scheduler(optimizer, cfg):
    name = cfg.optim.scheduler.lower()
    warmup = max(0, cfg.optim.warmup_epochs)

    if name == "none":
        return LambdaLR(optimizer, lambda _: 1.0)

    if name in {"cosine", "cosineannealinglr"}:
        main = CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs - warmup))
    elif name in {"step", "steplr"}:
        main = StepLR(optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler}")

    if warmup == 0:
        return main

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, main],
        milestones=[warmup],
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train plant organ classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    data = p.add_argument_group("data")
    data.add_argument("--data-dir", type=str, default="data/plant")
    data.add_argument("--image-size", type=int, default=336)
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--val-ratio", type=float, default=0.2)

    model = p.add_argument_group("model")
    model.add_argument("--backbone", type=str, default="resnet50",
                       help="timm backbone name (e.g. resnet50, convnext_base)")
    model.add_argument("--no-pretrained", action="store_true")

    optim = p.add_argument_group("optim")
    optim.add_argument("--optimizer", type=str, default="AdamW",
                       choices=["AdamW", "Adam", "SGD"])
    optim.add_argument("--lr", type=float, default=5e-5)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--momentum", type=float, default=0.9)
    optim.add_argument("--scheduler", type=str, default="CosineAnnealingLR",
                       choices=["CosineAnnealingLR", "StepLR", "none"])
    optim.add_argument("--warmup-epochs", type=int, default=1)
    optim.add_argument("--step-size", type=int, default=10)
    optim.add_argument("--gamma", type=float, default=0.1)

    train = p.add_argument_group("train")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--patience", type=int, default=None,
                       help="Early stopping patience (epochs), None = disabled")
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--log-every", type=int, default=100)
    train.add_argument("--tsne-every", type=int, default=0)

    paths = p.add_argument_group("paths")
    paths.add_argument("--experiment-name", type=str, default="plant_retrieval")
    paths.add_argument("--output-dir", type=str, default="logs")
    paths.add_argument("--checkpoint-dir", type=str, default="logs/checkpoints")

    return p.parse_args()


def build_cfg(args) -> Config:
    cfg = Config()

    cfg.data.root_dir = args.data_dir
    cfg.data.image_size = args.image_size
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers

    cfg.model.backbone = args.backbone
    cfg.model.pretrained = not args.no_pretrained

    cfg.optim.optimizer = args.optimizer
    cfg.optim.lr = args.lr
    cfg.optim.weight_decay = args.weight_decay
    cfg.optim.momentum = args.momentum
    cfg.optim.scheduler = args.scheduler
    cfg.optim.warmup_epochs = args.warmup_epochs
    cfg.optim.step_size = args.step_size
    cfg.optim.gamma = args.gamma

    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    cfg.train.early_stopping_patience = args.patience
    cfg.train.save_every = args.save_every
    cfg.train.log_every = args.log_every
    cfg.train.tsne_every = args.tsne_every

    cfg.paths.experiment_name = args.experiment_name
    cfg.paths.output_dir = args.output_dir
    cfg.paths.checkpoint_dir = args.checkpoint_dir

    return cfg


def main():
    args = parse_args()
    cfg = build_cfg(args)

    seed_everything(cfg.train.seed)

    train_loader, val_loader, num_classes, class_to_idx = build_dataloaders(
        cfg, val_ratio=args.val_ratio
    )
    cfg.model.num_classes = num_classes

    out_dir = Path(cfg.paths.output_dir) / cfg.paths.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")
    with (out_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    model = OrgansClassifier(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(model, optimizer, scheduler, cfg)

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        trainer.logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()