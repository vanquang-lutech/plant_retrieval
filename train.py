import argparse
import json
from functools import partial
from pathlib import Path
from timm.optim import create_optimizer_v2
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import Config
from src.dataset import PlantDataset
from src.model import OrgansClassifier
from src.trainer import Trainer
from src.utils import resize_pwd, seed_everything, UnshaprMask


def build_transforms(cfg):
    pad_resize = partial(
        resize_pwd,
        padding_color=cfg.data.padding_color,
        target_size=cfg.data.image_size,
    )
    size = cfg.data.image_size
    sharpen = UnshaprMask(radius=1.0, amount=1.0)

    train_transform = transforms.Compose([
        pad_resize,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.2),
        sharpen,
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.RandomErasing(
                p=1.0,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0,
            )
        ], p=0.4),

        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])

    val_test_transform = transforms.Compose([
        pad_resize,
        sharpen,
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])

    return train_transform, val_test_transform


def build_dataloaders(cfg):
    train_transform, val_test_transform = build_transforms(cfg)

    root = Path(cfg.data.root_dir)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"
    for d in (train_dir, val_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing split folder: {d}")

    train_ds = PlantDataset(str(train_dir), transform=train_transform)
    val_ds = PlantDataset(
        str(val_dir), transform=val_test_transform, class_to_idx=train_ds.class_to_idx,
    )
    test_ds = PlantDataset(
        str(test_dir), transform=val_test_transform, class_to_idx=train_ds.class_to_idx,
    )

    def make_loader(ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and cfg.train.device.startswith("cuda"),
            drop_last=drop_last,
        )

    return (
        make_loader(train_ds, shuffle=True, drop_last=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
        train_ds.num_classes,
        train_ds.class_to_idx,
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


def build_optimizer(model, cfg):
    return create_optimizer_v2(
        model,
        opt=cfg.optim.optimizer.lower(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
        layer_decay=0.75,  
        filter_bias_and_bn=True,  
    )


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
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup_sched, main], milestones=[warmup],
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train plant organ classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume", type=str, default=None)

    data = p.add_argument_group("data")
    data.add_argument("--data-dir", type=str, default="data/plant",
                      help="Root folder containing train/, val/, test/ subfolders")
    data.add_argument("--image-size", type=int, default=336)
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--num-workers", type=int, default=4)

    model = p.add_argument_group("model")
    model.add_argument("--backbone", type=str, default="resnet50")
    model.add_argument("--no-pretrained", action="store_true")

    optim = p.add_argument_group("optim")
    optim.add_argument("--optimizer", type=str, default="AdamW",
                       choices=["adamw", "adam", "sgd"])
    optim.add_argument("--lr", type=float, default=1e-4)
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
    train.add_argument("--patience", type=int, default=None)
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

    train_loader, val_loader, test_loader, num_classes, class_to_idx, sizes = \
        build_dataloaders(cfg)
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
    trainer.logger.info(
        f"Dataset — train={sizes['train']} val={sizes['val']} test={sizes['test']} "
        f"num_classes={num_classes}"
    )

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        trainer.logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    trainer.fit(train_loader, val_loader)

    best_ckpt = trainer.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
        trainer.logger.info(f"[TEST] Loaded best checkpoint from {best_ckpt}")
    else:
        trainer.logger.info("[TEST] best.pt not found, using final weights")

    test_m = trainer.validate(test_loader)
    trainer.logger.info(
        f"[TEST] loss={test_m['loss']:.4f} acc={test_m['acc']*100:.2f}%"
    )

    with (out_dir / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump({"test_loss": test_m["loss"], "test_acc": test_m["acc"]}, f, indent=2)


if __name__ == "__main__":
    main()