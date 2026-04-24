import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.visualize import plot_metrics, plot_tsne
from src.utils import AverageMeter, MetricLogger, setup_logging


class Trainer:
    def __init__(self, model, optimizer, scheduler, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()

        self.use_amp = cfg.train.device.startswith("cuda")
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_metric = 0.0
        self.early_stopping_count = 0

        self.checkpoint_dir = Path(cfg.paths.checkpoint_dir) / cfg.paths.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path(cfg.paths.output_dir) / cfg.paths.experiment_name
        self.logger = setup_logging(cfg.paths.experiment_name, log_dir / "train.log")
        self.metric_logger = MetricLogger(log_dir)

        self.plots_dir = log_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tsne_every = getattr(cfg.train, "tsne_every", 0)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        start = time.time()

        pbar = tqdm(
            loader,
            desc=f"[train] ep {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            acc = (logits.argmax(1) == labels).float().mean().item()
            loss_meter.update(loss.item(), n=batch_size)
            acc_meter.update(acc, n=batch_size)

            # Update progress bar every 10 batches to avoid excessive redrawing
            if (step + 1) % 100 == 0:
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    acc=f"{acc_meter.avg*100:.2f}%",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

            if (step + 1) % self.cfg.train.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"[train][ep {epoch:03d}][{step+1:>5}/{len(loader)}] "
                    f"loss={loss_meter.avg:.4f} acc={acc_meter.avg*100:.2f}% lr={lr:.2e}"
                )

        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
            "time": time.time() - start,
        }

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(
            loader,
            desc="[val]",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(
                loss=f"{total_loss/total_samples:.4f}",
                acc=f"{total_correct/total_samples*100:.2f}%",
            )

        return {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
        }

    def fit(self, train_loader, val_loader):
        self.logger.info(
            f"Start training — {self.cfg.train.epochs} epochs on {self.device}"
        )

        epoch_bar = tqdm(
            range(1, self.cfg.train.epochs + 1),
            desc="[epochs]",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            train_m = self.train_one_epoch(train_loader, epoch)
            val_m = self.validate(val_loader)

            epoch_bar.set_postfix(
                train_acc=f"{train_m['acc']*100:.2f}%",
                val_acc=f"{val_m['acc']*100:.2f}%",
                best=f"{self.best_metric*100:.2f}%",
            )

            self.logger.info(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']*100:.2f}% | "
                f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']*100:.2f}% | "
                f"time={train_m['time']:.1f}s"
            )

            self.metric_logger.log({
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_acc": train_m["acc"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            plot_metrics(
                csv_path=self.metric_logger.csv_file,
                out_path=self.plots_dir / "metrics.png",
            )
            if self.tsne_every and epoch % self.tsne_every == 0:
                ds = val_loader.dataset
                idx_to_class = getattr(ds, "idx_to_class", None) \
                    or getattr(getattr(ds, "dataset", None), "idx_to_class", None)
                plot_tsne(
                    model=self.model,
                    loader=val_loader,
                    device=self.device,
                    out_path=self.plots_dir / f"tsne_epoch_{epoch:03d}.png",
                    epoch=epoch,
                    idx_to_class=idx_to_class,
                )

            self.scheduler.step()

            if val_m["acc"] > self.best_metric:
                self.best_metric = val_m["acc"]
                self.early_stopping_count = 0
                self.save_checkpoint("best.pt", epoch, val_m)
            else:
                self.early_stopping_count += 1

            if epoch % self.cfg.train.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, val_m)

            patience = self.cfg.train.early_stopping_patience
            if patience is not None and self.early_stopping_count >= patience:
                self.logger.info(
                    f"[early stop] no improve in {patience} epochs. "
                    f"Best acc={self.best_metric*100:.2f}%"
                )
                break

        self.save_checkpoint("last.pt", epoch, val_m)
        self.logger.info(f"Finished. Best val_acc={self.best_metric*100:.2f}%")
        return self.best_metric

    def save_checkpoint(self, name, epoch, metrics):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "metrics": metrics,
                "best_metric": self.best_metric,
            },
            path,
        )
        self.logger.info(f"[ckpt] saved at {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_metric = checkpoint.get("best_metric", 0.0)
        return checkpoint["epoch"]