# Plant Organ Classification

A minimal but complete PyTorch pipeline for classifying plant organs (leaf, flower, fruit, stem, ...) using transfer learning with `timm` backbones. The project favors a CLI-first workflow with clean dataclass configs, AMP training, checkpointing, early stopping, metric plots, and optional t-SNE visualization.

## Project structure

```text
plant_retrieval/
├── src/
│   ├── __init__.py
│   ├── config.py          # Dataclass config (data / model / optim / train / paths)
│   ├── dataset.py         # PlantDataset - reads <class>/<image> folders
│   ├── model.py           # OrgansClassifier - timm backbone + classifier head
│   ├── trainer.py         # Trainer - train/val loop, AMP, checkpoint, logging
│   ├── utils.py           # resize_pwd, setup_logging, AverageMeter, MetricLogger, seed_everything
│   ├── visualize.py       # plot_metrics, plot_tsne
│   └── losses.py          # Placeholder (CrossEntropyLoss is used inline in Trainer)
├── notebook/              # Experimental notebooks
├── train.py               # Entry point - CLI training
├── main.py                # Placeholder
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python >= 3.11
- CUDA GPU recommended (AMP is enabled automatically when CUDA is available)
- Dependency management via [`uv`](https://github.com/astral-sh/uv)

### Installation

```bash
uv sync

uv pip install torch torchvision timm tqdm \
               numpy pandas matplotlib scikit-learn Pillow
```

Alternatively, install the same packages with plain `pip` in any Python 3.11+ environment.

It is recommended to extend `pyproject.toml` with the full dependency list so that a single `uv sync` is enough:

```toml
dependencies = [
    "torch>=2.2",
    "torchvision",
    "timm",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "Pillow",
    "tqdm",
]
```

## Dataset layout

The dataset must be pre-split into three folders (`train`, `val`, `test`). Each folder contains one sub-folder per class:

```text
data/plant/
├── train/
│   ├── class_1/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── class_2/
│   └── .../
├── val/
│   ├── class_1/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── class_2/
│   └── .../
└── test/
│   ├── class_1/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── class_2/
│   └── .../
```

Requirements:

- All classes found in `val/` and `test/` must also exist in `train/`. `PlantDataset` raises `ValueError` otherwise.
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
- Corrupted images are skipped and logged (`[skip] <path>: <error>`); the dataloader is not interrupted.

## Training

### Smoke test (quickly verify the pipeline)

```bash
python train.py --epochs 1 --batch-size 8 --num-workers 0 \
                --save-every 1 --experiment-name smoke_test
```

### Full training run

```bash
python train.py \
    --data-dir data/plant \
    --backbone resnet50 \
    --image-size 336 \
    --batch-size 64 \
    --epochs 50 \
    --lr 5e-5 \
    --optimizer AdamW \
    --scheduler CosineAnnealingLR \
    --warmup-epochs 1 \
    --experiment-name resnet50_baseline
```

### Enable t-SNE visualization every 10 epochs

```bash
python train.py --epochs 50 --tsne-every 10 --experiment-name tsne_demo
```

### Resume from a checkpoint

```bash
python train.py --resume logs/checkpoints/resnet50_baseline/best.pt \
                --experiment-name resnet50_baseline
```

### Inspect all available flags

```bash
python train.py --help
```

## CLI arguments

| Group | Flag | Default | Description |
|-------|------|---------|-------------|
| data  | `--data-dir` | `data/plant` | Root folder that contains `train/`, `val/`, `test/` |
|       | `--image-size` | `336` | Square pad + resize |
|       | `--batch-size` | `64` | Batch size for train/val/test loaders |
|       | `--num-workers` | `4` | DataLoader worker count |
| model | `--backbone` | `resnet50` | Any backbone supported by `timm` |
|       | `--no-pretrained` | off | Disable pretrained weights |
| optim | `--optimizer` | `AdamW` | `AdamW` / `Adam` / `SGD` |
|       | `--lr` | `5e-5` | Base learning rate |
|       | `--weight-decay` | `1e-4` | Weight decay |
|       | `--momentum` | `0.9` | Used by SGD |
|       | `--scheduler` | `CosineAnnealingLR` | `CosineAnnealingLR` / `StepLR` / `none` |
|       | `--warmup-epochs` | `1` | Linear warmup epochs |
|       | `--step-size` | `10` | StepLR step size |
|       | `--gamma` | `0.1` | StepLR decay factor |
| train | `--epochs` | `100` | Total training epochs |
|       | `--seed` | `42` | Random seed |
|       | `--device` | auto | `cuda` if available, else `cpu` |
|       | `--patience` | `None` | Early stopping patience (epochs). `None` disables it |
|       | `--save-every` | `10` | Periodic checkpoint frequency |
|       | `--log-every` | `100` | Step-level log frequency during training |
|       | `--tsne-every` | `0` | t-SNE plot frequency (0 disables) |
| paths | `--experiment-name` | `plant_retrieval` | Name of the output folder |
|       | `--output-dir` | `logs` | Root directory for logs, plots, configs |
|       | `--checkpoint-dir` | `logs/checkpoints` | Root directory for checkpoints |

## Output layout

After training, you will get:

```text
logs/<experiment-name>/
├── config.json              # Full config snapshot (for reproducibility)
├── class_to_idx.json        # Class name -> index mapping
├── train.log                # Plain-text log
├── metrics_<timestamp>.csv  # Per-epoch metrics (used for plotting)
├── metrics_<timestamp>.json # NDJSON version of the same metrics
├── test_results.json        # Final metrics on the test set
└── plots/
    ├── metrics.png          # Loss and accuracy curves (overwritten each epoch)
    └── tsne_epoch_XXX.png   # Only produced when --tsne-every > 0

logs/checkpoints/<experiment-name>/
├── best.pt                  # Checkpoint with the best validation accuracy
├── last.pt                  # Checkpoint from the final epoch
└── epoch_XXX.pt             # Periodic checkpoints controlled by --save-every
```

## Checkpoint format

Every `.pt` file contains:

```python
{
    "epoch": int,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "metrics": {"loss": ..., "acc": ...},
    "best_metric": float,
}
```

Loading a checkpoint manually:

```python
import torch
from src.config import Config
from src.model import OrgansClassifier

cfg = Config.load("logs/resnet50_baseline/config.json")
model = OrgansClassifier(
    backbone=cfg.model.backbone,
    pretrained=False,
    num_classes=cfg.model.num_classes,
)
ckpt = torch.load(
    "logs/checkpoints/resnet50_baseline/best.pt",
    map_location="cpu",
)
model.load_state_dict(ckpt["model"])
model.eval()
```

## Model architecture

`OrgansClassifier` (`src/model.py`):

```text
Input (B, 3, H, W)
    |
    v
timm backbone (frozen, num_classes=0)
    |  -> features (B, in_features)
    v
Linear(in_features -> in_features / 2)
    v
GELU
    v
Dropout(0.3)
    v
Linear(in_features / 2 -> num_classes)
    |
    v
Logits (B, num_classes)
```

The backbone parameters are frozen (`requires_grad=False`) and kept in `eval()` mode so that BatchNorm statistics do not drift. Only the classifier head is trained.

## Training loop overview

At each epoch, `Trainer.fit`:

1. Runs `train_one_epoch` with AMP and a tqdm progress bar.
2. Runs `validate` on the validation set.
3. Logs metrics to the console, file logger, CSV, and JSONL.
4. Regenerates `plots/metrics.png` (loss and accuracy curves).
5. Optionally generates a t-SNE plot if `tsne_every > 0` and the epoch matches.
6. Calls `scheduler.step()`.
7. Saves `best.pt` whenever validation accuracy improves.
8. Saves a periodic checkpoint when the epoch is a multiple of `save_every`.
9. Triggers early stopping if no improvement is seen for `patience` epochs.

When training finishes, `best.pt` is loaded and evaluated on the test loader. The final test metrics are written to `test_results.json`.



