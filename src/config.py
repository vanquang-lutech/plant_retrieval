from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
from pathlib import Path
import json

@dataclass
class DataConfig:
    root_dir: str = "data/plant"
    image_size: int = 336
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    padding_color: tuple = (128, 128, 128)
@dataclass
class ModelConfig:
    backbone: str = ""
    pretrained: bool = True
    num_classes: Optional[int] = None

@dataclass
class OptimizerConfig:
    optimizer: str = "AdamW"
    lr: float = 5e-5
    weight_decay: float = 1e-4
    momentum: float = 0.9

    scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 1
    step_size: int = 10
    gamma: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    save_every: int = 10
    early_stopping_patience: Optional[int] = None
    tsne_every: int = 10

@dataclass
class PathConfig:
    output_dir: str = "logs"
    checkpoint_dir: str = "logs/checkpoints"
    experiment_name: str = "plant_retrieval"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            optim=OptimizerConfig(**data.get("optim", {})),
            train=TrainConfig(**data.get("train", {})),
            paths=PathConfig(**data.get("paths", {})),
        )

default_config = Config()
    