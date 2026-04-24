from PIL import Image, ImageOps
import numpy as np
import csv
import json
import logging
import random
import sys
from pathlib import Path
import torch

from datetime import datetime
from typing import Optional


def resize_pwd(image, padding_color, target_size):
    return ImageOps.pad(
        image = image, 
        size = (target_size, target_size), 
        method = Image.BICUBIC,
        color = padding_color, 
        centering=(0.5, 0.5),
        )

def setup_logging(name, log_file=None, level=logging.INFO):

    """
    Setup logging for the given name and log file.
    - Format: [timestamp] [level] [message]
    - Log to console and file.
    - No duplicate logs.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    
    return logger


class AverageMeter:
    """
    Track running average of a metric.
    - Update: add new value to the running average.
    - Value: get the current average value.
    - Reset: reset the running average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
    
    def update(self, value, n):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

class MetricLogger:
    """
    Log metrics for each epoch.
    - Log and save metrics to a CSV file.
    - Metric save: epoch, train_loss, val_acc
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.json_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._fieldnames: Optional[list] = None

    def log(self, metric):
        with self.json_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        
        write_header = not self.csv_file.exists()
        if self._fieldnames is None:
            self._fieldnames = list(metric.keys())

        with self.csv_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(metric)


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    Seed everything for reproducibility.
    - Seed: random, numpy, torch, torch.cuda
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True



