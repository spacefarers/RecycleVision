"""Configuration utilities for RecycleVision."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _expand(path: str | Path) -> Path:
    """Ensure that user-provided paths expand to absolute Paths."""
    return Path(path).expanduser().resolve()


@dataclass(slots=True)
class DataConfig:
    """Configuration for dataset loading."""

    root: Path
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    augment: bool = True

    def __post_init__(self) -> None:
        self.root = _expand(self.root)
        if self.train_dir is not None:
            self.train_dir = _expand(self.train_dir)
        if self.val_dir is not None:
            self.val_dir = _expand(self.val_dir)


@dataclass(slots=True)
class OptimConfig:
    """Optimizer hyperparameters."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    warmup_steps: int = 1000
    label_smoothing: float = 0.1


@dataclass(slots=True)
class TrainConfig:
    """High level training run configuration."""

    epochs: int = 30
    mixed_precision: bool = False
    log_interval: int = 20
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "mps"
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    # Run validation every N epochs (1 = every epoch)
    eval_interval_epochs: int = 5
    mixup_alpha: float = 0.4
    log_timing: bool = False

    def __post_init__(self) -> None:
        self.checkpoint_dir = _expand(self.checkpoint_dir)


@dataclass(slots=True)
class ExperimentConfig:
    """Full experiment configuration bound together."""

    num_classes: int
    data: DataConfig
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


def default_config(data_root: str | Path, num_classes: int) -> ExperimentConfig:
    """Convenience factory to produce a default configuration."""
    return ExperimentConfig(
        num_classes=num_classes,
        data=DataConfig(root=data_root),
    )
