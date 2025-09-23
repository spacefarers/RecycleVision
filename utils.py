"""Utility helpers for RecycleVision."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from config import DataConfig, ExperimentConfig, OptimConfig, TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML or JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".json"}:
            payload: Dict[str, Any] = json.load(handle)
        else:
            payload = yaml.safe_load(handle)  # type: ignore[assignment]

    data_cfg = payload.pop("data")
    opt_cfg = payload.pop("optimizer", {})
    train_cfg = payload.pop("training", {})

    return ExperimentConfig(
        data=DataConfig(**data_cfg),
        optimizer=OptimConfig(**opt_cfg),
        training=TrainConfig(**train_cfg),
        **payload,
    )
