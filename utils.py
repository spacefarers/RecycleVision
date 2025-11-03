"""Utility helpers for RecycleVision."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from config import (
    DataConfig, 
    ExperimentConfig, 
    OptimConfig, 
    TrainConfig,
    TwoStageConfig,
    PretrainConfig,
    FinetuneConfig,
)


def load_config(path: str | Path) -> Union[ExperimentConfig, TwoStageConfig]:
    """Load an experiment configuration from a YAML or JSON file.
    
    Automatically detects whether it's a single-stage (ExperimentConfig) or
    two-stage (TwoStageConfig) configuration based on the presence of 
    'pretrain' and 'finetune' keys.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".json"}:
            payload: Dict[str, Any] = json.load(handle)
        else:
            payload = yaml.safe_load(handle)  # type: ignore[assignment]

    # Check if this is a two-stage config
    if "pretrain" in payload or "finetune" in payload:
        pretrain_cfg = payload.pop("pretrain", {})
        finetune_cfg = payload.pop("finetune", {})
        opt_cfg = payload.pop("optimizer", {})
        train_cfg = payload.pop("training", {})
        
        return TwoStageConfig(
            pretrain=PretrainConfig(**pretrain_cfg),
            finetune=FinetuneConfig(**finetune_cfg),
            optimizer=OptimConfig(**opt_cfg),
            training=TrainConfig(**train_cfg),
            **payload,
        )
    else:
        # Original single-stage config
        data_cfg = payload.pop("data")
        opt_cfg = payload.pop("optimizer", {})
        train_cfg = payload.pop("training", {})

        return ExperimentConfig(
            data=DataConfig(**data_cfg),
            optimizer=OptimConfig(**opt_cfg),
            training=TrainConfig(**train_cfg),
            **payload,
        )
