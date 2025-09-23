"""Entry point for RecycleVision training jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig, default_config
from train import train
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RecycleVision MobileNetV3 trainer")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root directory containing train/val subfolders",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of target classes. Required without --config.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)
    if args.data_root is None or args.num_classes is None:
        raise SystemExit("--data-root and --num-classes are required when no --config is provided")
    return default_config(args.data_root, args.num_classes)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()
