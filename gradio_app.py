"""Gradio-based inference UI for RecycleVision."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from gradio.flagging import FlaggingCallback

from config import ExperimentConfig, default_config
from data import build_inference_transform, class_names_from_data
from model import create_model
from utils import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for RecycleVision inference")
    parser.add_argument("--config", type=Path, help="Path to YAML/JSON experiment config")
    parser.add_argument("--data-root", type=Path, help="Dataset root containing train/val folders")
    parser.add_argument("--num-classes", type=int, help="Number of classes (required without --config)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device for inference")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--server-name", type=str, default=None, help="Override Gradio server name")
    parser.add_argument("--server-port", type=int, default=None, help="Override Gradio server port")
    parser.add_argument("--flag-dir", type=Path, default=Path("flags/processed"), help="Directory to store flagged processed images")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)
    if args.data_root is None or args.num_classes is None:
        raise SystemExit("Either --config or both --data-root and --num-classes must be provided")
    config = default_config(args.data_root, args.num_classes)
    return config


def _select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(config: ExperimentConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = create_model(config.num_classes)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to a PIL image for saving."""
    to_pil = transforms.ToPILImage()
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0.0, 1.0)
    return to_pil(tensor.cpu())


def _prepare_predictor(
    config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
) -> Tuple[callable, List[str]]:
    data_cfg = config.data
    transform = build_inference_transform(data_cfg)
    class_names = class_names_from_data(data_cfg)
    model = _load_model(config, checkpoint, device)

    def predict(image: np.ndarray | Image.Image) -> Tuple[Dict[str, float], Image.Image]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        processed = _tensor_to_pil(tensor.squeeze(0))
        return {name: float(probs[idx]) for idx, name in enumerate(class_names)}, processed

    return predict, class_names


class ProcessedImageFlagger(FlaggingCallback):
    """Custom flagging callback that saves the processed model input."""

    def __init__(self, target_dir: Path):
        self.target_dir = Path(target_dir)
        self._count = 0

    def setup(self, components: List[gr.components.Component], flagging_dir: str) -> str:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.components = list(components)
        return str(self.target_dir)

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str | None = None,
        username: str | None = None,
    ) -> int:
        saved_path: Path | None = None

        for component, sample in zip(self.components, flag_data):
            if isinstance(component, gr.Image) and sample is not None:
                raw = component.flag(sample, self.target_dir)
                if raw:
                    try:
                        meta = json.loads(raw)
                        candidate = meta.get("path") if isinstance(meta, dict) else None
                    except json.JSONDecodeError:
                        candidate = raw
                    if candidate:
                        path = Path(candidate)
                        saved_path = path if path.is_absolute() else self.target_dir / path
                break

        if saved_path and saved_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            reason = (flag_option or "flagged").replace(" ", "_")
            final_path = self.target_dir / f"{timestamp}-{reason}{saved_path.suffix or '.png'}"
            saved_path.replace(final_path)

        self._count += 1
        return self._count


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)
    device = _select_device(args.device)
    if device.type != args.device:
        print(f"Requested device {args.device} unavailable; falling back to {device.type}.")

    predictor, class_names = _prepare_predictor(config, args.checkpoint, device)
    # flagger = ProcessedImageFlagger(args.flag_dir)

    iface = gr.Interface(
        fn=predictor,
        inputs=gr.Image(type="numpy", label="Upload waste image"),
        outputs=[
            gr.Label(num_top_classes=min(5, len(class_names)), label="Class probabilities"),
            gr.Image(type="pil", label="Processed Image", visible=False),
        ],
        title="RecycleVision Inference",
        description="Upload an image to classify recycling material using the trained RecycleVision model.",
        # allow_flagging="manual",
        # flagging_options=["incorrect", "uncertain", "other"],
        # flagging_callback=flagger,
    )

    launch_kwargs = {
        "share": args.share,
        "server_name": args.server_name,
        "server_port": args.server_port,
    }
    iface.launch(**{k: v for k, v in launch_kwargs.items() if v is not None or k == "share"})


if __name__ == "__main__":
    main()
