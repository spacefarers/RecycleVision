#!/usr/bin/env python3
"""
End-to-end quantization/export pipeline for RecycleVision.

Highlights:
    1. Loads the EfficientNet-based QAT checkpoint.
    2. Verifies that int8 accuracy matches the float32 teacher within a user-defined tolerance.
    3. Exports the sanitized float32 model (with in-graph normalization) to ONNX.
    4. Runs nncase PTQ using the provided calibration images to emit the final K230-ready kmodel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from config import TwoStageConfig
from model import create_model
from qat_utils import QATModelWrapper


# ---------------------------------------------------------------------------
# Model wrappers and export utilities
# ---------------------------------------------------------------------------

class Uint8ToNormalized(nn.Module):
    """Wrap the float32 backbone so the exported ONNX graph performs uint8 -> normalized float."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) / 255.0
        return self.backbone((x - self.mean) / self.std)


def export_to_onnx(model: nn.Module, input_shape: Tuple[int, int, int, int], output_path: Path) -> None:
    """Export a PyTorch model (uint8 input) to ONNX."""
    model.eval()
    dummy = torch.randint(low=0, high=256, size=input_shape, dtype=torch.uint8)
    print(f"Exporting ONNX to {output_path}")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    print("✓ ONNX export complete")


def simplify_onnx(input_path: Path, output_path: Path) -> None:
    """Run onnx-simplifier and drop unsupported attributes."""
    print("Simplifying ONNX graph...")
    onnx_model = onnx.load(str(input_path))
    try:
        simplified_model, check = onnxsim.simplify(onnx_model)
        if not check:
            print("⚠️  Warning: simplified model validation failed. Proceed with caution.")
    except RuntimeError as e:
        print(f"⚠️  onnxsim simplification failed: {e}")
        print("   Using unsimplified model as fallback.")
        simplified_model = onnx_model

    for node in simplified_model.graph.node:
        if node.op_type == "Reshape":
            # Filter out the 'allowzero' attribute
            attrs_to_keep = [attr for attr in node.attribute if attr.name != "allowzero"]
            del node.attribute[:]
            for attr in attrs_to_keep:
                node.attribute.append(attr)

    onnx.save(simplified_model, str(output_path))
    print(f"✓ Simplified model written to {output_path}")


def load_representative_images(
    image_paths: Iterable[Path],
    input_size: int,
    nchw: bool = True,
) -> List[List[np.ndarray]]:
    """Load representative calibration images as uint8 arrays."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for calibration. Install opencv-python.") from exc

    samples: List[List[np.ndarray]] = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        if nchw:
            tensor = np.transpose(img, (2, 0, 1))[None, ...]
        else:
            tensor = img[None, ...]
        samples.append([np.ascontiguousarray(tensor.astype(np.uint8))])
    return samples


def gather_image_paths(root: Path) -> List[Path]:
    """Collect calibration image paths recursively."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(ext))
    return files


def convert_to_kmodel(
    onnx_path: Path,
    output_path: Path,
    input_shape: Tuple[int, int, int, int],
    target: str,
    num_calibration_samples: int,
    calibration_dir: Path,
) -> None:
    """Call nncase PTQ to compile the ONNX graph into a kmodel."""
    print(f"Compiling kmodel for target={target} with nncase...")
    try:
        import nncase
    except ImportError as exc:
        raise ImportError("nncase is required for kmodel export. Install via `pip install nncase`.") from exc

    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_type = "uint8"
    compile_options.input_shape = list(input_shape)
    compile_options.input_layout = "NCHW"
    compile_options.preprocess = False
    compile_options.input_range = [0, 255]
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = "nncase_out"

    compiler = nncase.Compiler(compile_options)
    with open(onnx_path, "rb") as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())

    image_paths = gather_image_paths(calibration_dir)
    if not image_paths:
        raise RuntimeError(f"No calibration images found under {calibration_dir}")

    image_paths = sorted(image_paths)[:num_calibration_samples]
    samples = load_representative_images(image_paths, input_shape[-1], nchw=True)

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = len(samples)
    ptq_options.quant_type = "int8"
    ptq_options.w_quant_type = "int8"
    ptq_options.calibrate_method = "Kld"
    ptq_options.finetune_weights_method = "NoFineTuneWeights"
    ptq_options.set_tensor_data(samples)
    compiler.use_ptq(ptq_options)

    compiler.compile()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(compiler.gencode_tobytes())
    print(f"✓ kmodel saved to {output_path}")


# ---------------------------------------------------------------------------
# QAT helpers
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_eval_dir(data_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {path}")
        return path

    candidates = ["val", "validation", "eval", "test"]
    for name in candidates:
        candidate = data_root / name
        if candidate.exists():
            return candidate
    if (data_root / "train").exists():
        print("⚠️  Using train split for evaluation (val/test not found).")
        return data_root / "train"
    return data_root


def build_eval_loader(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    eval_dir: Optional[Path],
    max_samples: Optional[int],
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset_root = resolve_eval_dir(data_root, eval_dir)
    dataset = datasets.ImageFolder(dataset_root, transform=transform)

    if max_samples is not None and max_samples < len(dataset):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    accuracy = 100.0 * correct / max(1, total)
    return {"accuracy": accuracy, "samples": total}


def extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint format not recognized. Expected a dict with model weights.")


def detect_qat_artifacts(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any("_fake_quant" in key or "activation_post_process" in key for key in state_dict)


def prepare_qat_model(
    state_dict: Dict[str, torch.Tensor],
    num_classes: int,
    backend: str,
    per_channel: bool,
    requires_prepare: bool,
    device: torch.device,
) -> QATModelWrapper:
    base = create_model(num_classes=num_classes, pretrained=False)
    wrapper = QATModelWrapper(base, backend=backend, per_channel=per_channel)
    if requires_prepare:
        wrapper.prepare_qat()
    load_result = wrapper.load_state_dict(state_dict, strict=requires_prepare)
    if hasattr(load_result, "missing_keys") and load_result.missing_keys:
        print(f"⚠️  Missing keys when loading QAT state: {len(load_result.missing_keys)}")
    if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
        print(f"⚠️  Unexpected keys when loading QAT state: {len(load_result.unexpected_keys)}")
    wrapper.to(device)
    return wrapper


def strip_qat_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    blocked_tokens = ("_fake_quant", "activation_post_process", "quant.", "dequant.")

    def remove_prefix(key: str, prefix: str) -> str:
        return key[len(prefix) :] if key.startswith(prefix) else key

    clean_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        candidate = remove_prefix(remove_prefix(key, "module."), "model.")
        if any(token in candidate for token in blocked_tokens):
            continue
        clean_state[candidate] = tensor
    return clean_state


def load_checkpoint(path: Path) -> Dict:
    add_safe_globals([TwoStageConfig])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint payload type: {type(checkpoint)}")
    return checkpoint


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT-aware quantization/export pipeline for EfficientNet.")
    parser.add_argument("--input", type=str, default="checkpoints/finetune/best-epoch0.pt", help="Path to QAT checkpoint.")
    parser.add_argument("--output", type=str, default="models/recyclevision_qat.kmodel", help="Destination kmodel path.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes (otherwise taken from checkpoint).")
    parser.add_argument("--device", type=str, default="auto", help="Device for evaluation (cuda, mps, cpu, auto).")
    parser.add_argument("--backend", type=str, default="qnnpack", choices=["qnnpack", "fbgemm", "x86"], help="Quantization backend for PyTorch QAT.")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weight quantization during verification.")
    parser.add_argument("--data-root", type=str, default="data/sorted_2_class", help="Dataset root used for evaluation and fallback discovery.")
    parser.add_argument("--eval-dir", type=str, default=None, help="Explicit ImageFolder directory for evaluation (optional).")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers for evaluation.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Limit evaluation to N samples (random subset).")
    parser.add_argument("--eval-batches", type=int, default=None, help="Limit evaluation to N batches.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip accuracy verification.")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.75, help="Allowed accuracy drop (percentage points).")
    parser.add_argument("--input-size", type=int, default=224, help="Square input resolution used during export/eval.")
    parser.add_argument("--target", type=str, default="k230", choices=["k210", "k510", "k230"], help="nncase target.")
    parser.add_argument("--calibration-dir", type=str, required=True, help="Directory with representative calibration images.")
    parser.add_argument("--calibration-samples", type=int, default=256, help="Number of calibration samples.")
    parser.add_argument("--onnx-dir", type=str, default="conversion_output", help="Directory for intermediate ONNX artifacts.")
    parser.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.quantized.engine = args.backend

    checkpoint_path = Path(args.input)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    raw_state = extract_state_dict(checkpoint)
    requires_qat = detect_qat_artifacts(raw_state)

    num_classes = (
        args.num_classes
        or checkpoint.get("num_classes")
        or getattr(checkpoint.get("config"), "num_classes", None)
        or 3
    )

    device = resolve_device(args.device)
    print("\n" + "=" * 70)
    print("RecycleVision QAT Quantization Pipeline")
    print("=" * 70)
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Classes:         {num_classes}")
    print(f"Device:          {device}")
    print(f"Backend:         {args.backend} (per-channel={args.per_channel})")
    print(f"Eval dataset:    {args.eval_dir or args.data_root}")
    print(f"Calibration dir: {args.calibration_dir}")
    print("=" * 70 + "\n")

    eval_metrics = None
    if args.skip_eval:
        print("Skipping accuracy verification (--skip-eval).")
    elif not requires_qat:
        print("⚠️  Checkpoint does not contain QAT observers. Accuracy verification skipped.")
    else:
        eval_loader = build_eval_loader(
            data_root=Path(args.data_root),
            image_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_dir=Path(args.eval_dir) if args.eval_dir else None,
            max_samples=args.eval_samples,
        )

        qat_float = prepare_qat_model(raw_state, num_classes, args.backend, args.per_channel, requires_qat, device)
        qat_int8 = prepare_qat_model(raw_state, num_classes, args.backend, args.per_channel, requires_qat, device)
        qat_int8.convert_to_int8()

        float_stats = evaluate_accuracy(qat_float, eval_loader, device, args.eval_batches)
        int8_stats = evaluate_accuracy(qat_int8, eval_loader, device, args.eval_batches)
        accuracy_drop = float_stats["accuracy"] - int8_stats["accuracy"]

        print("\n--- Accuracy Verification ---")
        print(f"Float32 (QAT) accuracy: {float_stats['accuracy']:.2f}%")
        print(f"Int8 accuracy:          {int8_stats['accuracy']:.2f}%")
        print(f"Drop:                   {accuracy_drop:.2f} percentage points")

        if accuracy_drop > args.max_accuracy_drop:
            raise RuntimeError(
                f"Accuracy drop ({accuracy_drop:.2f}pp) exceeds threshold ({args.max_accuracy_drop}pp). "
                "Inspect QAT training or increase calibration coverage."
            )
        eval_metrics = {
            "float32_accuracy": float_stats["accuracy"],
            "int8_accuracy": int8_stats["accuracy"],
            "accuracy_drop": accuracy_drop,
        }

    print("\n--- Preparing float32 export model ---")
    clean_state = strip_qat_keys(raw_state)
    deploy_model = create_model(num_classes=num_classes, pretrained=False)
    missing = deploy_model.load_state_dict(clean_state, strict=False)
    if hasattr(missing, "missing_keys") and missing.missing_keys:
        print(f"⚠️  Missing {len(missing.missing_keys)} keys when loading deployment model.")
    if hasattr(missing, "unexpected_keys") and missing.unexpected_keys:
        print(f"⚠️  Unexpected {len(missing.unexpected_keys)} keys when loading deployment model.")
    deploy_model.eval()
    wrapped_model = Uint8ToNormalized(deploy_model).eval()

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"
    simplified_path = onnx_dir / "model_simplified.onnx"
    input_shape = (1, 3, args.input_size, args.input_size)

    export_to_onnx(wrapped_model, input_shape, onnx_path)
    if args.no_simplify:
        simplified_path = onnx_path
    else:
        simplify_onnx(onnx_path, simplified_path)

    convert_to_kmodel(
        onnx_path=simplified_path,
        output_path=Path(args.output),
        input_shape=input_shape,
        target=args.target,
        num_calibration_samples=args.calibration_samples,
        calibration_dir=Path(args.calibration_dir),
    )

    print("\n" + "=" * 70)
    print("✓ Quantization pipeline completed successfully!")
    if eval_metrics:
        print(f"Float32 accuracy: {eval_metrics['float32_accuracy']:.2f}%")
        print(f"Int8 accuracy:    {eval_metrics['int8_accuracy']:.2f}%")
        print(f"Accuracy drop:    {eval_metrics['accuracy_drop']:.2f}pp")
    print(f"kmodel saved to:  {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
