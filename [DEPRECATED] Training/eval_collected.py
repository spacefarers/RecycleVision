#!/usr/bin/env python3
"""
Evaluation script comparing unquantized vs quantized models on collected images.
Generates a performance degradation report.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from datetime import datetime

from model import create_model
from qat_utils import QATModelWrapper


class CollectedImageEvaluator:
    """Evaluate both unquantized and quantized models on collected images."""

    def __init__(self, num_classes: int = 3, device: str = "cuda"):
        self.num_classes = num_classes
        self.device_fp32 = torch.device(device if torch.cuda.is_available() else "cpu")
        # INT8 quantized models must run on CPU
        self.device_int8 = torch.device("cpu")
        self.class_names = ["Recyclable", "Trash", "Empty"][:num_classes]
        self.input_size = 224

        # Normalization values for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image to uint8 format."""
        try:
            img = Image.open(image_path).convert("RGB")
            # Resize
            img = img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
            # Convert to uint8 numpy array
            img_array = np.array(img, dtype=np.uint8)
            return img_array
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def load_unquantized_model(self, checkpoint_path: Path) -> nn.Module:
        """Load unquantized PyTorch model."""
        print(f"Loading unquantized model from {checkpoint_path}...")
        model = create_model(num_classes=self.num_classes, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device_fp32, weights_only=False)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device_fp32)
        model.eval()
        return model

    def load_quantized_model_pytorch(self, checkpoint_path: Path) -> nn.Module:
        """Load QAT PyTorch model with fake quantization (FP32 evaluation)."""
        print(f"Loading QAT model from {checkpoint_path}...")
        model = create_model(num_classes=self.num_classes, pretrained=False)
        wrapper = QATModelWrapper(model, backend="qnnpack", per_channel=False)
        wrapper.prepare_qat()

        checkpoint = torch.load(checkpoint_path, map_location=self.device_fp32, weights_only=False)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        wrapper.load_state_dict(state_dict, strict=False)
        # Keep in FP32 with fake quantization for evaluation
        wrapper = wrapper.to(self.device_fp32)
        wrapper.eval()
        return wrapper

    def infer_pytorch(self, model: nn.Module, image_array: np.ndarray, device: torch.device = None) -> Tuple[int, float, float]:
        """Run inference on PyTorch model and return class, confidence, and latency."""
        if device is None:
            device = self.device_fp32

        # Convert uint8 to normalized tensor
        x = torch.from_numpy(image_array).to(device, dtype=torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW
        x = x / 255.0  # Normalize to [0, 1]
        x = (x - self.mean.to(device)) / self.std.to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            output = model(x)
            latency = (time.perf_counter() - start_time) * 1000  # ms

        probs = torch.softmax(output, dim=1)
        confidence, class_id = torch.max(probs, dim=1)

        return class_id.item(), confidence.item(), latency

    def infer_onnx(self, session: ort.InferenceSession, image_array: np.ndarray) -> Tuple[int, float, float]:
        """Run inference on ONNX model and return class, confidence, and latency."""
        # Prepare input: uint8 -> float32 normalized
        x = image_array.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)) / \
            np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)  # Add batch dimension

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: x.astype(np.float32)})
        latency = (time.perf_counter() - start_time) * 1000  # ms

        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        class_id = np.argmax(logits)
        confidence = probs[class_id]

        return int(class_id), float(confidence), latency

    def evaluate_on_collected_images(
        self,
        collected_dir: Path,
        checkpoint_path: Path,
        onnx_path: Path = None,
    ) -> Dict:
        """Evaluate both unquantized and quantized models."""

        collected_dir = Path(collected_dir)
        image_files = sorted([f for f in collected_dir.glob("*.png") if f.is_file()])

        if not image_files:
            print(f"No PNG images found in {collected_dir}")
            return {}

        print(f"Found {len(image_files)} collected images\n")

        # Load models
        model_fp32 = self.load_unquantized_model(checkpoint_path)
        model_int8 = self.load_quantized_model_pytorch(checkpoint_path)

        onnx_session = None
        if onnx_path and Path(onnx_path).exists():
            print(f"Loading ONNX model from {onnx_path}...")
            try:
                onnx_session = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"]  # Use CPU provider as fallback
                )
            except Exception as e:
                print(f"Warning: Failed to load ONNX model: {e}")
                print("Continuing evaluation with PyTorch models only...")

        # Run inference
        results = {
            "fp32_predictions": [],
            "int8_predictions": [],
            "onnx_predictions": [],
            "latencies": {"fp32": [], "int8": [], "onnx": []},
            "agreement": {"fp32_vs_int8": 0, "fp32_vs_onnx": 0},
        }

        print("Running inference...")
        for idx, image_path in enumerate(image_files):
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images")

            image_array = self.load_image(image_path)
            if image_array is None:
                continue

            # FP32 inference
            class_fp32, conf_fp32, lat_fp32 = self.infer_pytorch(model_fp32, image_array, self.device_fp32)
            results["fp32_predictions"].append({
                "file": image_path.name,
                "class": class_fp32,
                "confidence": conf_fp32,
            })
            results["latencies"]["fp32"].append(lat_fp32)

            # QAT inference (FP32 with fake quantization)
            class_int8, conf_int8, lat_int8 = self.infer_pytorch(model_int8, image_array, self.device_fp32)
            results["int8_predictions"].append({
                "file": image_path.name,
                "class": class_int8,
                "confidence": conf_int8,
            })
            results["latencies"]["int8"].append(lat_int8)

            # ONNX inference
            if onnx_session:
                class_onnx, conf_onnx, lat_onnx = self.infer_onnx(onnx_session, image_array)
                results["onnx_predictions"].append({
                    "file": image_path.name,
                    "class": class_onnx,
                    "confidence": conf_onnx,
                })
                results["latencies"]["onnx"].append(lat_onnx)

            # Track agreement
            if class_fp32 == class_int8:
                results["agreement"]["fp32_vs_int8"] += 1
            if onnx_session and class_fp32 == class_onnx:
                results["agreement"]["fp32_vs_onnx"] += 1

        return results

    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """Generate a comprehensive performance report."""

        if not results.get("fp32_predictions"):
            return "No results to report"

        num_samples = len(results["fp32_predictions"])

        # Calculate statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "latency_ms": {
                "fp32": {
                    "mean": np.mean(results["latencies"]["fp32"]),
                    "std": np.std(results["latencies"]["fp32"]),
                    "min": np.min(results["latencies"]["fp32"]),
                    "max": np.max(results["latencies"]["fp32"]),
                },
                "int8": {
                    "mean": np.mean(results["latencies"]["int8"]),
                    "std": np.std(results["latencies"]["int8"]),
                    "min": np.min(results["latencies"]["int8"]),
                    "max": np.max(results["latencies"]["int8"]),
                },
            },
            "agreement": {
                "fp32_vs_int8": {
                    "count": results["agreement"]["fp32_vs_int8"],
                    "percentage": 100.0 * results["agreement"]["fp32_vs_int8"] / num_samples,
                },
            },
        }

        if results["latencies"]["onnx"]:
            stats["latency_ms"]["onnx"] = {
                "mean": np.mean(results["latencies"]["onnx"]),
                "std": np.std(results["latencies"]["onnx"]),
                "min": np.min(results["latencies"]["onnx"]),
                "max": np.max(results["latencies"]["onnx"]),
            }
            stats["agreement"]["fp32_vs_onnx"] = {
                "count": results["agreement"]["fp32_vs_onnx"],
                "percentage": 100.0 * results["agreement"]["fp32_vs_onnx"] / num_samples,
            }

        # Calculate confidence differences
        conf_diff = []
        for fp32_pred, int8_pred in zip(results["fp32_predictions"], results["int8_predictions"]):
            conf_diff.append(abs(fp32_pred["confidence"] - int8_pred["confidence"]))

        stats["confidence_difference"] = {
            "mean": np.mean(conf_diff),
            "std": np.std(conf_diff),
            "min": np.min(conf_diff),
            "max": np.max(conf_diff),
        }

        # Generate report text
        report = []
        report.append("=" * 80)
        report.append("RecycleVision Model Performance Degradation Report")
        report.append("=" * 80)
        report.append(f"Timestamp: {stats['timestamp']}")
        report.append(f"Total samples evaluated: {stats['num_samples']}")
        report.append("")

        # Latency Analysis
        report.append("LATENCY ANALYSIS (milliseconds)")
        report.append("-" * 80)
        report.append(f"{'Model':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        report.append("-" * 80)

        lat_fp32 = stats["latency_ms"]["fp32"]
        report.append(f"{'FP32':<20} {lat_fp32['mean']:<12.4f} {lat_fp32['std']:<12.4f} {lat_fp32['min']:<12.4f} {lat_fp32['max']:<12.4f}")

        lat_int8 = stats["latency_ms"]["int8"]
        report.append(f"{'INT8 (QAT)':<20} {lat_int8['mean']:<12.4f} {lat_int8['std']:<12.4f} {lat_int8['min']:<12.4f} {lat_int8['max']:<12.4f}")

        # Latency degradation
        lat_degradation = 100 * (lat_int8["mean"] - lat_fp32["mean"]) / lat_fp32["mean"]
        lat_improvement = -lat_degradation  # Negative degradation = improvement
        report.append("")
        report.append(f"INT8 Latency vs FP32: {lat_improvement:+.2f}% {'(speedup)' if lat_improvement > 0 else '(slowdown)'}")

        if "onnx" in stats["latency_ms"]:
            lat_onnx = stats["latency_ms"]["onnx"]
            report.append(f"{'ONNX':<20} {lat_onnx['mean']:<12.4f} {lat_onnx['std']:<12.4f} {lat_onnx['min']:<12.4f} {lat_onnx['max']:<12.4f}")

        report.append("")
        report.append("PREDICTION AGREEMENT")
        report.append("-" * 80)

        agreement_fp32_int8 = stats["agreement"]["fp32_vs_int8"]
        report.append(f"FP32 vs INT8: {agreement_fp32_int8['count']}/{stats['num_samples']} ({agreement_fp32_int8['percentage']:.2f}%)")

        if "fp32_vs_onnx" in stats["agreement"]:
            agreement_fp32_onnx = stats["agreement"]["fp32_vs_onnx"]
            report.append(f"FP32 vs ONNX: {agreement_fp32_onnx['count']}/{stats['num_samples']} ({agreement_fp32_onnx['percentage']:.2f}%)")

        report.append("")
        report.append("CONFIDENCE DIFFERENCE (FP32 vs INT8)")
        report.append("-" * 80)
        conf = stats["confidence_difference"]
        report.append(f"Mean: {conf['mean']:.6f}")
        report.append(f"Std:  {conf['std']:.6f}")
        report.append(f"Min:  {conf['min']:.6f}")
        report.append(f"Max:  {conf['max']:.6f}")

        report.append("")
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"✓ Prediction agreement (FP32 vs INT8): {agreement_fp32_int8['percentage']:.1f}%")
        report.append(f"✓ Average confidence difference: {conf['mean']:.6f}")
        report.append(f"✓ Latency change: {lat_improvement:+.2f}%")
        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        # Save report
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")

        # Save JSON stats
        json_path = output_path.with_suffix(".json") if output_path else Path("eval_report.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"JSON stats saved to: {json_path}")

        return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate unquantized vs quantized models on collected images")
    parser.add_argument("--collected-dir", type=Path, default=Path("data/collected"),
                        help="Directory containing collected images")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetune/best-epoch0.pt"),
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx-model", type=Path, default=Path("conversion_output/model.onnx"),
                        help="Path to ONNX model (optional)")
    parser.add_argument("--output", type=Path, default=Path("eval_results/performance_report.txt"),
                        help="Output path for report")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    evaluator = CollectedImageEvaluator(num_classes=args.num_classes, device=args.device)

    print(f"Evaluating models on collected images from: {args.collected_dir}")
    results = evaluator.evaluate_on_collected_images(
        args.collected_dir,
        args.checkpoint,
        args.onnx_model if args.onnx_model.exists() else None,
    )

    report = evaluator.generate_report(results, args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
