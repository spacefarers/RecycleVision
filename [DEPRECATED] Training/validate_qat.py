"""
QAT Validation Script: Compare float32 and int8 quantized models.

This script:
1. Loads a QAT-trained model
2. Converts it to int8
3. Compares accuracy, inference speed, and model size
4. Provides detailed analysis of quantization impact
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple

# Import project modules
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from data import get_transforms, GarbageDataset
from qat_utils import QATModelWrapper, compare_qat_accuracy, build_calibration_dataset
from torch.utils.data import DataLoader, random_split


class QATValidator:
    """Validates QAT conversion and compares model performance"""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model_float32 = None
        self.model_int8 = None

    def _get_device(self, device: str) -> torch.device:
        """Get device (CUDA, MPS, or CPU)"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def load_checkpoint(self, checkpoint_path: str, num_classes: int = 3) -> Tuple[nn.Module, Dict]:
        """Load QAT model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        model = create_model(num_classes=num_classes, pretrained=False)
        model = QATModelWrapper(model)

        # Load state
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(self.device)
        model.eval()

        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  Accuracy at checkpoint: {checkpoint.get('val_accuracy', 'N/A')}%")

        return model, checkpoint

    def benchmark_model(self, model: nn.Module, test_loader: DataLoader, num_runs: int = 100) -> Dict:
        """Benchmark model inference speed"""
        model.eval()

        # Warmup
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                _ = model(images)
                break

        # Timed runs
        import time
        times = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                torch.cuda.synchronize() if self.device.type == "cuda" else None
                start = time.perf_counter()

                for _ in range(num_runs):
                    _ = model(images)

                torch.cuda.synchronize() if self.device.type == "cuda" else None
                end = time.perf_counter()

                elapsed = (end - start) / num_runs * 1000  # ms per batch
                times.append(elapsed)

                if len(times) >= 10:
                    break

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return {
            "avg_latency_ms": avg_time,
            "min_latency_ms": min_time,
            "max_latency_ms": max_time,
            "runs": len(times),
        }

    def get_model_size(self, model: nn.Module) -> Dict:
        """Calculate model size"""
        param_count = sum(p.numel() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

        # Estimate total size
        state_dict = model.state_dict()
        state_dict_size = 0
        for tensor in state_dict.values():
            state_dict_size += tensor.numel() * tensor.element_size()

        return {
            "parameters": param_count,
            "parameters_mb": param_count * 4 / (1024 ** 2),  # Assuming float32
            "state_dict_mb": state_dict_size / (1024 ** 2),
            "total_mb": (buffer_size + state_dict_size) / (1024 ** 2),
        }

    def validate_qat_conversion(self, checkpoint_path: str, data_root: str,
                               batch_size: int = 32, num_eval_samples: int = 500) -> Dict:
        """
        Full QAT validation pipeline.

        Args:
            checkpoint_path: Path to QAT checkpoint
            data_root: Data directory
            batch_size: Batch size for evaluation
            num_eval_samples: Number of samples for evaluation
        """
        print("\n" + "="*70)
        print("QAT VALIDATION PIPELINE")
        print("="*70)

        # 1. Load checkpoint
        print("\n[1/5] Loading QAT model...")
        self.model_float32, checkpoint = self.load_checkpoint(checkpoint_path)

        # 2. Load validation data
        print("\n[2/5] Loading validation data...")
        dataset = GarbageDataset(
            data_root=data_root,
            transform=get_transforms("val")
        )

        val_size = min(num_eval_samples, len(dataset))
        val_dataset = random_split(dataset, [val_size, len(dataset) - val_size])[0]

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"✓ Loaded {len(val_dataset)} validation samples")

        # 3. Evaluate float32 model
        print("\n[3/5] Evaluating float32 model...")
        self.model_float32.eval()

        correct_float32 = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model_float32(images)
                _, predictions = torch.max(outputs, 1)
                correct_float32 += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy_float32 = 100.0 * correct_float32 / total
        print(f"Float32 Accuracy: {accuracy_float32:.2f}%")

        # 4. Convert to int8
        print("\n[4/5] Converting to int8...")
        self.model_int8 = QATModelWrapper(self.model_float32.model)
        self.model_int8.load_state_dict(self.model_float32.state_dict())
        self.model_int8 = self.model_int8.to(self.device)
        self.model_int8.convert_to_int8()

        # 5. Evaluate int8 model
        print("\n[5/5] Evaluating int8 model...")
        self.model_int8.eval()

        correct_int8 = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model_int8(images)
                _, predictions = torch.max(outputs, 1)
                correct_int8 += (predictions == labels).sum().item()

        accuracy_int8 = 100.0 * correct_int8 / total
        print(f"Int8 Accuracy: {accuracy_int8:.2f}%")

        # Comparison metrics
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)

        accuracy_drop = accuracy_float32 - accuracy_int8
        print(f"\nAccuracy Comparison:")
        print(f"  Float32: {accuracy_float32:.2f}%")
        print(f"  Int8:    {accuracy_int8:.2f}%")
        print(f"  Drop:    {accuracy_drop:.2f}%")

        if accuracy_drop < 0.5:
            status = "✓ EXCELLENT"
        elif accuracy_drop < 1.0:
            status = "✓ GOOD"
        elif accuracy_drop < 2.0:
            status = "⚠️ FAIR"
        else:
            status = "❌ POOR"

        print(f"  Status:  {status}")

        # Model size comparison
        print(f"\nModel Size:")
        size_float32 = self.get_model_size(self.model_float32)
        size_int8 = self.get_model_size(self.model_int8)

        print(f"  Float32: {size_float32['total_mb']:.2f} MB")
        print(f"  Int8:    {size_int8['total_mb']:.2f} MB")
        print(f"  Reduction: {(1 - size_int8['total_mb']/size_float32['total_mb'])*100:.1f}%")

        # Inference speed (if possible)
        print(f"\nInference Speed (sampled):")
        try:
            speed_float32 = self.benchmark_model(self.model_float32, val_loader, num_runs=10)
            speed_int8 = self.benchmark_model(self.model_int8, val_loader, num_runs=10)

            print(f"  Float32: {speed_float32['avg_latency_ms']:.2f} ms/batch (avg)")
            print(f"  Int8:    {speed_int8['avg_latency_ms']:.2f} ms/batch (avg)")
            speedup = speed_float32['avg_latency_ms'] / speed_int8['avg_latency_ms']
            print(f"  Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"  ⚠️ Could not benchmark: {e}")
            speed_float32 = speed_int8 = None

        # Summary report
        results = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "device": str(self.device),
            "accuracy": {
                "float32": accuracy_float32,
                "int8": accuracy_int8,
                "drop": accuracy_drop,
                "status": status,
            },
            "model_size": {
                "float32_mb": size_float32['total_mb'],
                "int8_mb": size_int8['total_mb'],
                "reduction_percent": (1 - size_int8['total_mb']/size_float32['total_mb'])*100,
            },
            "inference_speed": {
                "float32_ms": speed_float32['avg_latency_ms'] if speed_float32 else None,
                "int8_ms": speed_int8['avg_latency_ms'] if speed_int8 else None,
                "speedup": speedup if speed_float32 and speed_int8 else None,
            },
            "eval_samples": len(val_dataset),
        }

        print("\n" + "="*70)
        print("✓ Validation complete!")
        print("="*70 + "\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="QAT Validation Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to QAT checkpoint")
    parser.add_argument("--data-root", type=str, default="data/sorted_2_class",
                        help="Data directory for validation")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Number of samples for evaluation")
    parser.add_argument("--save-report", action="store_true",
                        help="Save validation report to JSON")

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Run validation
    validator = QATValidator(device=args.device)
    results = validator.validate_qat_conversion(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_eval_samples=args.eval_samples
    )

    # Save report if requested
    if args.save_report:
        report_path = Path(args.checkpoint).parent / f"qat_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved validation report: {report_path}")


if __name__ == "__main__":
    main()
