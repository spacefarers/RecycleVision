"""
Quantization-Aware Training (QAT) utilities for 8-bit training.
Prepares and converts models for int8 quantization.
"""

import torch
import torch.nn as nn
import torch.quantization as tq
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum


class CalibrationStrategy(Enum):
    """Calibration strategies for quantization parameter selection"""
    MINMAX = "minmax"  # Use min/max values
    ENTROPY = "entropy"  # KL divergence minimization
    PERCENTILE = "percentile"  # Use percentile values (0.1%, 99.9%)
    MOVING_AVERAGE = "moving_average"  # Running average across batches


class QATModelWrapper(nn.Module):
    """
    Wraps a model with quantization-aware training layers.
    Simulates int8 quantization during forward pass.
    """

    def __init__(self, model: nn.Module, backend: str = "qnnpack", per_channel: bool = False):
        """
        Args:
            model: PyTorch model to wrap
            backend: Quantization backend ("qnnpack", "fbgemm", "x86")
            per_channel: Use per-channel quantization instead of per-tensor (more accurate)
        """
        super().__init__()
        self.model = model
        self.backend = backend
        self.per_channel = per_channel
        self.calibration_data = {}

        # Quantization stubs for input/output
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization stubs"""
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def prepare_qat(self, calibration_method: str = "histogram"):
        """
        Prepare model for QAT.
        Inserts fake quantization layers after weights and activations.

        Args:
            calibration_method: "histogram", "min_max", or "entropy"
        """
        # Create quantization config with per-channel support
        if self.per_channel:
            self.model.qconfig = tq.QConfig(
                activation=tq.HistogramObserver.with_args(reduce_range=True),
                weight=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
            )
        else:
            self.model.qconfig = tq.get_default_qat_qconfig(self.backend)

        # Prepare for QAT - this inserts fake_quantize modules
        tq.prepare_qat(self, inplace=True)
        backend_str = f"{self.backend} (per-channel)" if self.per_channel else self.backend
        print(f"✓ Model prepared for QAT (backend: {backend_str}, method: {calibration_method})")

    def convert_to_int8(self):
        """
        Convert QAT model to pure int8 model.
        Replaces fake quantization with actual int8 ops.
        """
        tq.convert(self, inplace=True)
        print("✓ Model converted to int8")

    def set_calibration_mode(self):
        """Set model to calibration mode (observer mode)"""
        for module in self.modules():
            if isinstance(module, (tq.QuantStub, tq.DeQuantStub)):
                module.set_observer_enabled(True)

    def freeze_quantization(self):
        """Freeze quantization parameters after calibration"""
        for module in self.modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.scale = module.weight_fake_quant.scale.detach()
                module.weight_fake_quant.zero_point = module.weight_fake_quant.zero_point.detach()


class QuantizationMonitor:
    """Monitor quantization statistics during training"""

    def __init__(self):
        self.stats = {}
        self.convergence_history = []

    def log_quantization_stats(self, model: nn.Module, step: int) -> Dict:
        """
        Log quantization scale and zero-point information.
        Helps debug quantization issues and track convergence.
        """
        stats = {
            "weights": {},
            "activations": {},
            "layers_count": 0
        }

        for name, module in model.named_modules():
            # Weight quantization stats
            if hasattr(module, 'weight_fake_quant'):
                weight_quant = module.weight_fake_quant
                scale_val = weight_quant.scale.item() if weight_quant.scale.numel() == 1 else weight_quant.scale.mean().item()
                zero_point_val = weight_quant.zero_point.item() if weight_quant.zero_point.numel() == 1 else weight_quant.zero_point.mean().item()

                stats["weights"][name] = {
                    "scale": scale_val,
                    "zero_point": zero_point_val,
                    "scale_range": float(weight_quant.scale.max().item()) if weight_quant.scale.numel() > 1 else scale_val,
                }
                stats["layers_count"] += 1

            # Activation quantization stats
            if hasattr(module, 'activation_post_process'):
                act_quant = module.activation_post_process
                if hasattr(act_quant, 'scale') and hasattr(act_quant, 'zero_point'):
                    scale_val = act_quant.scale.item() if act_quant.scale.numel() == 1 else act_quant.scale.mean().item()
                    zero_point_val = act_quant.zero_point.item() if act_quant.zero_point.numel() == 1 else act_quant.zero_point.mean().item()

                    stats["activations"][name] = {
                        "scale": scale_val,
                        "zero_point": zero_point_val,
                    }

        self.stats[step] = stats
        return stats

    def print_calibration_status(self, verbose: bool = False):
        """Print calibration convergence status"""
        if len(self.stats) < 2:
            print("⚠️  Need at least 2 calibration steps for convergence analysis")
            return

        print("\n" + "="*70)
        print("QUANTIZATION CALIBRATION CONVERGENCE REPORT")
        print("="*70)

        initial_step = list(self.stats.keys())[0]
        latest_step = list(self.stats.keys())[-1]

        initial_stats = self.stats[initial_step]
        latest_stats = self.stats[latest_step]

        total_layers = initial_stats.get("layers_count", 0)
        print(f"Total Quantized Layers: {total_layers}")
        print(f"Calibration Steps: {latest_step - initial_step + 1}\n")

        # Weight convergence analysis
        if initial_stats["weights"]:
            print("WEIGHT QUANTIZATION CONVERGENCE:")
            print("-" * 70)
            scale_changes = []

            for layer_name, init_weight_stats in initial_stats["weights"].items():
                if layer_name in latest_stats["weights"]:
                    latest_weight_stats = latest_stats["weights"][layer_name]
                    scale_change = abs(latest_weight_stats['scale'] - init_weight_stats['scale']) / (init_weight_stats['scale'] + 1e-7)
                    scale_changes.append(scale_change)

                    if verbose or scale_change > 0.1:
                        status = "⚠️ UNSTABLE" if scale_change > 0.2 else "✓ STABLE"
                        print(f"  {layer_name}: {status}")
                        print(f"    Initial scale: {init_weight_stats['scale']:.6f}")
                        print(f"    Latest scale:  {latest_weight_stats['scale']:.6f}")
                        print(f"    Change: {scale_change*100:.2f}%\n")

            avg_change = np.mean(scale_changes) if scale_changes else 0
            print(f"Average Scale Change: {avg_change*100:.2f}%")
            convergence_level = "Excellent" if avg_change < 0.05 else "Good" if avg_change < 0.1 else "Fair" if avg_change < 0.2 else "Poor"
            print(f"Convergence Level: {convergence_level}\n")

    def get_layer_count(self) -> int:
        """Get number of quantized layers"""
        if not self.stats:
            return 0
        latest_step = list(self.stats.keys())[-1]
        return self.stats[latest_step].get("layers_count", 0)


class QATLossScaler:
    """
    Custom loss scaling for QAT training.
    Prevents gradient overflow during quantization.
    """

    def __init__(self, init_scale: float = 2.0 ** 16):
        self.loss_scale = init_scale
        self.min_loss_scale = 1.0
        self.max_loss_scale = 2.0 ** 24
        self.scale_factor = 2.0
        self.scale_window = 2000

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass"""
        return loss * self.loss_scale

    def unscale_gradients(self, optimizer):
        """Unscale gradients after backward pass"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.loss_scale)

    def update_scale(self, overflow: bool = False):
        """Update loss scale based on overflow detection"""
        if overflow:
            self.loss_scale = max(self.loss_scale / self.scale_factor, self.min_loss_scale)
        else:
            self.loss_scale = min(self.loss_scale * self.scale_factor, self.max_loss_scale)


def calibrate_quantization(model: nn.Module,
                          calib_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          num_batches: int = 32) -> Dict:
    """
    Calibrate quantization parameters using representative data.

    Args:
        model: Model prepared for QAT
        calib_loader: DataLoader with calibration samples
        device: Device to run calibration on
        num_batches: Number of batches to use for calibration

    Returns:
        Dictionary with calibration statistics
    """
    model.eval()
    model.to(device)

    stats = {
        "num_batches": 0,
        "num_samples": 0,
        "avg_activations": [],
        "max_activations": [],
    }

    print(f"\nCalibrating quantization on {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calib_loader):
            if batch_idx >= num_batches:
                break

            images = images.to(device)
            _ = model(images)
            stats["num_batches"] += 1
            stats["num_samples"] += images.size(0)

            if (batch_idx + 1) % max(1, num_batches // 4) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches})")

    print(f"✓ Calibration complete ({stats['num_batches']} batches, {stats['num_samples']} samples)")
    return stats


def build_calibration_dataset(dataset, num_samples: int = 500, strategy: str = "random") -> torch.utils.data.Subset:
    """
    Build a calibration dataset from full dataset.

    Args:
        dataset: Full training dataset
        num_samples: Number of calibration samples to select
        strategy: Selection strategy ("random", "stratified", "uniform")

    Returns:
        torch.utils.data.Subset with calibration samples
    """
    dataset_size = len(dataset)
    num_samples = min(num_samples, dataset_size)

    if strategy == "random":
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
    elif strategy == "stratified":
        # Select samples uniformly across dataset (every k-th sample)
        step = max(1, dataset_size // num_samples)
        indices = list(range(0, dataset_size, step))[:num_samples]
    elif strategy == "uniform":
        indices = list(range(0, dataset_size, max(1, dataset_size // num_samples)))[:num_samples]
    else:
        indices = torch.randperm(dataset_size)[:num_samples].tolist()

    print(f"✓ Built calibration dataset: {len(indices)} samples ({strategy} strategy)")
    return torch.utils.data.Subset(dataset, indices)


def compare_qat_accuracy(model_float32: nn.Module,
                        model_int8: nn.Module,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device) -> Dict:
    """
    Compare accuracy between float32 and int8 models.
    Helps validate QAT conversion quality.

    Args:
        model_float32: QAT model before conversion
        model_int8: QAT model after conversion to int8
        test_loader: DataLoader with test samples
        device: Device to run evaluation on

    Returns:
        Dictionary with accuracy comparison
    """
    model_float32.eval()
    model_int8.eval()

    correct_float32 = 0
    correct_int8 = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Float32 predictions
            outputs_float32 = model_float32(images)
            _, preds_float32 = torch.max(outputs_float32.data, 1)
            correct_float32 += (preds_float32 == labels).sum().item()

            # Int8 predictions
            outputs_int8 = model_int8(images)
            _, preds_int8 = torch.max(outputs_int8.data, 1)
            correct_int8 += (preds_int8 == labels).sum().item()

            total += labels.size(0)

    acc_float32 = 100.0 * correct_float32 / total
    acc_int8 = 100.0 * correct_int8 / total
    accuracy_drop = acc_float32 - acc_int8

    comparison = {
        "float32_accuracy": acc_float32,
        "int8_accuracy": acc_int8,
        "accuracy_drop": accuracy_drop,
        "total_samples": total,
    }

    print(f"\n=== QAT Conversion Results ===")
    print(f"Float32 Accuracy: {acc_float32:.2f}%")
    print(f"Int8 Accuracy:    {acc_int8:.2f}%")
    print(f"Accuracy Drop:    {accuracy_drop:.2f}%")

    if accuracy_drop > 1.0:
        print("⚠️  WARNING: Large accuracy drop detected! Consider:")
        print("   - Increase calibration batches")
        print("   - Use per-channel quantization instead of per-tensor")
        print("   - Train longer with QAT before conversion")
    else:
        print("✓ QAT conversion successful (minimal accuracy loss)")

    return comparison


def get_quantization_config(backend: str = "qnnpack") -> tq.QConfig:
    """
    Get appropriate quantization config for backend.

    Args:
        backend: Quantization backend

    Returns:
        QConfig object for torch.quantization.prepare_qat
    """
    if backend == "qnnpack":
        return tq.get_default_qat_qconfig("qnnpack")
    elif backend == "fbgemm":
        return tq.get_default_qat_qconfig("fbgemm")
    elif backend == "x86":
        return tq.get_default_qat_qconfig("x86")
    else:
        return tq.get_default_qat_qconfig()


def print_quantization_summary(model: nn.Module):
    """Print summary of quantization layers in model"""
    qat_layers = 0
    quantized_weights = 0
    quantized_activations = 0

    for module in model.modules():
        if hasattr(module, 'weight_fake_quant'):
            qat_layers += 1
            quantized_weights += 1
        if hasattr(module, 'activation_post_process'):
            quantized_activations += 1

    print(f"\n=== Quantization Summary ===")
    print(f"QAT Layers: {qat_layers}")
    print(f"Quantized Weights: {quantized_weights}")
    print(f"Quantized Activations: {quantized_activations}")
    print(f"Total quantization points: {qat_layers + quantized_activations}")
