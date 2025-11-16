"""
Benchmark script to compare training performance:
- Float32 vs 8-bit QAT
- Training speed
- Memory usage
- Accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import psutil
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from data import get_transforms, GarbageDataset
from qat_utils import QATModelWrapper


class TrainingBenchmark:
    """Benchmark different training configurations"""

    def __init__(self, num_classes: int = 3, batch_size: int = 16):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data_loaders(self, data_root: str = "data/sorted_2_class"):
        """Load data"""
        dataset = GarbageDataset(
            data_root=data_root,
            transform=get_transforms("train")
        )

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = get_transforms("val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return train_loader, val_loader

    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_mem = 0

        return {
            "cpu_mb": mem_info.rss / 1024 / 1024,
            "gpu_mb": gpu_mem,
        }

    def benchmark_float32(self, train_loader: DataLoader, val_loader: DataLoader,
                         epochs: int = 3):
        """Benchmark standard float32 training"""
        print("\n" + "="*80)
        print("BENCHMARK: Float32 Training")
        print("="*80)

        model = create_model("mobilenet_v3_small", num_classes=self.num_classes, pretrained=True)
        model = model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        mem_start = self.get_memory_usage()
        time_start = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_time = time.time()

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predictions = torch.max(outputs.data, 1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            epoch_time = time.time() - epoch_time
            accuracy = 100.0 * correct / total
            print(f"  Epoch {epoch + 1}: Loss={epoch_loss/len(train_loader):.4f}, "
                  f"Acc={accuracy:.2f}%, Time={epoch_time:.2f}s")

        total_time = time.time() - time_start
        mem_end = self.get_memory_usage()

        results = {
            "config": "Float32",
            "total_time": total_time,
            "cpu_memory_mb": mem_start["cpu_mb"],
            "gpu_memory_mb": mem_start["gpu_mb"],
            "max_gpu_memory_mb": mem_end["gpu_mb"],
        }

        print(f"\nResults:")
        print(f"  Total Time: {total_time:.2f}s ({epochs} epochs)")
        print(f"  Avg Time/Epoch: {total_time/epochs:.2f}s")
        print(f"  GPU Memory: {mem_start['gpu_mb']:.0f} MB → {mem_end['gpu_mb']:.0f} MB")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return results

    def benchmark_8bit_qat(self, train_loader: DataLoader, val_loader: DataLoader,
                          epochs: int = 3):
        """Benchmark 8-bit QAT training"""
        print("\n" + "="*80)
        print("BENCHMARK: 8-Bit QAT Training (with Mixed Precision)")
        print("="*80)

        model = create_model("mobilenet_v3_small", num_classes=self.num_classes, pretrained=True)
        model_qat = QATModelWrapper(model, backend="qnnpack")
        model_qat.prepare_qat()
        model_qat = model_qat.to(self.device)

        # Try to use 8-bit AdamW if available
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(model_qat.parameters(), lr=5e-4, weight_decay=1e-4, block_wise=True)
            optimizer_type = "8-bit AdamW"
        except ImportError:
            optimizer = optim.AdamW(model_qat.parameters(), lr=5e-4, weight_decay=1e-4)
            optimizer_type = "Standard AdamW"

        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        mem_start = self.get_memory_usage()
        time_start = time.time()

        print(f"  Using optimizer: {optimizer_type}")

        for epoch in range(epochs):
            model_qat.train()
            epoch_loss = 0
            epoch_time = time.time()

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model_qat(images)
                    loss = criterion(outputs, labels)

                # Backward with loss scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_qat.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            # Validate
            model_qat.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = model_qat(images)
                    _, predictions = torch.max(outputs.data, 1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            epoch_time = time.time() - epoch_time
            accuracy = 100.0 * correct / total
            print(f"  Epoch {epoch + 1}: Loss={epoch_loss/len(train_loader):.4f}, "
                  f"Acc={accuracy:.2f}%, Time={epoch_time:.2f}s")

        total_time = time.time() - time_start
        mem_end = self.get_memory_usage()

        results = {
            "config": "8-Bit QAT + Mixed Precision",
            "total_time": total_time,
            "cpu_memory_mb": mem_start["cpu_mb"],
            "gpu_memory_mb": mem_start["gpu_mb"],
            "max_gpu_memory_mb": mem_end["gpu_mb"],
        }

        print(f"\nResults:")
        print(f"  Total Time: {total_time:.2f}s ({epochs} epochs)")
        print(f"  Avg Time/Epoch: {total_time/epochs:.2f}s")
        print(f"  GPU Memory: {mem_start['gpu_mb']:.0f} MB → {mem_end['gpu_mb']:.0f} MB")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return results

    def compare_results(self, float32_results: dict, qat_results: dict):
        """Compare benchmark results"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        speedup = float32_results["total_time"] / qat_results["total_time"]
        memory_reduction = (float32_results["max_gpu_memory_mb"] - qat_results["max_gpu_memory_mb"]) / float32_results["max_gpu_memory_mb"] * 100

        print(f"\n{'Metric':<30} {'Float32':<20} {'8-Bit QAT':<20}")
        print("-"*70)
        print(f"{'Total Training Time':<30} {float32_results['total_time']:.2f}s{'':<12} {qat_results['total_time']:.2f}s")
        print(f"{'Time per Epoch':<30} {float32_results['total_time']/3:.2f}s{'':<12} {qat_results['total_time']/3:.2f}s")
        print(f"{'Peak GPU Memory':<30} {float32_results['max_gpu_memory_mb']:.0f} MB{'':<12} {qat_results['max_gpu_memory_mb']:.0f} MB")

        print(f"\n{'IMPROVEMENTS':<30}")
        print("-"*70)
        print(f"{'Speedup:':<30} {speedup:.2f}x faster")
        print(f"{'Memory Reduction:':<30} {memory_reduction:.1f}% less GPU memory")
        print(f"{'Estimated Time for 50 epochs:':<30}")
        print(f"  Float32: {float32_results['total_time']/3 * 50:.0f}s ({float32_results['total_time']/3 * 50/60:.1f} min)")
        print(f"  8-Bit QAT: {qat_results['total_time']/3 * 50:.0f}s ({qat_results['total_time']/3 * 50/60:.1f} min)")

        print("\n" + "="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark 8-Bit Training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs to benchmark (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--data-root", type=str, default="data/sorted_2_class",
                        help="Data directory")
    parser.add_argument("--float32-only", action="store_true",
                        help="Only benchmark float32")
    parser.add_argument("--qat-only", action="store_true",
                        help="Only benchmark 8-bit QAT")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("8-BIT TRAINING PERFORMANCE BENCHMARK")
    print(f"{'='*80}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Benchmark Epochs: {args.epochs}")
    print(f"{'='*80}")

    benchmark = TrainingBenchmark(batch_size=args.batch_size)
    train_loader, val_loader = benchmark.get_data_loaders(args.data_root)

    results = []

    if not args.qat_only:
        float32_results = benchmark.benchmark_float32(train_loader, val_loader, args.epochs)
        results.append(float32_results)

    if not args.float32_only:
        qat_results = benchmark.benchmark_8bit_qat(train_loader, val_loader, args.epochs)
        results.append(qat_results)

    if len(results) == 2:
        benchmark.compare_results(results[0], results[1])
    else:
        print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
