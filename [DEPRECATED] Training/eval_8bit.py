"""
Evaluation script for 8-bit QAT models.
Compares accuracy between float32 and int8 versions.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from data import get_transforms, GarbageDataset
from qat_utils import QATModelWrapper


class ModelEvaluator:
    """Evaluate and compare model performance"""

    def __init__(self, num_classes: int, device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["Recyclable", "Trash", "Empty"][:num_classes]

    def load_float32_model(self, checkpoint_path: str) -> nn.Module:
        """Load float32 model"""
        model = create_model("mobilenet_v3_small", num_classes=self.num_classes, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        return model

    def load_qat_model(self, checkpoint_path: str) -> nn.Module:
        """Load QAT model (before int8 conversion)"""
        model = create_model("mobilenet_v3_small", num_classes=self.num_classes, pretrained=False)
        model_qat = QATModelWrapper(model, backend="qnnpack")
        model_qat.prepare_qat()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" in checkpoint:
            model_qat.load_state_dict(checkpoint["model_state"])
        else:
            model_qat.load_state_dict(checkpoint)

        model_qat = model_qat.to(self.device)
        model_qat.eval()
        return model_qat

    def load_int8_model(self, checkpoint_path: str) -> nn.Module:
        """Load converted int8 model"""
        model = create_model("mobilenet_v3_small", num_classes=self.num_classes, pretrained=False)
        model_qat = QATModelWrapper(model, backend="qnnpack")
        model_qat.prepare_qat()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" in checkpoint:
            model_qat.load_state_dict(checkpoint["model_state"])
        else:
            model_qat.load_state_dict(checkpoint)

        # Convert to int8
        model_qat.convert_to_int8()

        model_qat = model_qat.to(self.device)
        model_qat.eval()
        return model_qat

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader,
                      model_name: str = "Model") -> dict:
        """Evaluate model and return metrics"""
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, predictions = torch.max(outputs.data, 1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
        avg_loss = total_loss / len(test_loader)

        # Per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[self.class_names[i]] = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "loss": avg_loss,
            "per_class_accuracy": per_class_acc,
            "preds": all_preds,
            "labels": all_labels,
        }

    def compare_models(self, results_list: list):
        """Compare multiple model results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        # Summary table
        print(f"\n{'Model':<20} {'Accuracy':<15} {'Loss':<15}")
        print("-"*80)
        for result in results_list:
            print(f"{result['model_name']:<20} {result['accuracy']:>6.2f}%  {result['loss']:>12.4f}")

        # Per-class accuracy
        print(f"\n{'Per-Class Accuracy':<80}")
        print("-"*80)
        print(f"{'Class':<20}", end="")
        for result in results_list:
            print(f"{result['model_name']:<20}", end="")
        print()

        for class_name in self.class_names:
            print(f"{class_name:<20}", end="")
            for result in results_list:
                if class_name in result['per_class_accuracy']:
                    acc = result['per_class_accuracy'][class_name]
                    print(f"{acc:>6.2f}%           ", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()

        # Accuracy differences
        print(f"\n{'Accuracy Difference (vs First Model)':<80}")
        print("-"*80)
        baseline_acc = results_list[0]['accuracy']
        for i, result in enumerate(results_list[1:], 1):
            diff = result['accuracy'] - baseline_acc
            sign = "+" if diff >= 0 else ""
            print(f"{result['model_name']:<20} {sign}{diff:>6.2f}%")

        print("="*80)

    def save_comparison_report(self, results_list: list, output_path: str = "evaluation_report_8bit.txt"):
        """Save comparison report to file"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("8-BIT QAT MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            for result in results_list:
                f.write(f"{result['model_name']}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.2f}%\n")
                f.write(f"  Loss: {result['loss']:.4f}\n")
                f.write(f"  Per-Class Accuracy:\n")
                for class_name, acc in result['per_class_accuracy'].items():
                    f.write(f"    {class_name}: {acc:.2f}%\n")
                f.write("\n")

            if len(results_list) > 1:
                f.write("COMPARISON\n")
                f.write("-"*80 + "\n")
                baseline_acc = results_list[0]['accuracy']
                for result in results_list[1:]:
                    diff = result['accuracy'] - baseline_acc
                    sign = "+" if diff >= 0 else ""
                    f.write(f"{result['model_name']}: {sign}{diff:.2f}%\n")

        print(f"\n✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 8-bit QAT Models")
    parser.add_argument("--checkpoint-float32", type=str, default=None,
                        help="Float32 checkpoint path")
    parser.add_argument("--checkpoint-qat", type=str, default=None,
                        help="QAT checkpoint path")
    parser.add_argument("--checkpoint-int8", type=str, default=None,
                        help="Int8 converted checkpoint path")
    parser.add_argument("--data-root", type=str, default="data/sorted_2_class",
                        help="Data directory")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--report", type=str, default="evaluation_report_8bit.txt",
                        help="Output report path")

    args = parser.parse_args()

    # Get device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Create data loader
    dataset = GarbageDataset(
        data_root=args.data_root,
        transform=get_transforms("val")
    )
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Test samples: {len(dataset)}\n")

    # Create evaluator
    evaluator = ModelEvaluator(num_classes=args.num_classes, device=device)

    results = []

    # Evaluate float32 model
    if args.checkpoint_float32:
        print("Evaluating float32 model...")
        model = evaluator.load_float32_model(args.checkpoint_float32)
        result = evaluator.evaluate_model(model, test_loader, "Float32 (Baseline)")
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2f}%\n")

    # Evaluate QAT model (before conversion)
    if args.checkpoint_qat:
        print("Evaluating QAT model (before int8 conversion)...")
        model = evaluator.load_qat_model(args.checkpoint_qat)
        result = evaluator.evaluate_model(model, test_loader, "QAT (Float32)")
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2f}%\n")

    # Evaluate int8 model (after conversion)
    if args.checkpoint_int8:
        print("Evaluating int8 model (after conversion)...")
        model = evaluator.load_int8_model(args.checkpoint_int8)
        result = evaluator.evaluate_model(model, test_loader, "Int8 (Converted)")
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2f}%\n")

    # Compare results
    if len(results) > 1:
        evaluator.compare_models(results)
        evaluator.save_comparison_report(results, args.report)
    elif len(results) == 1:
        print(f"\n{'='*80}")
        print("MODEL EVALUATION")
        print(f"{'='*80}")
        result = results[0]
        print(f"\n{result['model_name']}:")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Per-Class Accuracy:")
        for class_name, acc in result['per_class_accuracy'].items():
            print(f"    {class_name}: {acc:.2f}%")
        print(f"{'='*80}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
