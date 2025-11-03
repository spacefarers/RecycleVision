from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

from config import TwoStageConfig, DataConfig, default_twostage_config
from data import _build_transforms
from data_mapping import MappedImageFolder
from model import create_model
from utils import load_config


def _prepare_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    lam = Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)
    indices = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[indices]
    shuffled_targets = targets[indices]
    return mixed_images, shuffled_targets, lam


def _build_dataloaders(data_root: Path, num_classes: int, batch_size: int = 64, num_workers: int = 4):
    data_config = DataConfig(root=data_root, image_size=224, batch_size=batch_size, num_workers=num_workers, augment=True)
    train_transform, val_transform = _build_transforms(data_config)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if train_dir.exists():
        train_dataset = MappedImageFolder(train_dir, transform=train_transform, num_classes=num_classes)
    else:
        train_dataset = MappedImageFolder(data_root, transform=train_transform, num_classes=num_classes)

    if val_dir.exists():
        val_dataset = MappedImageFolder(val_dir, transform=val_transform, num_classes=num_classes)
    else:
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create weighted sampler for balanced class sampling during training
    targets = train_dataset.targets if hasattr(train_dataset, 'targets') else [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = torch.zeros(num_classes)
    for target in targets:
        class_counts[target] += 1

    # Inverse frequency weights - classes with fewer samples get higher weight
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([class_weights[t].item() for t in targets])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(targets), replacement=True)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    return train_loader, val_loader


def _train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, device_type, amp_enabled, mixup_alpha):
    model.train()
    train_loss = 0.0
    correct = 0.0
    data_points = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        targets = labels.to(device)
        
        lam = None
        shuffled_targets = None
        if mixup_alpha > 0 and images.size(0) > 1:
            images, shuffled_targets, lam = _mixup_batch(images, targets, mixup_alpha)
        
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            if images.size(0) == 1:
                model.eval()
            outputs = model(images)
            if lam is not None:
                loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, shuffled_targets)
            else:
                loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            if images.size(0) == 1:
                model.train()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item() * images.size(0)
        if lam is not None:
            correct += lam * (preds == targets).sum().item() + (1 - lam) * (preds == shuffled_targets).sum().item()
        else:
            correct += (preds == targets).sum().item()
        data_points += images.size(0)
    
    return train_loss / data_points, correct / data_points


def _evaluate(model, val_loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = 0.0
    correct = 0
    data_points = 0
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            data_points += images.size(0)
            
            for c in range(num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c] += mask.sum().item()
    
    avg_loss = total_loss / data_points
    avg_acc = correct / data_points
    
    class_names = ['recyclable', 'trash', 'empty']
    print(f"  Per-class acc: ", end="")
    for c in range(num_classes):
        if per_class_total[c] > 0:
            acc = per_class_correct[c] / per_class_total[c]
            print(f"{class_names[c]}={acc:.1%} ", end="")
    print()
    
    return avg_loss, avg_acc


def _run_training(stage_name, data_root, epochs, learning_rate, checkpoint_dir, num_classes, config, pretrained_weights=None, freeze_backbone=False, freeze_epochs=0):
    print(f"\n{'='*80}")
    print(f"{stage_name}")
    print(f"{'='*80}")
    
    device = _prepare_device(config.training.device)
    # Keep ImageNet weights unless we explicitly load a checkpoint below.
    model = create_model(num_classes)
    
    if pretrained_weights and pretrained_weights.exists():
        print(f"Loading pretrained weights from {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
    
    model.to(device)
    
    if freeze_backbone:
        print(f"Freezing backbone for {freeze_epochs} epochs")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    batch_size = getattr(config, 'batch_size', 16)
    train_loader, val_loader = _build_dataloaders(data_root, num_classes, batch_size=batch_size)
    
    # Calculate class weights for imbalanced datasets
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.3)
    
    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_enabled = bool(config.training.mixed_precision and device_type in {"cuda", "mps"})
    scaler = torch.amp.GradScaler(device=device_type, enabled=amp_enabled)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    patience_counter = 0
    best_checkpoint_path = None
    
    print(f"Dataset: {data_root}")
    print(f"Epochs: {epochs}, LR: {learning_rate}, Device: {device}\n")
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"{stage_name} Epochs"):
        if freeze_backbone and epoch == freeze_epochs + 1:
            print(f"\nUnfreezing backbone at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=learning_rate * 0.1)
            remaining_steps = len(train_loader) * (epochs - epoch + 1)
            scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 0.1, total_steps=remaining_steps, pct_start=0.3)
        
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, device_type, amp_enabled, config.training.mixup_alpha)
        print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.3%}")
        
        if epoch % config.training.eval_interval_epochs == 0:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device, num_classes)
            print(f"Epoch {epoch}: Val loss: {val_loss:.4f}, Val acc: {val_acc:.3%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_checkpoint_path = checkpoint_dir / f"best-epoch{epoch}.pt"
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc, "num_classes": num_classes}, best_checkpoint_path)
                print(f"Saved checkpoint to {best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    final_path = checkpoint_dir / "final.pt"
    torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, final_path)
    print(f"\n{stage_name} complete. Best val acc: {best_acc:.3%}")
    
    return best_checkpoint_path or final_path


def train(config: TwoStageConfig):
    pretrained_weights = None
    
    if config.pretrain.enabled:
        pretrained_weights = _run_training(
            "STAGE 1: PRE-TRAINING",
            config.pretrain.data_root,
            config.pretrain.epochs,
            config.pretrain.learning_rate,
            config.pretrain.checkpoint_dir,
            config.num_classes,
            config
        )
    
    if config.finetune.enabled:
        if config.finetune.pretrained_weights:
            pretrained_weights = config.finetune.pretrained_weights
        
        _run_training(
            "STAGE 2: FINE-TUNING",
            config.finetune.data_root,
            config.finetune.epochs,
            config.finetune.learning_rate,
            config.finetune.checkpoint_dir,
            config.num_classes,
            config,
            pretrained_weights,
            config.finetune.freeze_backbone,
            config.finetune.freeze_epochs
        )


def parse_args():
    parser = argparse.ArgumentParser(description="RecycleVision Training")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
