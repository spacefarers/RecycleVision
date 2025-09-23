"""Training loop implementation."""
from __future__ import annotations

import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ExperimentConfig
from data import build_loaders
from model import create_model


def _prepare_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _mixup_batch(
        images: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply mixup and return mixed images, shuffled targets, and lambda."""
    lam = Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)
    indices = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[indices]
    shuffled_targets = targets[indices]
    return mixed_images, shuffled_targets, lam


def train(config: ExperimentConfig) -> None:
    """Execute the full training loop."""
    device = _prepare_device(config.training.device)
    model = create_model(config.num_classes)
    model.to(device)

    train_loader, val_loader = build_loaders(config.data)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.optimizer.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    total_steps = len(train_loader) * config.training.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.optimizer.learning_rate,
        total_steps=total_steps,
        pct_start=0.3
    )

    # Configure AMP with device-aware API
    device_type = (
        "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    )
    amp_enabled = bool(config.training.mixed_precision and device_type in {"cuda", "mps"})
    scaler = torch.amp.GradScaler(device=device_type, enabled=amp_enabled)

    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    patience_counter = 0

    print("Starting training with configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    # Top-level epoch bar
    epoch_bar = tqdm(range(1, config.training.epochs + 1), desc="Epochs", position=0, leave=True)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        correct = 0.0
        data_points = 0
        avg_loss = 0.0
        acc = 0.0
        epoch_start = time.time()
        train_start = epoch_start

        # Batch-level bar nested under epoch
        batch_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.training.epochs}", position=1, leave=False)
        for step, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            targets = labels.to(device)
            lam = None
            shuffled_targets = None
            mixup_alpha = config.training.mixup_alpha
            if mixup_alpha > 0 and images.size(0) > 1:
                images, shuffled_targets, lam = _mixup_batch(images, targets, mixup_alpha)

            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                outputs = model(images)
                if lam is not None and shuffled_targets is not None:
                    loss = (
                        lam * criterion(outputs, targets)
                        + (1 - lam) * criterion(outputs, shuffled_targets)
                    )
                else:
                    loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            if lam is not None and shuffled_targets is not None:
                correct += (
                    lam * (preds == targets).sum().item()
                    + (1 - lam) * (preds == shuffled_targets).sum().item()
                )
            else:
                correct += (preds == targets).sum().item()
            data_points += images.size(0)

            if data_points > 0:
                avg_loss = train_loss / data_points
                acc = correct / data_points
                batch_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3%}"})
            batch_bar.update(1)
        batch_bar.close()
        train_duration = time.time() - train_start
        print(f"Train loss: {avg_loss:.4f}, Train acc: {acc:.3%}")

        # Evaluate only on configured epoch intervals
        should_eval = (epoch % max(1, config.training.eval_interval_epochs)) == 0
        checkpoint_duration = None
        if should_eval:
            eval_start = time.time()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            eval_duration = time.time() - eval_start
            print(f"Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.3%}")
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                checkpoint_path = config.training.checkpoint_dir / f"best-epoch{epoch}.pt"
                checkpoint_start = time.time()
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, checkpoint_path)
                checkpoint_duration = time.time() - checkpoint_start
                print(f"Saved checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
        else:
            eval_duration = None

        if config.training.log_timing:
            epoch_time = time.time() - epoch_start
            timing_msg = f"Timing: epoch={epoch_time:.1f}s train={train_duration:.1f}s"
            if eval_duration is not None:
                timing_msg += f" eval={eval_duration:.1f}s"
            if checkpoint_duration is not None:
                timing_msg += f" checkpoint={checkpoint_duration:.1f}s"
            print(timing_msg)

    final_path = config.training.checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final weights saved to {final_path}")


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> tuple[float, float]:
    """Run evaluation over the validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    data_points = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            data_points += images.size(0)

    avg_loss = total_loss / data_points
    accuracy = correct / data_points
    return avg_loss, accuracy
