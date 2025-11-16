"""
Pretraining script for garbage classification.
Trains on the larger garbage-classification-2class dataset before fine-tuning on sorted_2_class.
"""
import os
import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18
from scripts.dataset import CustomDataset
from scripts import splitdata
from utils.read_yaml import read_yaml
from collections import Counter


def pretrain_pipeline(config):
    """
    Pretrain ResNet18 on garbage-classification-2class dataset.
    """
    # Get pretrain-specific config
    pretrain_config = config.get("pretrain", {})
    root_folder = pretrain_config.get("root_folder", "../data/garbage-classification-2class")
    split = pretrain_config.get("split", True)
    train_ratio = pretrain_config.get("train_ratio", 0.8)
    val_ratio = pretrain_config.get("val_ratio", 0.2)
    test_ratio = pretrain_config.get("test_ratio", 0.0)

    # Get train config
    train_config = config.get("train", {})
    device = train_config.get("device", "cpu")
    txt_path = pretrain_config.get("txt_path", '../gen/pretrain')
    image_size = train_config.get("image_size", [224, 224])
    mean = train_config.get("mean", [0.485, 0.456, 0.406])
    std = train_config.get("std", [0.229, 0.224, 0.225])
    epochs = pretrain_config.get("epochs", 30)
    batchsize = train_config.get("batchsize", 8)
    learningrate = pretrain_config.get("learningrate", 0.001)
    save_path = pretrain_config.get("save_path", '../checkpoints/pretrain')

    # Create directories
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Split dataset
    if split:
        print(f"Splitting pretrain dataset from {root_folder}")
        splitdata.split_dataset(root_folder, train_ratio, val_ratio, test_ratio, txt_path)

    train_txt = os.path.join(txt_path, "train.txt")
    val_txt = os.path.join(txt_path, "val.txt")
    labels_txt = os.path.join(txt_path, "labels.txt")

    # Data preprocessing with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Minimal transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Initialize datasets
    train_dataset = CustomDataset(txt_file=train_txt, root_folder=root_folder, transform=train_transform)
    val_dataset = CustomDataset(txt_file=val_txt, root_folder=root_folder, transform=val_transform)

    # Calculate class weights for balanced sampling
    train_labels = [label for _, label in train_dataset.data]
    class_counts = Counter(train_labels)
    print(f"Pretrain class distribution: {dict(class_counts)}")

    # Calculate weights for each sample (inverse of class frequency)
    total_samples = len(train_labels)
    class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]

    # Create WeightedRandomSampler for balanced training
    sampler = WeightedRandomSampler(sample_weights, num_samples=total_samples, replacement=True)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batchsize, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    # Get class names
    class_names = []
    with open(labels_txt, 'r') as file:
        for line in file:
            class_name = line.strip()
            class_names.append(class_name)

    num_classes = len(class_names)
    print(f"Pretraining on {num_classes} classes: {class_names}")

    # Initialize model (from scratch, no ImageNet weights)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10

    print(f"\n{'='*60}")
    print(f"Starting pretraining on {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            if i % 10 == 0:
                print(f"[Pretrain] Epoch {epoch + 1}/{epochs}, Batch {i + 1}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / batch_count

        # Validate model
        model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_accuracy = correct_predictions / total_samples
        scheduler.step(val_accuracy)

        print(f"[Pretrain] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "pretrain_best.pth"))
            print(f"  -> New best model saved! (Val Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"[Pretrain] Early stopping after {epoch + 1} epochs (no improvement for {patience_limit} epochs)")
            break

    print("\n[Pretrain] Finished pretraining!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save last model
    torch.save(model.state_dict(), os.path.join(save_path, "pretrain_last.pth"))
    print(f"[Pretrain] Models saved to {save_path}")

    return os.path.join(save_path, "pretrain_best.pth")


def main(config):
    pretrain_weights_path = pretrain_pipeline(config)
    print(f"\n[Pretrain] Pretrained weights ready at: {pretrain_weights_path}")
    return pretrain_weights_path


if __name__ == "__main__":
    config_file_path = "../yaml/config.yaml"
    config = read_yaml(config_file_path)
    main(config)