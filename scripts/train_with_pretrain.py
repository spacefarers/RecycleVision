"""
Full training pipeline with pretraining.
1. Pretrains ResNet18 on garbage-classification-2class dataset (2,527 images, 2 classes)
2. Fine-tunes on sorted_2_class dataset (89 images, 3 classes with 'empty')
"""
import os
import sys

sys.path.append('../')
from utils.read_yaml import read_yaml
from scripts.pretrain import pretrain_pipeline
from scripts.main import pipeline


def full_training_pipeline(config_path="../yaml/config.yaml"):
    """
    Execute full training pipeline with pretraining.

    Steps:
    1. Load configuration
    2. Run pretraining on garbage-classification-2class
    3. Run fine-tuning on sorted_2_class with pretrained weights
    """
    print("\n" + "="*80)
    print("FULL TRAINING PIPELINE WITH PRETRAINING")
    print("="*80 + "\n")

    # Load config
    config = read_yaml(config_path)

    # Check if pretraining should be run
    train_config = config.get("train", {})
    use_pretrained = train_config.get("use_pretrained", False)

    if use_pretrained:
        pretrained_weights = train_config.get("pretrained_weights", "../checkpoints/pretrain/pretrain_best.pth")

        # Check if pretrained weights already exist
        if os.path.exists(pretrained_weights):
            print(f"Pretrained weights already exist at {pretrained_weights}")
            print("Skipping pretraining phase. Delete the weights file to retrain.\n")
        else:
            print("="*80)
            print("PHASE 1: PRETRAINING")
            print("="*80)
            print("Dataset: garbage-classification-2class (2,527 images)")
            print("Classes: recyclable, trash")
            print("="*80 + "\n")

            # Run pretraining
            pretrained_weights_path = pretrain_pipeline(config)

            print("\n" + "="*80)
            print(f"PRETRAINING COMPLETE!")
            print(f"Weights saved to: {pretrained_weights_path}")
            print("="*80 + "\n")

    # Run fine-tuning
    print("="*80)
    print("PHASE 2: FINE-TUNING")
    print("="*80)
    print("Dataset: sorted_2_class (89 images)")
    print("Classes: empty, recyclable, trash")
    if use_pretrained:
        print(f"Using pretrained weights: {pretrained_weights}")
    else:
        print("Training from scratch (no pretraining)")
    print("="*80 + "\n")

    # Run main training pipeline
    pipeline(config)

    print("\n" + "="*80)
    print("FULL TRAINING PIPELINE COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    full_training_pipeline()
