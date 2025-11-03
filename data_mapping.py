"""Dataset utilities for mapping multi-class datasets to 3 classes."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


# Mapping from common waste categories to our 3 classes
# recyclable (0), trash (1), empty (2)
CLASS_MAPPINGS = {
    # For garbage-classification dataset (6 classes -> 3 classes)
    "cardboard": 0,    # recyclable
    "glass": 0,        # recyclable
    "metal": 0,        # recyclable
    "paper": 0,        # recyclable
    "plastic": 0,      # recyclable
    "trash": 1,        # trash
    
    # For our sorted_2_class dataset (already 3 classes)
    "recyclable": 0,
    "empty": 2,
}


class MappedImageFolder(Dataset):
    """ImageFolder that maps original classes to target classes.
    
    This allows using datasets with different class structures for pre-training.
    """
    
    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        class_mapping: dict[str, int] | None = None,
        num_classes: int = 3,
    ):
        """
        Args:
            root: Root directory with class subdirectories
            transform: Image transforms to apply
            class_mapping: Dict mapping class names to target class indices
            num_classes: Number of target classes
        """
        self.root = Path(root)
        self.transform = transform
        self.num_classes = num_classes
        
        # Load base ImageFolder dataset
        self.base_dataset = datasets.ImageFolder(root)
        
        # Build class mapping
        if class_mapping is None:
            class_mapping = CLASS_MAPPINGS
        
        # Create index mapping from original indices to target indices
        self.index_mapping = {}
        for orig_idx, class_name in enumerate(self.base_dataset.classes):
            target_idx = class_mapping.get(class_name.lower())
            if target_idx is None:
                # Try to infer: if class name contains recyclable materials
                class_lower = class_name.lower()
                if any(mat in class_lower for mat in ["cardboard", "glass", "metal", "paper", "plastic"]):
                    target_idx = 0  # recyclable
                elif "empty" in class_lower:
                    target_idx = 2  # empty
                else:
                    target_idx = 1  # default to trash
            self.index_mapping[orig_idx] = target_idx
        
        print(f"Class mapping for {root}:")
        for orig_idx, class_name in enumerate(self.base_dataset.classes):
            target_idx = self.index_mapping[orig_idx]
            target_names = ["recyclable", "trash", "empty"]
            print(f"  {class_name} -> {target_names[target_idx]} ({target_idx})")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, original_label = self.base_dataset[idx]
        mapped_label = self.index_mapping[original_label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, mapped_label
    
    @property
    def classes(self):
        """Return target class names."""
        return ["recyclable", "trash", "empty"][:self.num_classes]
