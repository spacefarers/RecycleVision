"""
Process garbage-classification dataset to match sorted_2_class structure.
Maps 6 classes (cardboard, glass, metal, paper, plastic, trash) to 2 classes (recyclable, trash).
"""
import os
import shutil
from pathlib import Path

def process_garbage_classification_dataset(
    source_dir="/Users/spacefarers/code/RecycleVision/data/garbage-classification",
    target_dir="/Users/spacefarers/code/RecycleVision/data/garbage-classification-2class"
):
    """
    Process garbage-classification dataset and create 2-class version.

    Mapping:
    - cardboard, glass, metal, paper, plastic -> recyclable
    - trash -> trash
    """

    # Define class mapping
    recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    trash_classes = ['trash']

    # Create target directory structure
    target_recyclable = os.path.join(target_dir, 'recyclable')
    target_trash = os.path.join(target_dir, 'trash')

    os.makedirs(target_recyclable, exist_ok=True)
    os.makedirs(target_trash, exist_ok=True)

    print(f"Processing dataset from {source_dir}")
    print(f"Target directory: {target_dir}")

    total_files = 0
    recyclable_count = 0
    trash_count = 0

    # Process both train and val directories
    for split in ['train', 'val']:
        split_dir = os.path.join(source_dir, split)

        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found, skipping...")
            continue

        print(f"\nProcessing {split} split...")

        # Process recyclable classes
        for class_name in recyclable_classes:
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"  Warning: {class_dir} not found, skipping...")
                continue

            # Get all image files
            image_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"  {class_name}: {len(image_files)} images -> recyclable")

            # Copy files to recyclable folder with prefix to avoid name collisions
            for img_file in image_files:
                source_path = os.path.join(class_dir, img_file)
                # Add prefix to maintain source info
                target_filename = f"{split}_{class_name}_{img_file}"
                target_path = os.path.join(target_recyclable, target_filename)

                shutil.copy2(source_path, target_path)
                recyclable_count += 1
                total_files += 1

        # Process trash class
        for class_name in trash_classes:
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"  Warning: {class_dir} not found, skipping...")
                continue

            image_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"  {class_name}: {len(image_files)} images -> trash")

            # Copy files to trash folder
            for img_file in image_files:
                source_path = os.path.join(class_dir, img_file)
                target_filename = f"{split}_{class_name}_{img_file}"
                target_path = os.path.join(target_trash, target_filename)

                shutil.copy2(source_path, target_path)
                trash_count += 1
                total_files += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files processed: {total_files}")
    print(f"  - recyclable: {recyclable_count} images")
    print(f"  - trash: {trash_count} images")
    print(f"Target directory: {target_dir}")
    print(f"{'='*60}")

    return target_dir


if __name__ == "__main__":
    result_dir = process_garbage_classification_dataset()
    print(f"\nDataset ready at: {result_dir}")