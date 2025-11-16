#!/usr/bin/env python3
"""
Test multiple images to find where models disagree.
"""

import os
import sys
import site
from pathlib import Path

plugin_path = site.getsitepackages()[0]
os.environ['NNCASE_PLUGIN_PATH'] = plugin_path
os.environ['PATH'] = f"{plugin_path}:{os.environ.get('PATH', '')}"

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import nncase

from model import create_model

PYTORCH_CHECKPOINT = "checkpoints/finetune/final.pt"
KMODEL_PATH = "models/recyclevision.kmodel"
NUM_CLASSES = 3
CLASS_NAMES = ["recyclable", "trash", "empty"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup models
pytorch_model = create_model(NUM_CLASSES)
checkpoint = torch.load(PYTORCH_CHECKPOINT, map_location=DEVICE)
pytorch_model.load_state_dict(checkpoint.get("model_state", checkpoint))
pytorch_model.to(DEVICE)
pytorch_model.eval()

def read_model_file(f):
    with open(f, 'rb') as file:
        return file.read()

kmodel_sim = nncase.Simulator()
kmodel_sim.load_model(read_model_file(KMODEL_PATH))
kmodel_input_desc = kmodel_sim.get_input_desc(0)

transform_pytorch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_for_pytorch(image):
    return transform_pytorch(image).unsqueeze(0)

def preprocess_for_kmodel(image, input_desc):
    image = image.resize((224, 224))
    expected_dtype = input_desc.dtype
    if expected_dtype == np.uint8:
        img_array = np.array(image, dtype=np.uint8)
    elif expected_dtype == np.int8:
        img_array = np.array(image, dtype=np.float32)
        img_array = np.clip((img_array - 128), -128, 127).astype(np.int8)
    else:
        img_array = np.array(image, dtype=np.float32)
        if np.max(img_array) > 1.0:
            img_array = img_array / 255.0

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return np.ascontiguousarray(img_array)

def classify_pytorch(image):
    tensor = preprocess_for_pytorch(image).to(DEVICE)
    with torch.no_grad():
        logits = pytorch_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    return np.argmax(probs), probs.max()

def classify_kmodel(image):
    try:
        input_data = preprocess_for_kmodel(image, kmodel_input_desc)
        kmodel_sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_data))
        kmodel_sim.run()
        output_tensor = kmodel_sim.get_output_tensor(0)
        output = output_tensor.to_numpy()
        if output.dtype != np.float32:
            output = output.astype(np.float32)
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()
        probs = probs.squeeze()
        return np.argmax(probs), probs.max()
    except:
        return -1, 0.0

# Test all images
print("=" * 80)
print("Testing all collected images...")
print("=" * 80)

image_files = sorted(Path("data/collected").glob("*.png"))
print(f"Found {len(image_files)} images\n")

disagreements = []
pytorch_vs_kmodel_count = 0

for image_path in tqdm(image_files, desc="Processing images"):
    image = Image.open(image_path).convert("RGB")

    pytorch_class, pytorch_conf = classify_pytorch(image)
    kmodel_class, kmodel_conf = classify_kmodel(image)

    pytorch_pred = CLASS_NAMES[pytorch_class]
    kmodel_pred = CLASS_NAMES[kmodel_class] if kmodel_class >= 0 else "ERROR"

    match_pk = pytorch_class == kmodel_class if kmodel_class >= 0 else False

    if not match_pk:
        disagreements.append({
            'file': image_path.name,
            'pytorch': pytorch_pred,
            'kmodel': kmodel_pred,
            'pytorch_conf': pytorch_conf,
            'kmodel_conf': kmodel_conf,
        })
        pytorch_vs_kmodel_count += 1

        tqdm.write(f"\n❌ DISAGREEMENT #{pytorch_vs_kmodel_count}: {image_path.name}")
        tqdm.write(f"   PyTorch: {pytorch_pred:12s} (conf: {pytorch_conf:.4f})")
        tqdm.write(f"   kModel:  {kmodel_pred:12s} (conf: {kmodel_conf:.4f})")

if not disagreements:
    print("✓ All images: Models agree on all predictions!")
else:
    print(f"\n{'=' * 80}")
    print(f"DISAGREEMENTS FOUND: {len(disagreements)} out of {len(image_files)} images")
    print(f"{'=' * 80}")
    print(f"PyTorch vs kmodel mismatches: {pytorch_vs_kmodel_count}")

    print(f"\nDetails (first 10):")
    for d in disagreements[:10]:
        print(f"\n{d['file']}:")
        print(f"  PyTorch: {d['pytorch']} ({d['pytorch_conf']:.4f})")
        print(f"  kmodel:  {d['kmodel']} ({d['kmodel_conf']:.4f})")
