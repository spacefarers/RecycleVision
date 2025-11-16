#!/usr/bin/env python3
"""
Test all model versions to identify which one is correct.
"""

import os
import sys
import site
import numpy as np
import torch
from PIL import Image
from pathlib import Path

plugin_path = site.getsitepackages()[0]
os.environ['NNCASE_PLUGIN_PATH'] = plugin_path
os.environ['PATH'] = f"{plugin_path}:{os.environ.get('PATH', '')}"

import nncase
from torchvision import transforms
from model import create_model

class_names = ['recyclable', 'trash', 'empty']

# Load PyTorch model (reference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

print("=" * 80)
print("LOADING PYTORCH MODEL (Reference)")
print("=" * 80)

checkpoint = torch.load("checkpoints/finetune/final.pt", map_location=device, weights_only=False)
pytorch_model = create_model(num_classes=3, pretrained=False)
pytorch_model.load_state_dict(checkpoint["model_state"], strict=False)
pytorch_model.to(device)
pytorch_model.eval()

# Test image
image_path = sorted(Path("data/collected").glob("*.png"))[0]
image = Image.open(image_path).convert("RGB")

# PyTorch inference
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output_pytorch = pytorch_model(input_tensor)
    probs_pytorch = torch.softmax(output_pytorch, dim=1).squeeze().cpu().numpy()

pred_pytorch = class_names[np.argmax(probs_pytorch)]
print(f"\nPyTorch Result:")
print(f"  Logits: {output_pytorch.squeeze().cpu().numpy()}")
print(f"  Probabilities: {probs_pytorch}")
print(f"  Prediction: {pred_pytorch} ({probs_pytorch.max():.4f})")

# Test kmodel versions
def test_kmodel(model_path, model_name):
    print(f"\n{'=' * 80}")
    print(f"TESTING: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 80)

    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        return

    try:
        def read_model_file(f):
            with open(f, 'rb') as file:
                return file.read()

        kmodel_sim = nncase.Simulator()
        kmodel_sim.load_model(read_model_file(model_path))

        input_desc = kmodel_sim.get_input_desc(0)
        output_desc = kmodel_sim.get_output_desc(0)

        print(f"\nModel Info:")
        print(f"  Input dtype: {input_desc.dtype}")
        print(f"  Output dtype: {output_desc.dtype}")

        # Prepare input: uint8, RGB, NCHW
        img_np = np.array(image.resize((224, 224)), dtype=np.uint8)
        img_np = np.transpose(img_np, (2, 0, 1))  # RGB, NCHW
        img_np = np.expand_dims(img_np, 0)  # Add batch

        print(f"\nInput shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Value range: [{img_np.min()}, {img_np.max()}]")

        kmodel_sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(img_np))
        kmodel_sim.run()
        output_tensor = kmodel_sim.get_output_tensor(0)
        output_kmodel = output_tensor.to_numpy().squeeze()

        print(f"\nOutput:")
        print(f"  Raw output: {output_kmodel}")
        print(f"  Output dtype: {output_kmodel.dtype}")

        # Try different softmax approaches
        # Approach 1: Direct softmax
        try:
            probs_1 = np.exp(output_kmodel - np.max(output_kmodel))
            probs_1 = probs_1 / np.sum(probs_1)
            pred_1 = class_names[np.argmax(probs_1)]
            print(f"\nApproach 1 (Direct softmax):")
            print(f"  Probabilities: {probs_1}")
            print(f"  Prediction: {pred_1}")
        except:
            print("  (Failed)")

        # Approach 2: As-is (raw output treated as logits)
        try:
            probs_2 = np.abs(output_kmodel)
            probs_2 = probs_2 / np.sum(probs_2)
            pred_2 = class_names[np.argmax(probs_2)]
            print(f"\nApproach 2 (Abs values, normalized):")
            print(f"  Probabilities: {probs_2}")
            print(f"  Prediction: {pred_2}")
        except:
            print("  (Failed)")

        # Approach 3: Clamp to positive and normalize
        try:
            probs_3 = np.maximum(output_kmodel, 0)
            probs_3 = probs_3 / np.sum(probs_3)
            pred_3 = class_names[np.argmax(probs_3)]
            print(f"\nApproach 3 (Clamp positive):")
            print(f"  Probabilities: {probs_3}")
            print(f"  Prediction: {pred_3}")
        except:
            print("  (Failed)")

        # Check match with PyTorch
        match_1 = np.argmax(probs_1) == np.argmax(probs_pytorch) if 'probs_1' in locals() else False
        match_3 = np.argmax(probs_3) == np.argmax(probs_pytorch) if 'probs_3' in locals() else False

        if match_1:
            print(f"\n✓ Approach 1 MATCHES PyTorch")
        elif match_3:
            print(f"\n✓ Approach 3 MATCHES PyTorch")
        else:
            print(f"\n✗ MISMATCH with PyTorch prediction ({pred_pytorch})")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Test all kmodel versions
test_kmodel("models/recyclevision.kmodel", "Old recyclevision.kmodel")
test_kmodel("models/recyclevision_noquant.kmodel", "recyclevision_noquant.kmodel (no quantization)")
test_kmodel("models/recyclevision_qat.kmodel", "NEW recyclevision_qat.kmodel (QAT quantized)")

print("\n" * 2)
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"PyTorch reference prediction: {pred_pytorch}")
print("\nRecommendation: Update gradio_app.py to use the model that matches PyTorch")
