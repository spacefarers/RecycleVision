#!/usr/bin/env python3
"""
Run the exact comparison from gradio_app.py to see what it outputs.
"""

import os
import sys
import site
import time

# Set NNCASE_PLUGIN_PATH before importing nncase
plugin_path = site.getsitepackages()[0]
os.environ['NNCASE_PLUGIN_PATH'] = plugin_path
os.environ['PATH'] = f"{plugin_path}:{os.environ.get('PATH', '')}"

import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from pathlib import Path
import nncase

from model import create_model

# Configuration (from gradio_app.py)
PYTORCH_CHECKPOINT = "checkpoints/finetune/final.pt"
ONNX_MODEL = "models/recyclevision.onnx"
KMODEL_PATH = "models/recyclevision.kmodel"
NUM_CLASSES = 3
CLASS_NAMES = ["recyclable", "trash", "empty"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("Testing Exact Gradio App Flow")
print("=" * 80)

# Load PyTorch model (from gradio_app.py)
print("\n1. Loading PyTorch model...")
pytorch_model = create_model(NUM_CLASSES)
checkpoint = torch.load(PYTORCH_CHECKPOINT, map_location=DEVICE)
state_dict = checkpoint.get("model_state", checkpoint)
pytorch_model.load_state_dict(state_dict)
pytorch_model.to(DEVICE)
pytorch_model.eval()
print("   ✓ Loaded")

# Load ONNX model
print("2. Loading ONNX model...")
onnx_session = ort.InferenceSession(ONNX_MODEL)
onnx_input_name = onnx_session.get_inputs()[0].name
print("   ✓ Loaded")

# Load kmodel
print("3. Loading kmodel...")
def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        return f.read()

kmodel_sim = nncase.Simulator()
kmodel = read_model_file(KMODEL_PATH)
kmodel_sim.load_model(kmodel)
kmodel_input_desc = kmodel_sim.get_input_desc(0)
kmodel_output_desc = kmodel_sim.get_output_desc(0)
print("   ✓ Loaded")

# Image preprocessing
transform_pytorch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_for_pytorch(image):
    tensor = transform_pytorch(image).unsqueeze(0)
    return tensor

def preprocess_for_kmodel(image, input_desc):
    """Exact from gradio_app.py"""
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
    img_array = np.ascontiguousarray(img_array)

    return img_array

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def classify_pytorch(image):
    tensor = preprocess_for_pytorch(image).to(DEVICE)
    with torch.no_grad():
        logits = pytorch_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    return probs

def classify_onnx(image):
    tensor = preprocess_for_pytorch(image).numpy()
    outputs = onnx_session.run(None, {onnx_input_name: tensor})
    logits = outputs[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    probs = probs.squeeze()
    return probs

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

        return probs
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load test image
print("\n4. Loading test image...")
image_path = sorted(Path("data/collected").glob("*.png"))[0]
image = Image.open(image_path).convert("RGB")
print(f"   Using: {image_path.name}")

# Run inference
print("\n" + "=" * 80)
print("INFERENCE RESULTS")
print("=" * 80)

pytorch_probs = classify_pytorch(image)
pytorch_pred = CLASS_NAMES[np.argmax(pytorch_probs)]
print(f"\nPyTorch:")
print(f"  Probabilities: {pytorch_probs}")
print(f"  Prediction: {pytorch_pred} ({pytorch_probs.max():.6f})")

onnx_probs = classify_onnx(image)
onnx_pred = CLASS_NAMES[np.argmax(onnx_probs)]
pytorch_onnx_sim = cosine_similarity(pytorch_probs, onnx_probs)
print(f"\nONNX:")
print(f"  Probabilities: {onnx_probs}")
print(f"  Prediction: {onnx_pred} ({onnx_probs.max():.6f})")
print(f"  PyTorch ↔ ONNX Similarity: {pytorch_onnx_sim:.6f}")
print(f"  Predictions Match: {'✓' if pytorch_pred == onnx_pred else '✗'}")

kmodel_probs = classify_kmodel(image)
if kmodel_probs is not None:
    kmodel_pred = CLASS_NAMES[np.argmax(kmodel_probs)]
    pytorch_kmodel_sim = cosine_similarity(pytorch_probs, kmodel_probs)
    print(f"\nkmodel:")
    print(f"  Probabilities: {kmodel_probs}")
    print(f"  Prediction: {kmodel_pred} ({kmodel_probs.max():.6f})")
    print(f"  PyTorch ↔ kmodel Similarity: {pytorch_kmodel_sim:.6f}")
    print(f"  Predictions Match: {'✓' if pytorch_pred == kmodel_pred else '✗'}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if pytorch_pred != onnx_pred or (kmodel_probs is not None and pytorch_pred != kmodel_pred):
    print("\n⚠️  DISAGREEMENT DETECTED!")
    if pytorch_pred != onnx_pred:
        print(f"  PyTorch ({pytorch_pred}) vs ONNX ({onnx_pred})")
    if kmodel_probs is not None and pytorch_pred != kmodel_pred:
        print(f"  PyTorch ({pytorch_pred}) vs kmodel ({kmodel_pred})")

    print("\nPossible causes:")
    print("  1. Different model checkpoints used (final.pt vs the one used for ONNX/kmodel)")
    print("  2. ONNX export didn't preserve model weights")
    print("  3. kmodel compilation introduced errors")
else:
    print("\n✓ All models agree!")
