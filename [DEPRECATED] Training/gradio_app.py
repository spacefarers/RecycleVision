"""Gradio-based inference UI for comparing float32 and quantized RecycleVision models."""
import os
import sys
import site
import time

# Set NNCASE_PLUGIN_PATH before importing nncase
# This allows nncase to find the k230 simulator executable
plugin_path = site.getsitepackages()[0]
os.environ['NNCASE_PLUGIN_PATH'] = plugin_path
os.environ['PATH'] = f"{plugin_path}:{os.environ.get('PATH', '')}"

import torch
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import nncase

from model import create_model

# Configuration
PYTORCH_CHECKPOINT = "checkpoints/finetune/final.pt"
ONNX_MODEL = "conversion_output/model_simple.onnx"  # Float32 ONNX model (clean export without uint8 preprocessing)
KMODEL_PATH = "models/recyclevision_qat.kmodel"  # Quantized kmodel (latest from Nov 10)
NUM_CLASSES = 3
CLASS_NAMES = ["recyclable", "trash", "empty"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
pytorch_model = create_model(NUM_CLASSES)
checkpoint = torch.load(PYTORCH_CHECKPOINT, map_location=DEVICE)
state_dict = checkpoint.get("model_state", checkpoint)
pytorch_model.load_state_dict(state_dict)
pytorch_model.to(DEVICE)
pytorch_model.eval()

# Load ONNX model (float32 baseline)
onnx_session = ort.InferenceSession(ONNX_MODEL)
onnx_input_name = onnx_session.get_inputs()[0].name

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

kmodel_sim = nncase.Simulator()
kmodel = read_model_file(KMODEL_PATH)
kmodel_sim.load_model(kmodel)

# Get input and output descriptors using the Simulator API
kmodel_input_desc = kmodel_sim.get_input_desc(0)
kmodel_output_desc = kmodel_sim.get_output_desc(0)

print("✓ kmodel loaded successfully")
print(f"  Input: dtype={kmodel_input_desc.dtype}, shape from tensor")
print(f"  Output: dtype={kmodel_output_desc.dtype}, shape from tensor")

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
    """Preprocess image for PyTorch model (float32, normalized)."""
    tensor = transform_pytorch(image).unsqueeze(0)
    return tensor

def preprocess_for_kmodel(image, input_desc):
    """Preprocess image for kmodel with proper dtype handling."""
    # Resize to 224x224
    image = image.resize((224, 224))

    # Get the expected input dtype from the model descriptor
    expected_dtype = input_desc.dtype

    # Convert to numpy array with proper dtype
    if expected_dtype == np.uint8:
        # For uint8, convert directly to uint8 (0-255 range)
        img_array = np.array(image, dtype=np.uint8)
    elif expected_dtype == np.int8:
        # For int8, convert to int8 with proper scaling
        img_array = np.array(image, dtype=np.float32)
        img_array = np.clip((img_array - 128), -128, 127).astype(np.int8)
    else:
        # For float32, keep as float32 normalized to [0, 1]
        img_array = np.array(image, dtype=np.float32)
        if np.max(img_array) > 1.0:  # If values are in 0-255 range
            img_array = img_array / 255.0

    # Keep channel order as RGB because the exported nncase model already
    # performs uint8->normalized preprocessing internally.

    # Convert to NCHW format: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure array is contiguous in memory (required for tensor conversion)
    img_array = np.ascontiguousarray(img_array)

    # Validate output shape and dtype
    print(f"[DEBUG] kmodel input shape: {img_array.shape}, dtype: {img_array.dtype}, contiguous: {img_array.flags['C_CONTIGUOUS']}")

    return img_array

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def classify_pytorch(image):
    """Classify using PyTorch float32 model."""
    start_time = time.time()

    tensor = preprocess_for_pytorch(image).to(DEVICE)

    with torch.no_grad():
        logits = pytorch_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    return probs, inference_time

def classify_onnx(image):
    """Classify using ONNX float32 model."""
    start_time = time.time()

    tensor = preprocess_for_pytorch(image).numpy()

    outputs = onnx_session.run(None, {onnx_input_name: tensor})
    logits = outputs[0]

    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    probs = probs.squeeze()

    inference_time = (time.time() - start_time) * 1000

    return probs, inference_time

def classify_kmodel(image):
    """Classify using quantized kmodel with proper Simulator API usage."""
    if kmodel_sim is None or kmodel_input_desc is None:
        print("[ERROR] kmodel simulator or input descriptor is None")
        return None, None

    start_time = time.time()

    try:
        # Preprocess with input descriptor to get proper dtype
        input_data = preprocess_for_kmodel(image, kmodel_input_desc)

        print(f"[DEBUG] Input descriptor dtype: {kmodel_input_desc.dtype}")
        print(f"[DEBUG] Prepared input shape: {input_data.shape}, dtype: {input_data.dtype}")

        # Set input tensor using Simulator API
        print("[DEBUG] Setting input tensor...")
        kmodel_sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_data))

        # Run inference using Simulator API
        print("[DEBUG] Running kmodel inference...")
        kmodel_sim.run()

        # Get output tensor using Simulator API
        print("[DEBUG] Getting output tensor...")
        output_tensor = kmodel_sim.get_output_tensor(0)
        output = output_tensor.to_numpy()

        print(f"[DEBUG] Output shape: {output.shape}, dtype: {output.dtype}")

        # Handle quantized output - convert to float32 for softmax if needed
        if output.dtype != np.float32:
            output = output.astype(np.float32)

        # Apply softmax to get probabilities
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()
        probs = probs.squeeze()

        print(f"[DEBUG] Final probabilities: {probs}")

        inference_time = (time.time() - start_time) * 1000

        return probs, inference_time

    except Exception as e:
        import traceback
        print(f"[ERROR] Error during kmodel inference: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return None, None

def compare_models(image):
    """Compare all three models and return results."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    image = image.convert("RGB")

    results = {}

    # PyTorch inference
    pytorch_probs, pytorch_time = classify_pytorch(image)
    results["PyTorch (Float32)"] = {
        name: float(pytorch_probs[idx]) for idx, name in enumerate(CLASS_NAMES)
    }
    pytorch_pred = CLASS_NAMES[np.argmax(pytorch_probs)]

    # ONNX inference
    onnx_probs, onnx_time = classify_onnx(image)
    results["ONNX (Float32)"] = {
        name: float(onnx_probs[idx]) for idx, name in enumerate(CLASS_NAMES)
    }
    onnx_pred = CLASS_NAMES[np.argmax(onnx_probs)]

    # Calculate similarity between PyTorch and ONNX
    pytorch_onnx_similarity = cosine_similarity(pytorch_probs, onnx_probs)

    # kmodel inference
    kmodel_probs = None
    kmodel_time = None
    kmodel_pred = "N/A"
    pytorch_kmodel_similarity = 0.0

    if kmodel_sim is not None:
        kmodel_probs, kmodel_time = classify_kmodel(image)
        if kmodel_probs is not None:
            results["kmodel (Quantized uint8)"] = {
                name: float(kmodel_probs[idx]) for idx, name in enumerate(CLASS_NAMES)
            }
            kmodel_pred = CLASS_NAMES[np.argmax(kmodel_probs)]

            # Calculate similarity with float32 baseline
            pytorch_kmodel_similarity = cosine_similarity(pytorch_probs, kmodel_probs)
        else:
            results["kmodel (Quantized uint8)"] = {}
            kmodel_time = None

    # Format timing information
    timing_info = f"""### ⏱️ Inference Time
    **PyTorch (Float32)**: {pytorch_time:.2f} ms
    **ONNX (Float32)**: {onnx_time:.2f} ms
    **kmodel (Quantized)**: {kmodel_time if kmodel_time else "N/A"}
    
    ### 📊 Accuracy Metrics
    **PyTorch ↔ ONNX Similarity**: {pytorch_onnx_similarity:.6f}
    **PyTorch ↔ kmodel Similarity**: {pytorch_kmodel_similarity:.6f}
    
    ### 🎯 Predictions
    **PyTorch**: {pytorch_pred}
    **ONNX**: {onnx_pred}
    **kmodel**: {kmodel_pred}
    """

    if kmodel_sim is not None and kmodel_probs is not None:
        # Calculate probability differences
        max_diff = np.max(np.abs(pytorch_probs - kmodel_probs))
        timing_info += f"\n**Max Probability Difference**: {max_diff:.6f}"

        # Speed comparison
        if kmodel_time and pytorch_time:
            speedup = pytorch_time / kmodel_time
            timing_info += f"\n**Speedup (PyTorch/kmodel)**: {speedup:.2f}x"
    elif kmodel_sim is not None:
        timing_info += "\n**kmodel**: Failed to run inference"

    return results["PyTorch (Float32)"], results["ONNX (Float32)"], results.get("kmodel (Quantized uint8)", {}), timing_info


# Create Gradio interface
with gr.Blocks(title="RecycleVision: Float32 vs Quantized Model Comparison") as demo:
    gr.Markdown("# 🔍 RecycleVision Model Comparison")
    gr.Markdown("Compare inference results between float32 PyTorch, float32 ONNX, and quantized kmodel.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload waste image")
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            pytorch_output = gr.Label(num_top_classes=NUM_CLASSES, label="PyTorch (Float32)")
            onnx_output = gr.Label(num_top_classes=NUM_CLASSES, label="ONNX (Float32)")
            kmodel_output = gr.Label(num_top_classes=NUM_CLASSES, label="kmodel (Quantized uint8)")

    with gr.Row():
        metrics_output = gr.Markdown(label="Performance Metrics")

    classify_btn.click(
        fn=compare_models,
        inputs=[image_input],
        outputs=[pytorch_output, onnx_output, kmodel_output, metrics_output]
    )

    gr.Markdown("""
    ### 📝 Notes
    - **Cosine Similarity**: Measures how similar the output probability distributions are (1.0 = identical)
    - **Typical quantization similarity**: >0.99 indicates good quantization quality
    - **kmodel** uses uint8 quantization which reduces model size and increases inference speed
    - Lower inference time is better (measured in milliseconds)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
