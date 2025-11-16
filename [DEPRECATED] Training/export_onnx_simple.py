#!/usr/bin/env python3
"""Simple ONNX export without problematic preprocessing wrappers."""
import torch
from pathlib import Path
from model import create_model

def export_simple_onnx():
    """Export model directly without uint8->normalized wrapper."""
    checkpoint_path = Path("checkpoints/finetune/final.pt")
    output_path = Path("conversion_output/model_simple.onnx")

    num_classes = 3
    device = torch.device("cpu")

    # Load model
    print("Loading model...")
    model = create_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded")

    # Create dummy input (normalized float32)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print("✓ ONNX export complete")

    # Verify it loads
    print("Verifying ONNX model...")
    import onnxruntime as ort
    try:
        session = ort.InferenceSession(str(output_path))
        print("✓ ONNX model verified and loads successfully")
        return True
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False

if __name__ == "__main__":
    export_simple_onnx()
