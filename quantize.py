"""Convert PyTorch RecycleVision model to kmodel format."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def create_model(num_classes: int, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """Create a MobileNetV3-Small classification model with enhanced classifier."""
    model = mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_rate),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_rate),
        nn.Linear(256, num_classes),
    )

    # Initialize classifier layers with proper weights
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model


def generate_calibration_data(input_shape: tuple, num_samples: int) -> list[list[np.ndarray]]:
    """Generate random calibration data for PTQ quantization.

    The calibration data must be float32 to match the ONNX model's expected input type.
    Data is normalized to [0, 1] range as expected by ImageNet-pretrained models.
    """
    calibration_data = []
    for _ in range(num_samples):
        # Generate random image data as float32 in [0, 1] range
        # This matches the normalized input expected by the model
        sample = np.random.uniform(0, 1, size=input_shape).astype(np.float32)
        calibration_data.append([sample])

    return calibration_data


def export_to_onnx(model: nn.Module, input_shape: tuple, output_path: str) -> None:
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape[1:])

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,  # nncase supports opset 11
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed input shape
    )
    print("✓ ONNX export complete")


def simplify_onnx(input_path: str, output_path: str) -> None:
    """Simplify ONNX model using onnxsim."""
    print(f"Simplifying ONNX model...")

    onnx_model = onnx.load(input_path)
    onnx_model, check = onnxsim.simplify(onnx_model)

    if not check:
        print("⚠ Warning: Simplified model may not be valid")

    # Remove unsupported attributes from Reshape operations
    print("Removing unsupported Reshape attributes...")
    for node in onnx_model.graph.node:
        if node.op_type == "Reshape":
            # Remove allowzero attribute if it exists
            attrs_to_remove = []
            for i, attr in enumerate(node.attribute):
                if attr.name == "allowzero":
                    attrs_to_remove.append(i)

            # Remove in reverse order to maintain indices
            for i in reversed(attrs_to_remove):
                del node.attribute[i]

    onnx.save(onnx_model, output_path)
    print("✓ ONNX simplification complete")


def convert_to_kmodel(
    onnx_path: str,
    output_path: str,
    input_shape: tuple,
    target: str = "k230",
    num_calibration_samples: int = 10,
) -> None:
    """Convert ONNX model to kmodel using nncase."""
    print(f"Converting to kmodel for target: {target}")

    try:
        import nncase
    except ImportError:
        raise ImportError(
            "nncase is not installed. Install with: pip install nncase"
        )

    # Configure compilation options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_type = "uint8"  # Use uint8 for image input
    compile_options.input_shape = list(input_shape)
    compile_options.input_layout = "NCHW"
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = "output"

    # Create compiler
    compiler = nncase.Compiler(compile_options)

    # Import ONNX model
    print("Importing ONNX model...")
    with open(onnx_path, "rb") as f:
        model_content = f.read()

    import_options = nncase.ImportOptions()
    compiler.import_onnx(model_content, import_options)

    # Configure PTQ quantization options
    print(f"Configuring PTQ with {num_calibration_samples} calibration samples...")
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = num_calibration_samples
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.calibrate_method = "Kld"  # Kullback-Leibler divergence
    ptq_options.finetune_weights_method = "NoFineTuneWeights"

    # Generate and set calibration data
    print("Generating calibration data...")
    calibration_data = generate_calibration_data(input_shape, num_calibration_samples)
    ptq_options.set_tensor_data(calibration_data)

    # Apply PTQ
    compiler.use_ptq(ptq_options)

    # Compile
    print("Compiling model...")
    compiler.compile()

    # Generate kmodel
    print(f"Generating kmodel: {output_path}")
    kmodel = compiler.gencode_tobytes()
    with open(output_path, "wb") as f:
        f.write(kmodel)

    print("✓ kmodel conversion complete")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to kmodel")
    parser.add_argument(
        "--input",
        type=str,
        default="checkpoints/finetune/final.pt",
        help="Path to input .pt model file (default: checkpoints/finetune/final.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/recyclevision.kmodel",
        help="Output kmodel filename (default: model.kmodel)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="k230",
        choices=["k210", "k510", "k230"],
        help="Target device (default: k230)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes (default: 3)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=10,
        help="Number of calibration samples for PTQ (default: 10)",
    )

    args = parser.parse_args()

    # Define paths
    input_path = Path(args.input)
    output_dir = Path("conversion_output")
    output_dir.mkdir(exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    simplified_onnx_path = output_dir / "model_simplified.onnx"
    kmodel_path = Path(args.output)

    # Define input shape (batch_size, channels, height, width)
    input_shape = (1, 3, args.input_size, args.input_size)

    # Load checkpoint first to get num_classes if available
    checkpoint = torch.load(input_path, map_location="cpu")
    num_classes = args.num_classes

    if isinstance(checkpoint, dict) and "num_classes" in checkpoint:
        num_classes = checkpoint["num_classes"]
        print(f"Using num_classes from checkpoint: {num_classes}")

    print("=" * 60)
    print("PyTorch to kmodel Conversion Pipeline")
    print("=" * 60)
    print(f"Input model: {input_path}")
    print(f"Output kmodel: {kmodel_path}")
    print(f"Target device: {args.target}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print("=" * 60)

    # Step 1: Load PyTorch model
    print("\n[Step 1/4] Loading PyTorch model...")
    model = create_model(num_classes=num_classes, pretrained=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ Model loaded successfully")

    # Step 2: Export to ONNX
    print("\n[Step 2/4] Converting to ONNX...")
    export_to_onnx(model, input_shape, str(onnx_path))

    # Step 3: Simplify ONNX
    print("\n[Step 3/4] Simplifying ONNX model...")
    simplify_onnx(str(onnx_path), str(simplified_onnx_path))

    # Step 4: Convert to kmodel
    print("\n[Step 4/4] Converting to kmodel...")
    convert_to_kmodel(
        str(simplified_onnx_path),
        str(kmodel_path),
        input_shape,
        target=args.target,
        num_calibration_samples=args.calibration_samples,
    )

    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print(f"Output saved to: {kmodel_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
