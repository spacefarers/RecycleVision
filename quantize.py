"""Quantize RecycleVision model for K230 deployment using nncase."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import torch

import nncase
from model import create_model


def replace_flatten_with_reshape(onnx_path: str | Path) -> None:
    """Replace Flatten nodes with Reshape nodes for K230 compatibility.

    Args:
        onnx_path: Path to the ONNX model to modify
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # Find Flatten nodes and replace with Reshape in-place
    modified = False
    for i, node in enumerate(graph.node):
        if node.op_type == "Flatten":
            print(f"Replacing Flatten node: {node.name}")
            modified = True

            # Add constant for reshape target shape [-1, features]
            # Flatten with axis=1 flattens to (batch, -1)
            shape_name = node.name + "_shape"
            shape_tensor = onnx.helper.make_tensor(
                shape_name,
                onnx.TensorProto.INT64,
                [2],
                [0, -1]  # 0 means keep batch size, -1 means infer
            )
            graph.initializer.append(shape_tensor)

            # Create a Reshape node instead
            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[node.input[0], shape_name],
                outputs=node.output,
                name=node.name + "_reshape"
            )

            # Replace the node in place
            graph.node[i].CopyFrom(reshape_node)

    if modified:
        # Save modified model
        onnx.save(model, onnx_path)
        print(f"Modified ONNX model saved to {onnx_path}")
    else:
        print("No Flatten nodes found")


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    onnx_path: str | Path,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (batch, channels, height, width)
        onnx_path: Path to save the ONNX model
    """
    model.eval()
    example_inputs = (torch.randn(*input_shape),)

    print(f"Exporting model to ONNX with input shape {input_shape}...")
    torch.onnx.export(
        model,
        example_inputs,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")


def generate_calibration_data(
    num_samples: int,
    input_shape: tuple[int, ...],
    data_loader: torch.utils.data.DataLoader | None = None
) -> list[list[np.ndarray]]:
    """Generate calibration data for PTQ quantization.

    Uses real data from the provided DataLoader if available, otherwise generates random data.

    Args:
        num_samples: Number of calibration samples
        input_shape: Input tensor shape (batch, channels, height, width)
        data_loader: Optional DataLoader to use for real calibration data

    Returns:
        List of calibration data in the format expected by nncase
    """
    print(f"Generating {num_samples} calibration samples...")
    calib_data = []

    if data_loader is not None:
        print("Using real validation data for calibration...")
        for batch_idx, (images, _) in enumerate(data_loader):
            if len(calib_data) >= num_samples:
                break
            # Convert to uint8 range [0, 255]
            # Images are normalized in [-1, 1] range, convert back
            images = (images * 255).byte().numpy()
            for img in images:
                if len(calib_data) >= num_samples:
                    break
                # Ensure shape matches input_shape
                if img.shape != input_shape[1:]:
                    # Handle potential shape mismatches
                    if len(img.shape) == 3:
                        img = img.reshape(input_shape[1:]) if img.size == np.prod(input_shape[1:]) else img
                calib_data.append(img)
    else:
        print("Using random calibration data (real data recommended for better accuracy)...")
        for i in range(num_samples):
            # Generate random data in the range [0, 255] as uint8
            sample = np.random.randint(0, 256, size=input_shape[1:], dtype=np.uint8)
            calib_data.append(sample)

    # nncase expects format: [[sample1, sample2, ...]]
    return [calib_data[:num_samples]]


def quantize_model(
    onnx_path: str | Path,
    kmodel_path: str | Path,
    dump_dir: str | Path,
    input_shape: tuple[int, ...],
    num_calib_samples: int = 10,
) -> None:
    """Quantize ONNX model to kmodel using nncase for K230 deployment.

    Args:
        onnx_path: Path to the ONNX model
        kmodel_path: Path to save the quantized kmodel
        dump_dir: Directory to dump intermediate compilation artifacts
        input_shape: Input tensor shape (batch, channels, height, width)
        num_calib_samples: Number of calibration samples for PTQ
    """
    print(f"\nQuantizing model with nncase...")

    # Setup compilation options
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = str(dump_dir)

    # Configure preprocessing
    compile_options.preprocess = True
    compile_options.input_type = "uint8"
    compile_options.input_shape = list(input_shape)
    compile_options.input_layout = "NCHW"
    compile_options.input_range = [0, 1]
    compile_options.swapRB = False

    # ImageNet normalization (same as training)
    compile_options.mean = [0.485, 0.456, 0.406]
    compile_options.std = [0.229, 0.224, 0.225]
    compile_options.output_layout = "NCHW"

    print(f"Compile options:")
    print(f"  Target: {compile_options.target}")
    print(f"  Input shape: {compile_options.input_shape}")
    print(f"  Input type: {compile_options.input_type}")
    print(f"  Preprocess: {compile_options.preprocess}")

    # Setup PTQ options
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.calibrate_method = "Kld"
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.samples_count = num_calib_samples

    # Generate calibration data
    calib_data = generate_calibration_data(num_calib_samples, input_shape)
    ptq_options.set_tensor_data(calib_data)

    print(f"\nPTQ options:")
    print(f"  Calibration method: {ptq_options.calibrate_method}")
    print(f"  Quantization type: {ptq_options.quant_type}")
    print(f"  Weight quant type: {ptq_options.w_quant_type}")
    print(f"  Calibration samples: {ptq_options.samples_count}")

    # Read ONNX model
    with open(onnx_path, "rb") as f:
        model_content = f.read()

    # Setup import options
    import_options = nncase.ImportOptions()

    # Compile model
    print("\nCompiling model...")
    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(model_content, import_options)
    compiler.use_ptq(ptq_options)
    compiler.compile()

    # Generate kmodel
    print("Generating kmodel...")
    kmodel = compiler.gencode_tobytes()

    # Save kmodel
    with open(kmodel_path, "wb") as f:
        f.write(kmodel)

    print(f"\nQuantized model saved to {kmodel_path}")
    print(f"Compilation artifacts saved to {dump_dir}")


def main() -> None:
    """Main quantization pipeline."""
    # Configuration
    checkpoint_path = Path("checkpoints/finetune/final.pt")
    onnx_path = Path("models/recyclevision.onnx")
    kmodel_path = Path("models/recyclevision.kmodel")
    dump_dir = Path("models/dump")

    num_classes = 3  # recyclable, trash, empty
    input_shape = (1, 3, 224, 224)  # NCHW format

    # Create output directories
    onnx_path.parent.mkdir(exist_ok=True)
    dump_dir.mkdir(exist_ok=True)

    # Load trained model
    print(f"Loading model from {checkpoint_path}...")
    model = create_model(num_classes=num_classes, pretrained=False)

    # Load checkpoint - handle both direct state_dict and checkpoint dict formats
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # Export to ONNX
    export_to_onnx(model, input_shape, onnx_path)

    # Replace Flatten with Reshape for K230 compatibility
    print("\nReplacing Flatten nodes with Reshape...")
    replace_flatten_with_reshape(onnx_path)

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

    # Quantize to kmodel
    quantize_model(
        onnx_path=onnx_path,
        kmodel_path=kmodel_path,
        dump_dir=dump_dir,
        input_shape=input_shape,
        num_calib_samples=10,
    )

    print("\n" + "="*60)
    print("Quantization complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"kmodel: {kmodel_path}")
    print(f"Artifacts: {dump_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
