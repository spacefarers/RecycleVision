"""Export RecycleVision model to ONNX format (without K230 quantization)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch

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


def main() -> None:
    """Main export pipeline."""
    # Configuration
    checkpoint_path = Path("checkpoints/finetune/final.pt")
    onnx_path = Path("models/recyclevision.onnx")

    num_classes = 3  # recyclable, trash, empty
    input_shape = (1, 3, 224, 224)  # NCHW format

    # Create output directories
    onnx_path.parent.mkdir(exist_ok=True)

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

    print("\n" + "="*60)
    print("ONNX export complete!")
    print(f"ONNX model: {onnx_path}")
    print("="*60)
    print("\nTo compile for K230, run quantize.py on a Linux system with nncase installed.")


if __name__ == "__main__":
    main()
