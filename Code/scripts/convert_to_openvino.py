"""
Convert YOLOv3-Tiny PyTorch weights to OpenVINO Intermediate Representation.

Usage:
    python scripts/convert_to_openvino.py --weights weights/yolov3-tiny.pt

Requirements:
    pip install torch onnx openvino
"""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.yolo_tiny import YOLOv3Tiny


def export_to_onnx(model, onnx_path, input_size=(416, 416)):
    dummy = torch.randn(1, 3, *input_size)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output_13", "output_26"],
        dynamic_axes={"input": {0: "batch"}, "output_13": {0: "batch"}, "output_26": {0: "batch"}},
        opset_version=11,
    )
    print(f"ONNX model saved to: {onnx_path}")


def convert_to_openvino(onnx_path, output_dir):
    from openvino import convert_model, save_model

    model = convert_model(onnx_path)
    xml_path = Path(output_dir) / "yolov3-tiny-openvino.xml"
    bin_path = Path(output_dir) / "yolov3-tiny-openvino.bin"
    save_model(model, str(xml_path), compress_to_fp16=False)
    print(f"OpenVINO IR saved to:\n  {xml_path}\n  {bin_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv3-Tiny to OpenVINO IR")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--output", type=str, default="weights", help="Output directory")
    parser.add_argument("--input-size", type=int, nargs=2, default=(416, 416))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv3-Tiny model...")
    model = YOLOv3Tiny(num_classes=80)
    model.eval()

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    print("Weights loaded successfully.")

    onnx_path = output_dir / "yolov3-tiny.onnx"
    export_to_onnx(model, onnx_path, args.input_size)

    try:
        convert_to_openvino(onnx_path, output_dir)
    except ImportError:
        print("OpenVINO not installed. Skipping IR conversion.")
        print("Install with: pip install openvino")
        print(f"ONNX model available at: {onnx_path}")


if __name__ == "__main__":
    main()
