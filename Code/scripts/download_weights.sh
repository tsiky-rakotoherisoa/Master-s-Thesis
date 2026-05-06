#!/bin/bash
# Download pre-trained YOLOv3-Tiny weights and convert to required formats.
# Usage: bash scripts/download_weights.sh

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading YOLOv3-Tiny PyTorch weights..."
wget -c "https://github.com/ultralytics/yolov3/releases/download/v9.5/yolov3-tiny.pt" \
  -O "$WEIGHTS_DIR/yolov3-tiny.pt"

echo "Done."
echo ""
echo "Next steps:"
echo "  1. Convert to ONNX:"
echo "     python scripts/convert_to_openvino.py --weights weights/yolov3-tiny.pt"
echo ""
echo "  2. The OpenVINO IR files will be saved to $WEIGHTS_DIR/"
