# Traffic Speed Estimation on Raspberry Pi: YOLOv3-Tiny vs OpenVINO

This repository contains the full implementation for the MSc thesis:

**"A Comparative Analysis of OpenVINO and YOLO for Traffic Speed Estimation on Raspberry Pi"**  
_by Tsiky Tafita RAKOTOHERISOA, AIMS Rwanda, 2025._

## Overview

Two computer-vision approaches for real-time vehicle speed estimation are compared:

| Approach | Detection | Tracking | Speed |
|----------|-----------|----------|-------|
| YOLOv3-Tiny (PyTorch) | YOLOv3-Tiny | Farneback optical flow + SORT | 46.22 FPS |
| OpenVINO-optimized | YOLOv3-Tiny (INT8) | Farneback optical flow + SORT | 134.71 FPS |

Both run on a **Raspberry Pi 4 Model B** with a USB camera.

## Project Structure

```
Code/
├── main.py                          # CLI entry point
├── config.py                        # All configuration parameters
├── requirements.txt                 # Python dependencies
├── README.md
├── weights/                         # Model weights (download separately)
│   ├── yolov3-tiny.pt
│   ├── yolov3-tiny-openvino.xml
│   └── yolov3-tiny-openvino.bin
├── models/
│   ├── __init__.py
│   ├── yolo_tiny.py                 # YOLOv3-Tiny architecture (PyTorch)
│   └── openvino_model.py            # OpenVINO inference wrapper
├── detection/
│   ├── __init__.py
│   └── detector.py                  # YOLODetector & OpenVINODetector
├── tracking/
│   ├── __init__.py
│   └── optical_flow.py              # Farneback optical flow + SORT tracker
├── speed_estimation/
│   ├── __init__.py
│   └── estimator.py                 # Speed calculation from optical flow
├── utils/
│   ├── __init__.py
│   ├── calibration.py               # Camera calibration (homography)
│   ├── metrics.py                   # MAE, RMSE, MAPE, R², IoU, mAP
│   └── visualization.py             # Drawing helpers
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py                  # Benchmark & evaluation runner
├── scripts/
│   ├── download_weights.sh          # Download pre-trained weights
│   ├── convert_to_openvino.py       # PyTorch → ONNX → OpenVINO IR
│   └── run_experiment.py            # Full experiment pipeline
└── data/
    ├── videos/                      # Input video files
    ├── calibration/                 # Homography matrices
    └── ground_truth.csv             # Ground truth speed data (GPS logger)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

```bash
bash scripts/download_weights.sh
```

### 3. (Optional) Convert to OpenVINO IR

```bash
python scripts/convert_to_openvino.py --weights weights/yolov3-tiny.pt
```

### 4. Calibrate the camera

```bash
python main.py calibrate --image data/calibration/road_sample.png
```

Follow the interactive prompts to select four points on the road plane and enter their real-world coordinates.

## Usage

### Run with YOLOv3-Tiny (PyTorch baseline)

```bash
python main.py run --model yolo --video data/videos/test.mp4
```

### Run with OpenVINO-optimized model

```bash
python main.py run --model openvino --video data/videos/test.mp4
```

### Real-time with USB camera

```bash
python main.py run --model yolo
```

### With ground-truth evaluation

```bash
python main.py run --model yolo --video data/videos/test.mp4 --gt data/ground_truth.csv
```

## Key Results

| Metric | YOLOv3-Tiny | OpenVINO Optimized |
|--------|-------------|-------------------|
| **FPS** | 46.22 | 134.71 |
| **MAE (km/h)** | 0.78 | 2.48 |
| **RMSE (km/h)** | 1.17 | 3.18 |
| **MAPE (%)** | 16.9 | 50.4 |
| **R² (%)** | 98.12 | 86.14 |
| **IoU (%)** | 57.88 | 40.83 |
| **mAP (%)** | 64.59 | 54.74 |
| **Memory (%)** | 53.20 | 32.05 |
| **CPU (%)** | 30.01 | 53.50 |

## Hardware

- **Raspberry Pi 4 Model B** (4 GB RAM, Quad-core Cortex-A72)
- **USB Camera** (1080p @ 30 FPS, 2MP, 5–50 mm optical zoom)
- **Test vehicle**: Toyota Land Cruiser V8 (reference speeds 30–80 km/h)

## Citation

```bibtex
@mastersthesis{Rakotoherisoa2025,
  author  = {Tsiky Tafita RAKOTOHERISOA},
  title   = {A Comparative Analysis of OpenVINO and YOLO for
             Traffic Speed Estimation on Raspberry Pi},
  school  = {African Institute for Mathematical Sciences (AIMS), Rwanda},
  year    = {2025},
}
```

## License

This project is made available for academic and research purposes.
