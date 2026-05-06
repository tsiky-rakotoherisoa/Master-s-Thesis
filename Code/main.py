"""
A Comparative Analysis of OpenVINO and YOLO for Traffic Speed Estimation on Raspberry Pi

This is the main entry point for the thesis project. It provides a CLI to:
  1. Run real-time speed estimation using YOLOv3-Tiny + Farneback optical flow
  2. Run the OpenVINO-optimized counterpart
  3. Evaluate both approaches against ground truth
  4. Calibrate the camera (homography)

Usage examples:
    # Real-time detection with YOLO
    python main.py --model yolo

    # Process a video file with OpenVINO
    python main.py --model openvino --video data/videos/test.mp4

    # Calibrate camera from an image
    python main.py calibrate --image data/calibration/road.png

    # Run evaluation with ground truth
    python main.py --model yolo --video data/videos/test.mp4 --gt data/ground_truth.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import (
    YOLO_WEIGHTS_PATH, OPENVINO_MODEL_XML, OPENVINO_MODEL_BIN,
    CALIB_FILE, CAMERA_WIDTH, CAMERA_HEIGHT,
)
from scripts.run_experiment import main as run_experiment
from utils.calibration import CameraCalibrator, select_road_points


def calibrate(args):
    """Interactive camera calibration using homography."""
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        sys.exit(1)

    print("Select 4 points on the road plane in clockwise order (e.g., corners of a rectangle).")
    src_pts = select_road_points(image, num_points=4)
    print(f"Image points selected: {src_pts.tolist()}")

    print("\nEnter corresponding world coordinates (in metres).")
    print("Example: for a 10m x 5m rectangle:")
    dst_pts = []
    for i in range(4):
        x = float(input(f"  Point {i+1} world X (m): "))
        y = float(input(f"  Point {i+1} world Y (m): "))
        dst_pts.append([x, y])
    dst_pts = np.array(dst_pts, dtype=np.float32)

    calibrator = CameraCalibrator()
    H = calibrator.calibrate_from_points(image, src_pts, dst_pts)

    calib_path = args.output or CALIB_FILE
    calibrator.save(calib_path)
    print(f"Homography matrix saved to: {calib_path}")
    print(f"Matrix:\n{H}")

    # Optionally compute pixels-per-metre
    if len(src_pts) >= 2:
        known = float(input("\nKnown distance between points 1 and 2 (m): ") or "10")
        ppm = calibrator.calibrate_from_reference(known, tuple(src_pts[0]), tuple(src_pts[1]))
        print(f"Pixels per metre: {ppm:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Speed Estimation: YOLOv3-Tiny vs OpenVINO on Raspberry Pi"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run experiment subcommand
    run_parser = subparsers.add_parser("run", help="Run speed estimation experiment")
    run_parser.add_argument("--model", choices=["yolo", "openvino"], default="yolo")
    run_parser.add_argument("--video", type=str, default=None)
    run_parser.add_argument("--gt", type=str, default=None)
    run_parser.add_argument("--calib", type=str, default=None)
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.add_argument("--no-display", action="store_true")

    # Calibrate subcommand
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate camera")
    cal_parser.add_argument("--image", type=str, required=True)
    cal_parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "calibrate":
        calibrate(args)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
