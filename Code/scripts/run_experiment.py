"""
Run a full speed-estimation experiment using either YOLOv3-Tiny (PyTorch)
or the OpenVINO-optimized version, with Farneback optical flow tracking.

Usage:
    # YOLO baseline
    python scripts/run_experiment.py --model yolo --video data/videos/test.mp4

    # OpenVINO optimized
    python scripts/run_experiment.py --model openvino --video data/videos/test.mp4

    # With ground-truth evaluation
    python scripts/run_experiment.py --model yolo --video data/videos/test.mp4 --gt data/ground_truth.csv
"""

import argparse
import csv
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import (
    CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    YOLO_WEIGHTS_PATH, OPENVINO_MODEL_XML, OPENVINO_MODEL_BIN,
    CALIB_FILE, FRAME_SKIP, OUTPUT_DIR, PLOTS_DIR, RESULTS_FILE,
    GROUND_TRUTH_FILE,
)
from detection.detector import YOLODetector, OpenVINODetector
from tracking.optical_flow import FarnebackOpticalFlow, SORTTracker
from speed_estimation.estimator import SpeedEstimator
from evaluation.evaluate import Evaluator, Benchmark
from utils.visualization import (
    draw_tracks, draw_optical_flow, draw_fps, draw_info_panel
)
from utils.calibration import CameraCalibrator


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Speed Estimation Experiment")
    parser.add_argument("--model", choices=["yolo", "openvino"], default="yolo",
                        help="Detection model to use")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (uses camera if not provided)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to ground-truth CSV (timestamp, speed)")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to homography calibration .npy file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without display window")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup output dirs ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Video source ──────────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS

    # ── Detector ─────────────────────────────────────────────────────────
    if args.model == "yolo":
        print(f"Initializing YOLOv3-Tiny detector (weights: {YOLO_WEIGHTS_PATH})...")
        detector = YOLODetector(YOLO_WEIGHTS_PATH)
        model_name = "YOLOv3-Tiny (PyTorch)"
    else:
        print(f"Initializing OpenVINO detector ({OPENVINO_MODEL_XML})...")
        detector = OpenVINODetector(OPENVINO_MODEL_XML, OPENVINO_MODEL_BIN)
        model_name = "Optimized YOLOv3-Tiny (OpenVINO)"

    # ── Tracker & Optical Flow ───────────────────────────────────────────
    tracker = SORTTracker()
    flow_engine = FarnebackOpticalFlow()

    # ── Speed Estimator ──────────────────────────────────────────────────
    estimator = SpeedEstimator(camera_fps=fps_in)

    calib_path = args.calib or CALIB_FILE
    if calib_path and Path(calib_path).exists():
        print(f"Loading calibration from {calib_path}...")
        estimator.load_homography(calib_path)
    else:
        print("No calibration file found. Using pixel-based speed (requires pixels_per_metre).")

    # ── Evaluator ────────────────────────────────────────────────────────
    gt_path = args.gt or GROUND_TRUTH_FILE
    evaluator = Evaluator(gt_path if Path(gt_path).exists() else None)
    benchmark = Benchmark()
    all_predictions = []

    # ── Output video ─────────────────────────────────────────────────────
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps_in,
                                     (CAMERA_WIDTH, CAMERA_HEIGHT))

    # ── Resource tracking ────────────────────────────────────────────────
    cpu_samples, mem_samples = [], []
    frame_count = 0
    proc_start = time.perf_counter()

    print(f"\nRunning {model_name}...")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        curr_time = time.perf_counter()
        timestamp = frame_count / fps_in

        # Skip frames if configured
        if frame_count % (FRAME_SKIP + 1) != 0:
            continue

        # ── Detect vehicles ──────────────────────────────────────────────
        boxes, scores, class_ids = detector.detect(frame)

        # Filter for vehicle classes (car, truck, bus, motorcycle → IDs 2,7,5,3)
        vehicle_boxes = []
        for i, cls_id in enumerate(class_ids):
            if cls_id in {2, 3, 5, 7}:
                vehicle_boxes.append(boxes[i])

        # ── Track ────────────────────────────────────────────────────────
        tracks = tracker.update(vehicle_boxes)

        # ── Optical Flow ─────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = flow_engine.compute(gray)

        # ── Speed estimation ─────────────────────────────────────────────
        speeds = {}
        for track_id, bbox in tracks:
            dx, dy = flow_engine.get_vehicle_displacement(flow, bbox)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            speed = estimator.estimate_from_centroid(track_id, (cx, cy), timestamp)
            if speed is not None:
                speeds[track_id] = speed
                all_predictions.append((timestamp, speed))

        # ── Resource monitoring ──────────────────────────────────────────
        cpu_usage, mem_usage = benchmark.measure_resources()
        cpu_samples.append(cpu_usage)
        mem_samples.append(mem_usage)

        # ── Visualization ────────────────────────────────────────────────
        if not args.no_display or args.output:
            display = frame.copy()
            display = draw_tracks(display, [(tid, b, speeds.get(tid)) for tid, b in tracks])
            display = draw_optical_flow(display, flow, step=32)

            elapsed = time.perf_counter() - proc_start
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            display = draw_fps(display, current_fps)
            display = draw_info_panel(display, model_name, cpu_usage, mem_usage)

            if out_writer:
                out_writer.write(display)

            if not args.no_display:
                cv2.imshow("Speed Estimation", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    # ── Report ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - proc_start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0
    avg_mem = float(np.mean(mem_samples)) if mem_samples else 0

    print(f"\n{'='*50}")
    print(f"  Experiment Summary: {model_name}")
    print(f"{'='*50}")
    print(f"  Total frames processed : {frame_count}")
    print(f"  Total time (s)         : {total_time:.2f}")
    print(f"  Average FPS            : {avg_fps:.2f}")
    print(f"  Avg CPU usage (%)      : {avg_cpu:.2f}")
    print(f"  Avg Memory usage (%)   : {avg_mem:.2f}")

    # ── Evaluation ───────────────────────────────────────────────────────
    metrics = evaluator.evaluate_speed_estimates(all_predictions)
    Evaluator.print_results(metrics, model_name)

    # ── Save results ─────────────────────────────────────────────────────
    results = [(ts, evaluator.ground_truth.get(ts, ""), pred, model_name)
               for ts, pred in all_predictions]
    Evaluator.save_results(results, RESULTS_FILE)
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
