import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
CALIB_DIR = os.path.join(DATA_DIR, "calibration")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── YOLOv3-Tiny (PyTorch) ──────────────────────────────────────────────────
YOLO_INPUT_SIZE = (416, 416)
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.4
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "yolov3-tiny.pt")
YOLO_NUM_CLASSES = 80
YOLO_ANCHORS = [
    [(10, 14), (23, 27), (37, 58)],
    [(81, 82), (135, 169), (344, 319)],
]
YOLO_STRIDES = [16, 32]

# ─── OpenVINO ────────────────────────────────────────────────────────────────
OPENVINO_MODEL_XML = os.path.join(BASE_DIR, "weights", "yolov3-tiny-openvino.xml")
OPENVINO_MODEL_BIN = os.path.join(BASE_DIR, "weights", "yolov3-tiny-openvino.bin")
OPENVINO_DEVICE = "CPU"

# ─── Camera & Calibration ───────────────────────────────────────────────────
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

CALIB_FILE = os.path.join(CALIB_DIR, "homography.npy")

# Reference distance (metres) between two known points on the road plane
REFERENCE_DISTANCE_M = 10.0
REFERENCE_PIXEL_PAIRS = None  # set as [(x1,y1), (x2,y2)]

# ─── Speed Estimation ───────────────────────────────────────────────────────
FRAME_SKIP = 2          # process every Nth frame
PIXELS_PER_METRE = None # auto-computed from calibration if available
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

# ─── SORT Tracker ────────────────────────────────────────────────────────────
SORT_MAX_AGE = 30
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# ─── Evaluation ──────────────────────────────────────────────────────────────
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
