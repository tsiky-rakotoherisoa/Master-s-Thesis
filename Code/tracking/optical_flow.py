import cv2
import numpy as np

from config import FARNEBACK_PARAMS


class FarnebackOpticalFlow:
    def __init__(self, params=None):
        if params is None:
            params = FARNEBACK_PARAMS
        self.params = params
        self.prev_gray = None

    def set_previous_frame(self, frame_gray):
        self.prev_gray = frame_gray

    def compute(self, frame_gray):
        if self.prev_gray is None:
            h, w = frame_gray.shape
            self.prev_gray = frame_gray
            return np.zeros((h, w, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            frame_gray,
            None,
            self.params["pyr_scale"],
            self.params["levels"],
            self.params["winsize"],
            self.params["iterations"],
            self.params["poly_n"],
            self.params["poly_sigma"],
            self.params["flags"],
        )
        self.prev_gray = frame_gray
        return flow

    def get_vehicle_displacement(self, flow, bbox):
        x1, y1, x2, y2 = bbox
        region = flow[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0, 0.0
        mean_dx = float(np.mean(region[..., 0]))
        mean_dy = float(np.mean(region[..., 1]))
        return mean_dx, mean_dy


class SORTTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def _get_centroid(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def update(self, detections):
        for t in self.tracks:
            t["missed"] += 1

        matched_dets = set()
        for track in self.tracks:
            best_iou = 0
            best_d_idx = -1
            for di, det in enumerate(detections):
                if di in matched_dets:
                    continue
                iou = self._iou(track["bbox"], det)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = di
            if best_iou >= self.iou_threshold and best_d_idx >= 0:
                track["bbox"] = detections[best_d_idx]
                track["missed"] = 0
                track["hits"] += 1
                matched_dets.add(best_d_idx)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append({
                    "id": self.next_id,
                    "bbox": det,
                    "missed": 0,
                    "hits": 1,
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_age]

        result = []
        for t in self.tracks:
            if t["hits"] >= self.min_hits:
                result.append((t["id"], t["bbox"]))
        return result
