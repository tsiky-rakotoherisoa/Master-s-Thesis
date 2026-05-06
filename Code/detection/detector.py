import cv2
import torch
import numpy as np

from config import (
    YOLO_INPUT_SIZE, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    YOLO_ANCHORS, YOLO_STRIDES
)
from models.yolo_tiny import YOLOv3Tiny, decode_outputs
from models.openvino_model import OpenVINOModel


class YOLODetector:
    def __init__(self, weights_path, num_classes=80, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.anchors = YOLO_ANCHORS
        self.strides = YOLO_STRIDES
        self.input_size = YOLO_INPUT_SIZE
        self.conf_threshold = YOLO_CONF_THRESHOLD
        self.iou_threshold = YOLO_IOU_THRESHOLD

        self.model = YOLOv3Tiny(num_classes=num_classes).to(self.device)
        self.model.eval()

        if weights_path:
            self._load_weights(weights_path)

    def _load_weights(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def detect(self, image):
        h, w = image.shape[:2]
        blob = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = torch.from_numpy(blob).unsqueeze(0).to(self.device)

        out13, out26 = self.model(blob)

        boxes, scores, class_ids = [], [], []
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]

        for out, anchors, stride in zip([out13, out26], self.anchors, self.strides):
            bx, by, bw, bh, conf, cls_probs = decode_outputs(
                out, anchors, self.num_classes, stride
            )
            for b in range(bx.shape[0]):
                for a in range(bx.shape[1]):
                    for i in range(bx.shape[2]):
                        for j in range(bx.shape[3]):
                            c = float(conf[b, a, i, j])
                            if c < self.conf_threshold:
                                continue
                            cls_scores = cls_probs[b, a, i, j].cpu().numpy()
                            cls_id = int(np.argmax(cls_scores))
                            cls_c = float(cls_scores[cls_id])
                            final_c = c * cls_c
                            if final_c < self.conf_threshold:
                                continue

                            x_c = float(bx[b, a, i, j]) * scale_x
                            y_c = float(by[b, a, i, j]) * scale_y
                            box_w = float(bw[b, a, i, j]) * scale_x
                            box_h = float(bh[b, a, i, j]) * scale_y

                            x1 = int(x_c - box_w / 2)
                            y1 = int(y_c - box_h / 2)
                            x2 = int(x_c + box_w / 2)
                            y2 = int(y_c + box_h / 2)

                            boxes.append([x1, y1, x2, y2])
                            scores.append(final_c)
                            class_ids.append(cls_id)

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]

        return boxes, scores, class_ids


class OpenVINODetector:
    def __init__(self, xml_path, bin_path, device="CPU"):
        self.model = OpenVINOModel(xml_path, bin_path, device)
        self.conf_threshold = YOLO_CONF_THRESHOLD
        self.iou_threshold = YOLO_IOU_THRESHOLD

    def detect(self, image):
        return self.model.detect(
            image,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
