import numpy as np
from pathlib import Path

try:
    from openvino import Core
except ImportError:
    Core = None


class OpenVINOModel:
    def __init__(self, xml_path, bin_path, device="CPU"):
        if Core is None:
            raise ImportError(
                "OpenVINO is not installed. "
                "Install it with: pip install openvino"
            )

        self.xml_path = Path(xml_path)
        self.bin_path = Path(bin_path)
        self.device = device

        if not self.xml_path.exists():
            raise FileNotFoundError(f"OpenVINO XML not found: {self.xml_path}")
        if not self.bin_path.exists():
            raise FileNotFoundError(f"OpenVINO BIN not found: {self.bin_path}")

        self.core = Core()
        self.model = self.core.read_model(str(self.xml_path), str(self.bin_path))
        self.compiled = self.core.compile_model(self.model, self.device)
        self.infer_request = self.compiled.create_infer_request()

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.output_tensor_names = [out.get_any_name() for out in self.model.outputs]

    def preprocess(self, image, target_size=(416, 416)):
        import cv2
        h, w = image.shape[:2]
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        return blob, (w, h)

    def infer(self, blob):
        results = self.infer_request.infer({self.input_tensor_name: blob})
        return [results[name] for name in self.output_tensor_names]

    def postprocess(self, raw_outputs, original_shape, conf_threshold=0.5, iou_threshold=0.4):
        import cv2
        boxes, scores, class_ids = [], [], []
        orig_w, orig_h = original_shape

        for output in raw_outputs:
            output = output.squeeze()
            if output.ndim == 2:
                output = output.T
            for detection in output:
                scores_det = detection[4:]
                class_id = int(np.argmax(scores_det))
                confidence = float(scores_det[class_id])
                if confidence < conf_threshold:
                    continue

                cx, cy, bw, bh = detection[:4]
                x1 = int((cx - bw / 2) * orig_w / self.input_shape[2])
                y1 = int((cy - bh / 2) * orig_h / self.input_shape[1])
                x2 = int((cx + bw / 2) * orig_w / self.input_shape[2])
                y2 = int((cy + bh / 2) * orig_h / self.input_shape[1])

                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
                class_ids.append(class_id)

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]

        return boxes, scores, class_ids

    def detect(self, image, conf_threshold=0.5, iou_threshold=0.4):
        blob, orig_shape = self.preprocess(image)
        raw_outputs = self.infer(blob)
        return self.postprocess(raw_outputs, orig_shape, conf_threshold, iou_threshold)
