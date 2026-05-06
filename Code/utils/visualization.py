import cv2
import numpy as np


COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
]


def draw_detections(image, boxes, scores, class_ids, class_names=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = COLORS[class_ids[i] % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[class_ids[i]] if class_names else 'vehicle'}: {scores[i]:.2f}" if scores else "vehicle"
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def draw_tracks(image, tracks, speeds=None):
    for item in tracks:
        if len(item) == 2:
            track_id, bbox = item
            speed = speeds.get(track_id, None) if speeds else None
        else:
            track_id, bbox, speed = item if len(item) == 3 else (item[0], item[1], None)

        x1, y1, x2, y2 = bbox
        color = COLORS[track_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (cx, cy), 4, color, -1)

        info = f"ID {track_id}"
        if speed is not None:
            info += f"  {speed:.1f} km/h"
        cv2.putText(image, info, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def draw_optical_flow(image, flow, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2).astype(int)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return image


def draw_fps(image, fps):
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_info_panel(image, model_name, cpu_usage=None, memory_usage=None):
    cv2.putText(image, f"Model: {model_name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_off = 90
    if cpu_usage is not None:
        cv2.putText(image, f"CPU: {cpu_usage:.1f}%", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off += 30
    if memory_usage is not None:
        cv2.putText(image, f"RAM: {memory_usage:.1f}%", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image
