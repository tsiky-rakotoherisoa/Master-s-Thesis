import numpy as np


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)


def r_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def intersection_over_union(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def mean_average_precision(detections, ground_truths, iou_threshold=0.5):
    tp, fp, total_gt = 0, 0, len(ground_truths)
    matched_gt = set()

    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        for gi, gt in enumerate(ground_truths):
            if gi in matched_gt:
                continue
            iou = intersection_over_union(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    ap = (precision + recall) / 2 if (precision + recall) > 0 else 0.0
    return ap, precision, recall


def compute_all_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
    }
