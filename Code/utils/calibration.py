import cv2
import numpy as np


def select_road_points(image, num_points=4):
    window_name = "Select 4 points on the road plane (press SPACE after each, ESC to finish)"
    points = []

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, str(len(points)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_handler)

    while len(points) < num_points:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


def compute_homography(src_points, dst_points):
    H, _ = cv2.findHomography(src_points, dst_points)
    return H


def compute_pixels_per_metre(known_distance_m, pt1, pt2):
    dist_px = np.linalg.norm(np.array(pt2) - np.array(pt1))
    return dist_px / known_distance_m


class CameraCalibrator:
    def __init__(self):
        self.homography = None
        self.pixels_per_metre = None

    def calibrate_from_points(self, image, src_points, dst_points):
        self.homography = compute_homography(src_points, dst_points)
        return self.homography

    def calibrate_from_reference(self, known_distance_m, pt1, pt2):
        self.pixels_per_metre = compute_pixels_per_metre(known_distance_m, pt1, pt2)
        return self.pixels_per_metre

    def save(self, homography_path, ppm_path=None):
        if self.homography is not None:
            np.save(homography_path, self.homography)
        if ppm_path is not None and self.pixels_per_metre is not None:
            np.save(ppm_path, np.array([self.pixels_per_metre]))

    def load(self, homography_path, ppm_path=None):
        self.homography = np.load(homography_path)
        if ppm_path is not None:
            self.pixels_per_metre = float(np.load(ppm_path)[0])
        return self.homography
