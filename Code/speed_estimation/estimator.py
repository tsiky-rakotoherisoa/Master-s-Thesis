import numpy as np
import cv2


class SpeedEstimator:
    def __init__(self, camera_fps=30, pixels_per_metre=None, homography_matrix=None):
        self.camera_fps = camera_fps
        self.pixels_per_metre = pixels_per_metre
        self.homography = homography_matrix
        self.track_history = {}

    def set_homography(self, src_points, dst_points):
        self.homography = cv2.findHomography(src_points, dst_points)[0]

    def load_homography(self, path):
        self.homography = np.load(path)

    def _to_world(self, pt):
        if self.homography is None:
            return np.array(pt, dtype=float)
        pt_h = np.array([pt[0], pt[1], 1.0], dtype=float)
        world = self.homography @ pt_h
        return world[:2] / world[2]

    def _pixel_speed_to_kmh(self, displacement_px, dt):
        if dt <= 0:
            return 0.0
        if self.pixels_per_metre is None or self.pixels_per_metre <= 0:
            return 0.0
        displacement_m = displacement_px / self.pixels_per_metre
        speed_m_s = displacement_m / dt
        return speed_m_s * 3.6

    def _world_speed_to_kmh(self, displacement_world, dt):
        if dt <= 0:
            return 0.0
        dist_m = float(np.linalg.norm(displacement_world))
        speed_m_s = dist_m / dt
        return speed_m_s * 3.6

    def estimate_from_flow(self, vehicle_id, flow_dx, flow_dy, timestamp):
        if vehicle_id not in self.track_history:
            self.track_history[vehicle_id] = []
        self.track_history[vehicle_id].append({
            "timestamp": timestamp,
            "flow_dx": flow_dx,
            "flow_dy": flow_dy,
        })

        history = self.track_history[vehicle_id]
        if len(history) < 2:
            return None

        prev = history[-2]
        curr = history[-1]
        dt = curr["timestamp"] - prev["timestamp"]

        if dt <= 0:
            return None

        if self.homography is not None:
            p_px = np.array([prev["flow_dx"], prev["flow_dy"]])
            c_px = np.array([curr["flow_dx"], curr["flow_dy"]])
            p_w = self._to_world(p_px)
            c_w = self._to_world(c_px)
            disp = c_w - p_w
            speed = self._world_speed_to_kmh(disp, dt)
        else:
            disp_px = np.sqrt(curr["flow_dx"]**2 + curr["flow_dy"]**2)
            speed = self._pixel_speed_to_kmh(disp_px, dt)

        return round(speed, 2)

    def estimate_from_centroid(self, vehicle_id, centroid, timestamp):
        if vehicle_id not in self.track_history:
            self.track_history[vehicle_id] = []
        self.track_history[vehicle_id].append({
            "timestamp": timestamp,
            "centroid": centroid,
        })

        history = self.track_history[vehicle_id]
        if len(history) < 2:
            return None

        prev = history[-2]
        curr = history[-1]
        dt = curr["timestamp"] - prev["timestamp"]

        if dt <= 0:
            return None

        if self.homography is not None:
            p_w = self._to_world(prev["centroid"])
            c_w = self._to_world(curr["centroid"])
        else:
            p_w = np.array(prev["centroid"], dtype=float)
            c_w = np.array(curr["centroid"], dtype=float)

        disp = c_w - p_w
        return round(self._world_speed_to_kmh(disp, dt), 2)

    def reset(self):
        self.track_history.clear()
