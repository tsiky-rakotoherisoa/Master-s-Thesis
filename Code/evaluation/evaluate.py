import csv
import time
import psutil
import numpy as np

from utils.metrics import compute_all_metrics


class Benchmark:
    def __init__(self):
        self.process = psutil.Process()

    def measure_resources(self):
        cpu = self.process.cpu_percent(interval=0.1)
        mem = self.process.memory_percent()
        return cpu, mem

    def measure_fps(self, process_func, num_frames=100):
        start = time.perf_counter()
        for _ in range(num_frames):
            process_func()
        elapsed = time.perf_counter() - start
        return num_frames / elapsed if elapsed > 0 else 0.0


class Evaluator:
    def __init__(self, ground_truth_path=None):
        self.ground_truth = {}
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)

    def load_ground_truth(self, path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    timestamp = float(row[0])
                    speed = float(row[1])
                    self.ground_truth[timestamp] = speed

    def evaluate_speed_estimates(self, predictions):
        y_true = []
        y_pred = []
        for ts, pred_speed in predictions:
            if ts in self.ground_truth:
                y_true.append(self.ground_truth[ts])
                y_pred.append(pred_speed)

        if not y_true:
            return None

        metrics = compute_all_metrics(y_true, y_pred)
        metrics["num_samples"] = len(y_true)
        return metrics

    @staticmethod
    def print_results(metrics, model_name):
        print(f"\n{'='*45}")
        print(f"  {model_name}")
        print(f"{'='*45}")
        if metrics is None:
            print("  No valid data for evaluation.")
            return
        print(f"  Samples        : {metrics.get('num_samples', 'N/A')}")
        print(f"  MAE  (km/h)    : {metrics.get('MAE', 'N/A'):.4f}")
        print(f"  RMSE (km/h)    : {metrics.get('RMSE', 'N/A'):.4f}")
        print(f"  MAPE (%)       : {metrics.get('MAPE', 'N/A'):.2f}")
        print(f"  R²   (%)       : {metrics.get('R2', 'N/A') * 100:.2f}")
        print(f"{'='*45}\n")

    @staticmethod
    def save_results(results, path):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ground_truth", "predicted", "model"])
            for row in results:
                writer.writerow(row)
