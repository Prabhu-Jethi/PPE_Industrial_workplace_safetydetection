import os
import cv2
import time
from ultralytics import YOLO
from rule_engine import process_detections

## PATH CONFIGURATION
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_VIDEO = os.path.join(PROJECT_ROOT, "data", "sample_video.avi")
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, "annotated_video.avi")

class VideoProcessor:
    def __init__(self, model_path, conf=0.25, frame_skips=2):
        ## Initializes the Video Processor with YOLO and Rule Engine. frame_skips=2 means every 2nd frame is evaluated to optimize CPU speed.
        self.model = YOLO(model_path)
        self.conf = conf
        self.frame_skips = frame_skips

    def process_frame(self, frame):
        ## Runs YOLO detection + Rule Engine on a single frame and draws annotations.
        annotated = frame.copy()

        # 1. Run YOLO inference
        results = self.model(annotated, conf=self.conf, verbose=False)[0]

        # 2. Run Rule Engine
        workers, ppe_items = process_detections(
            boxes=results.boxes.xyxy.cpu().numpy(),
            classes=results.boxes.cls.cpu().numpy(),
            confidences=results.boxes.conf.cpu().numpy(),
            names=self.model.names
        )

        # 3. Draw individual PPE items (Thin boxes)
        for ppe in ppe_items:
            px1, py1, px2, py2 = map(int, ppe["box"])
            is_violation = ppe["is_violating"]
            item_color = (0, 0, 255) if is_violation else (0, 255, 0)
            item_label = f"{ppe['class']} ({ppe['conf']:.2f})"
            
            cv2.rectangle(annotated, (px1, py1), (px2, py2), item_color, 2)
            cv2.putText(annotated, item_label, (px1, max(py1 - 6, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, item_color, 1)

        # 4. Draw Workers (Thick boxes + Compliance status)
        violation_count = 0
        for worker in workers:
            wx1, wy1, wx2, wy2 = map(int, worker["box"])
            is_violating = worker["is_violating"]

            if is_violating:
                violation_count += 1
                worker_color = (0, 0, 255)
                status_text = "VIOLATION"
            else:
                worker_color = (0, 255, 0)
                status_text = "COMPLIANT"

            worker_label = f"[{status_text}] Worker"
            cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), worker_color, 3)
            cv2.putText(annotated, worker_label, (wx1, max(wy1 - 10, 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, worker_color, 2)

        return annotated, len(workers), violation_count

    def process_video(self, input_path, output_path):
        ## Reads input video frame-by-frame, runs the pipeline, and saves an annotated video.
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_violations = 0
        start_time = time.time()
        print(f"Starting video processing on: {input_path}")

        last_annotated_frame = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1

            # Process frame on skip intervals
            if frame_count % self.frame_skips == 0 or last_annotated_frame is None:
                annotated_frame, num_workers, violations = self.process_frame(frame)
                total_violations += violations
                last_annotated_frame = annotated_frame
            else:
                annotated_frame = last_annotated_frame

            out.write(annotated_frame)

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        fps_speed = frame_count / max(elapsed, 0.001)
        print(f"\nProcessing Complete!")
        print(f"Total Frames Processed: {frame_count} in {elapsed:.2f}s (~{fps_speed:.1f} FPS)")
        print(f"Total Violations Flagged: {total_violations}")
        print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    processor = VideoProcessor(MODEL_PATH, conf=0.20, frame_skips=2)
    processor.process_video(input_path=INPUT_VIDEO, output_path=OUTPUT_VIDEO)
