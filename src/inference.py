import os
import cv2
import random
import numpy as np
from ultralytics import YOLO


def run_inference():
    # Load the pretrained model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    model = YOLO(model_path)
    
    # Pick validation images
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "valid", "images"))
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    random.shuffle(images) # Shuffle the list so it tests randomly
    
    print("Searching for an image with actual detections...")
    for img in images:
        test_image = os.path.join(image_path, img)
        
        # Run model with conf=0.1
        results = model(test_image, conf=0.1, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # If we find an image with boxes, stop and draw!
        if len(boxes) > 0:
            print(f"\nFound detections in {test_image}!")
            
            img = cv2.imread(test_image)
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            names = model.names

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cls_id = int(classes[i])
                conf = confidences[i]
                label = f"{names[cls_id]} {conf:.2f}"
                
                is_violation = "NO-" in names[cls_id]
                color = (0, 0, 255) if is_violation else (0, 255, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
            os.makedirs(results_dir, exist_ok=True)
            output_file = os.path.join(results_dir, "inference_test_cv2.jpg")
            cv2.imwrite(output_file, img)
            print(f"Success! Check {output_file} to see the CV2 bounding boxes.")
            return

    print("Could not find ANY detections in the entire validation folder.")




if __name__ == "__main__":
    run_inference()
