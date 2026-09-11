import os
import cv2
import random
from ultralytics import YOLO
from rule_engine import process_detections

def run_inference():
    # 1. Load the model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    model = YOLO(model_path)
    
    # 2. Pick validation images randomly
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "valid", "images"))
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    random.shuffle(images)
    
    print("Searching for an image with workers and equipment...")
    for img_name in images:
        test_image = os.path.join(image_path, img_name)
        
        # Run YOLO model
        results = model(test_image, conf=0.20, verbose=False)[0]
        
        # Rule Engine processes both Workers AND individual PPE items
        workers, ppe_items = process_detections(
            boxes=results.boxes.xyxy.cpu().numpy(),
            classes=results.boxes.cls.cpu().numpy(),
            confidences=results.boxes.conf.cpu().numpy(),
            names=model.names
        )
        
        # If workers or equipment are found
        if len(workers) > 0 or len(ppe_items) > 0:
            print(f"Found {len(workers)} worker(s) and {len(ppe_items)} PPE item(s) in {img_name}!")
            
            img = cv2.imread(test_image)
            
            # 1. Draw individual PPE items
            for ppe in ppe_items:
                px1, py1, px2, py2 = map(int, ppe["box"])
                is_violation = ppe["is_violating"]
                
                # Red for No-Helmet/No-Vest, Green for Hard_hat/Vest/Boots/Glove
                item_color = (0, 0, 255) if is_violation else (0, 255, 0)
                item_label = f"{ppe['class']} ({ppe['conf']:.2f})"
                
                # Draw a thin box around the specific equipment
                cv2.rectangle(img, (px1, py1), (px2, py2), item_color, 2)
                cv2.putText(img, item_label, (px1, max(py1 - 6, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, item_color, 1)


            # 2. Draw the Worker bounding boxes with overall status
            for worker in workers:
                wx1, wy1, wx2, wy2 = map(int, worker["box"])
                is_violating = worker["is_violating"]
                
                # Thick RED box if violating, Thick GREEN box if compliant
                worker_color = (0, 0, 255) if is_violating else (0, 255, 0)
                status_text = "VIOLATION" if is_violating else "COMPLIANT"
                
                worker_label = f"[{status_text}] Worker"
                
                # Draw thick box around the entire person
                cv2.rectangle(img, (wx1, wy1), (wx2, wy2), worker_color, 3)
                
                # Background banner for readability
                cv2.putText(img, worker_label, (wx1, max(wy1 - 10, 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, worker_color, 2)
            
            # Save the annotated image
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
            os.makedirs(results_dir, exist_ok=True)
            output_file = os.path.join(results_dir, "inference_test_cv2.jpg")
            cv2.imwrite(output_file, img)
            
            print(f"Success! Check annotated image at: {output_file}")
            return

    print("No workers or equipment found in the validation images.")

if __name__ == "__main__":
    run_inference()
