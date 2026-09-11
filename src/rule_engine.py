import os
import cv2
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Helps determine if a PPE item (box2) is physically on a Worker (box1).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0
    return intersection / union

def process_detections(boxes, classes, confidences, names):
    """
    Takes raw YOLO detections, pairs PPE with workers, and returns:
    1. workers: list of workers with equipment assigned and compliance status.
    2. ppe_items: list of all detected PPE objects with coordinates and violation flags.
    """
    workers = []
    ppe_items = []

    # 1. Separate workers from individual PPE items
    for i in range(len(boxes)):
        class_name = names[int(classes[i])]
        is_no_item = class_name.lower().startswith("no-")
        
        detection = {
            "box": boxes[i],
            "class": class_name,
            "conf": confidences[i],
            "is_violating": is_no_item
        }
        
        if class_name == "Worker":
            detection["equipment"] = []
            detection["is_violating"] = False
            workers.append(detection)
        else:
            ppe_items.append(detection)

    # 2. Match PPE to the closest Worker
    for ppe in ppe_items:
        best_iou = 0
        assigned_worker = None
        
        for worker in workers:
            iou = calculate_iou(worker["box"], ppe["box"])
            if iou > best_iou:
                best_iou = iou
                assigned_worker = worker
                
        # If the PPE overlaps with a worker, assign it
        if assigned_worker is not None and best_iou > 0.05:
            assigned_worker["equipment"].append(ppe["class"])
            
            # If the item is a violation (e.g. No-Helmet, No-Vest), mark the worker as violating
            if ppe["is_violating"]:
                assigned_worker["is_violating"] = True

    return workers, ppe_items


if __name__ == "__main__":
    print("Testing the Rule Engine...")
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    model = YOLO(model_path)
    
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "valid", "images"))
    images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    test_image = os.path.join(image_dir, images[0])
    
    results = model(test_image, conf=0.15, verbose=False)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    names = model.names
    
    workers, ppe_items = process_detections(boxes, classes, confidences, names)
    
    print(f"Detected {len(workers)} Worker(s) and {len(ppe_items)} PPE item(s).")
    for idx, worker in enumerate(workers):
        status = "VIOLATION" if worker['is_violating'] else "COMPLIANT"
        print(f"Worker {idx+1}: [{status}] | Equipment: {worker['equipment']}")