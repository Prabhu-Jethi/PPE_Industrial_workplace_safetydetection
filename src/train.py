from ultralytics import YOLO
import os
import shutil


def train_model():
    # 1. Load the Nano model (fastest model, best for CPU training)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt"))
    model = YOLO(model_path) 
    
    # Path to your data.yaml file
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.yaml"))

    print(f"Starting training on your Roboflow dataset: {data_path}")


    # 2. Train the model
    results = model.train(
        data=data_path,      
        epochs=5,             
        imgsz=640,             
        batch=4,               
        project="models",      
        name="yolo_run",
        plots=True,
        device="cpu"      
    )

    print(f"Model trained..")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    exported_path = os.path.join(results.save_dir, "weights", "model_path")

    shutil.copy2(exported_path, model_path)
    print(f"Model successfully saved at {model_path}")

    return results
    

if __name__ == "__main__":
    train_model()