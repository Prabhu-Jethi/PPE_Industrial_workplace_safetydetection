import os
import cv2
from ultralytics import YOLO

def run_inference():
    # Load the pretrained model you renamed to best.pt
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    model = YOLO(model_path)
    
    # Let's pick a random image from your validation set to test
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "valid", "images"))
    
    if not os.path.exists(image_path):
        print("Could not find validation images.")
        return
        
    # Get the first image we can find
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    if not images:
        print("No images found to run inference on!")
        return
        
    test_image = os.path.join(image_path, images[0])
    print(f"Running inference on {test_image}...")
    
    # Run the model
    results = model(test_image)
    
    # Save the output
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "inference_test.jpg")
    results[0].save(output_file)
    print(f"Success! Check {output_file} to see the bounding boxes.")




if __name__ == "__main__":
    run_inference()

