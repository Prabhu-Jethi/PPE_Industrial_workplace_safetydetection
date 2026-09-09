import os
import urllib.request

def download_pretrained_ppe_model():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/best.pt"
    
    if os.path.exists(model_path):
        print(f"Pretrained model already exists at {model_path}")
        return model_path
    
    ## Pre-trained model 'yolov8n'
    url = "https://huggingface.co/keremberke/yolov8m-protective-equipment-detection/resolve/main/best.pt"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Successfully downloaded pretrained model to {model_path}")
    except Exception as e:
        print(f"Failed to download the model. Error: {e}")
        
    return model_path

if __name__ == "__main__":
    download_pretrained_ppe_model()