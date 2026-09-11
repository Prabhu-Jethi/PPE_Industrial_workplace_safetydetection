import os
import sys
import io
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, File, UploadFile
from src.rule_engine import process_detections
from api.schemas import DetectionResponse, HealthResponse, WorkerDetail

app = FastAPI(
    title="PPE Workplace Safety Detection API",
    description="Production API for detecting workers, PPE gear, and workplace safety violations.",
    version="1.0.0"
)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
model = YOLO(MODEL_PATH)


@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="online", model="YOLOv8-PPE", version="1.0.0")


@app.post("/detect", response_model=DetectionResponse)
async def detect_ppe(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, executes YOLOv8 detection + Spatial Rule Engine,
    and returns structured worker compliance details.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    results = model(image_np, conf=0.20, verbose=False)[0]

    workers, ppe_items = process_detections(
        boxes=results.boxes.xyxy.cpu().numpy(),
        classes=results.boxes.cls.cpu().numpy(),
        confidences=results.boxes.conf.cpu().numpy(),
        names=model.names
    )

    worker_details = [
        WorkerDetail(
            worker_id=idx + 1,
            box=[float(coord) for coord in worker["box"]],
            status="VIOLATION" if worker["is_violating"] else "COMPLIANT",
            equipment=worker["equipment"]
        )
        for idx, worker in enumerate(workers)
    ]

    return DetectionResponse(
        filename=file.filename or "uploaded_image.jpg",
        total_workers=len(workers),
        total_ppe_items=len(ppe_items),
        violations_detected=sum(1 for w in workers if w["is_violating"]),
        workers=worker_details
    )