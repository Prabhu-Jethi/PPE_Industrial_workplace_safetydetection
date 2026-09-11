from typing import List
from pydantic import BaseModel, Field

class WorkerDetail(BaseModel):
    worker_id: int = Field(..., description="ID assigned to the worker in this frame")
    box: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    status: str = Field(..., description="Compliance status: 'COMPLIANT' or 'VIOLATION'")
    equipment: List[str] = Field(default_factory=list, description="List of detected PPE equipment on this worker")

class DetectionResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed image file")
    total_workers: int = Field(..., description="Total count of workers detected")
    total_ppe_items: int = Field(..., description="Total individual PPE objects identified")
    violations_detected: int = Field(..., description="Count of workers in violation of safety rules")
    workers: List[WorkerDetail] = Field(default_factory=list, description="Detailed list of detected workers")

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str

