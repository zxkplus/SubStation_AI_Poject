from typing import List, Optional

from pydantic import BaseModel, Field


class ROI(BaseModel):
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)


class InferenceRequest(BaseModel):
    image_base64: str
    rois: Optional[List[ROI]] = Field(default_factory=list)
    weights_path: str = "yolov8n-seg.pt"
    conf_threshold: float = 0.25
    img_size: int = 640
    device: str = "cpu"


class MaskContour(BaseModel):
    points: List[List[int]]


class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    contours: List[MaskContour]


class ROIResult(BaseModel):
    roi: ROI
    detections: List[DetectionResult]


class InferenceResponse(BaseModel):
    image_width: int
    image_height: int
    results: List[ROIResult]
