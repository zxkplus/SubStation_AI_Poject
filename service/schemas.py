from typing import List, Optional
import json
from pathlib import Path

from pydantic import BaseModel, Field


class BaseSchema(BaseModel):
    """基础模式类，提供JSON序列化功能"""

    @classmethod
    def load_from_json(cls, file_path: str) -> 'BaseSchema':
        """从JSON文件中加载对象"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(**data)

    def save_to_json(self, file_path: str) -> None:
        """将对象保存为JSON文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)


class ROI(BaseModel):
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)


class InferenceRequest(BaseSchema):
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


class InferenceResponse(BaseSchema):
    image_width: int
    image_height: int
    results: List[ROIResult]
