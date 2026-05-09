import base64
import binascii
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from service.schemas import ROI


class YOLOMaskService:
    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        img_size: int = 640,
    ):
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model = YOLO(str(self.weights_path))

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray:
        try:
            payload = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("无法解码 base64 图像数据") from exc

        array = np.frombuffer(payload, dtype=np.uint8)
        image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("base64 图像内容无法转换为有效图像，请检查输入是否为合法 JPEG/PNG 数据。")

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """将图片文件编码为 base64 字符串"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片文件: {image_path}")

            # 转换为 JPEG 格式进行编码
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError(f"图片编码失败: {image_path}")

            return base64.b64encode(buffer.tobytes()).decode('utf-8')
        except Exception as exc:
            raise ValueError(f"图片编码失败: {str(exc)}") from exc

    @staticmethod
    def normalize_roi(roi: ROI, width: int, height: int) -> ROI:
        x1 = max(0, min(roi.x1, width - 1))
        y1 = max(0, min(roi.y1, height - 1))
        x2 = max(0, min(roi.x2, width))
        y2 = max(0, min(roi.y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI 坐标无效，必须满足 x2>x1 且 y2>y1。")

        return ROI(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def extract_contours(mask: np.ndarray, roi: ROI) -> List[dict]:
        if mask is None or mask.size == 0:
            return []

        mask_binary = (mask > 0.5).astype(np.uint8)
        roi_width = roi.x2 - roi.x1
        roi_height = roi.y2 - roi.y1

        if mask_binary.shape != (roi_height, roi_width):
            mask_binary = cv2.resize(mask_binary, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list: List[dict] = []
        for contour in contours:
            if contour is None or contour.size == 0:
                continue
            points = contour.reshape(-1, 2).tolist()
            shifted = [[int(pt[0]) + roi.x1, int(pt[1]) + roi.y1] for pt in points]
            contour_list.append({"points": shifted})

        return contour_list

    def predict(self, image_rgb: np.ndarray, rois: Optional[List[ROI]] = None):
        height, width = image_rgb.shape[:2]
        normalized_rois = rois or [ROI(x1=0, y1=0, x2=width, y2=height)]
        response = []

        for roi in normalized_rois:
            roi = self.normalize_roi(roi, width, height)
            crop = image_rgb[roi.y1:roi.y2, roi.x1:roi.x2]
            if crop.size == 0:
                response.append({"roi": roi, "detections": []})
                continue

            results = self.model.predict(
                source=crop,
                conf=self.conf_threshold,
                device=self.device,
                imgsz=self.img_size,
                verbose=False,
            )

            if not results or len(results) == 0:
                response.append({"roi": roi, "detections": []})
                continue

            result = results[0]
            detections = []

            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for idx in range(len(boxes)):
                    bbox = boxes.xyxy[idx].cpu().numpy().tolist()
                    bbox = [int(max(0, bbox[0] + roi.x1)), int(max(0, bbox[1] + roi.y1)), int(min(width, bbox[2] + roi.x1)), int(min(height, bbox[3] + roi.y1))]
                    confidence = float(boxes.conf[idx].cpu().item())
                    class_id = int(boxes.cls[idx].cpu().item())

                    mask_contours = []
                    if hasattr(result, "masks") and result.masks is not None:
                        mask_data = getattr(result.masks, "data", None)
                        if mask_data is not None and idx < len(mask_data):
                            mask = mask_data[idx].cpu().numpy()
                            mask_contours = self.extract_contours(mask, roi)

                    detections.append(
                        {
                            "bbox": bbox,
                            "confidence": confidence,
                            "class_id": class_id,
                            "contours": mask_contours,
                        }
                    )

            response.append({"roi": roi, "detections": detections})

        return {
            "image_width": width,
            "image_height": height,
            "results": response,
        }
