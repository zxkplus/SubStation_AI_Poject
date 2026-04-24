"""
服务接口测试
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

# 将仓库根目录加入路径，确保 service 包可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from service.app import app
from service.inference import YOLOMaskService

client = TestClient(app)


def create_base64_image() -> str:
    """创建一个简单的测试图像并返回 Base64 编码字符串。"""
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    success, buffer = cv2.imencode('.png', image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def test_decode_base64_image():
    image_base64 = create_base64_image()
    image_rgb = YOLOMaskService.decode_base64_image(image_base64)

    assert image_rgb.shape == (16, 16, 3)
    assert image_rgb.dtype == np.uint8
    assert image_rgb[0, 0, 0] == 128


def test_infer_endpoint_success(monkeypatch):
    image_base64 = create_base64_image()

    class DummyService:
        def predict(self, image_rgb, rois=None):
            return {
                "image_width": int(image_rgb.shape[1]),
                "image_height": int(image_rgb.shape[0]),
                "results": [
                    {
                        "roi": {
                            "x1": rois[0].x1,
                            "y1": rois[0].y1,
                            "x2": rois[0].x2,
                            "y2": rois[0].y2,
                        },
                        "detections": [
                            {
                                "bbox": [1, 2, 3, 4],
                                "confidence": 0.9,
                                "class_id": 0,
                                "contours": [{"points": [[1, 1], [2, 1], [2, 2], [1, 2]]}],
                            }
                        ],
                    }
                ],
            }

    monkeypatch.setattr("service.app.get_model", lambda weights_path, device, conf_threshold, img_size: DummyService())

    payload = {
        "image_base64": image_base64,
        "rois": [{"x1": 0, "y1": 0, "x2": 8, "y2": 8}],
    }

    response = client.post("/infer", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["image_width"] == 16
    assert data["image_height"] == 16
    assert len(data["results"]) == 1
    assert data["results"][0]["roi"]["x2"] == 8
    assert data["results"][0]["detections"][0]["bbox"] == [1, 2, 3, 4]


def test_infer_endpoint_invalid_base64():
    payload = {"image_base64": "not_a_base64_string", "rois": []}

    response = client.post("/infer", json=payload)
    assert response.status_code == 400
    assert "无法解码 base64 图像数据" in response.json()["detail"]


def test_encode_image_to_base64():
    """测试图片编码为 base64"""
    import tempfile

    # 创建临时图片文件
    test_image = np.full((16, 16, 3), 128, dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, test_image)

    try:
        # 测试编码
        base64_str = YOLOMaskService.encode_image_to_base64(temp_path)

        # 验证 base64 字符串不为空
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # 验证可以解码回原图片
        decoded_image = YOLOMaskService.decode_base64_image(base64_str)
        assert decoded_image.shape == (16, 16, 3)
        assert decoded_image.dtype == np.uint8

    finally:
        # 清理临时文件
        import os
        os.unlink(temp_path)

##测试真实图片
def test_real_image_infer():
    imagepath = "test_data/images/val/images/bikecircle_.jpg"
    with open(imagepath, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "image_base64": image_base64,
        "rois": [{"x1": 34, "y1": 52, "x2": 1116, "y2": 1978}],
        "weights_path": "weights/yolov8n-seg.pt",
        "conf_threshold": 0.25,
        "img_size": 640,
        "device": "cpu"
    }
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()

    ##将json的轮廓，在原图上画出来，保存为测试结果
    import json
    from PIL import Image, ImageDraw
    image = Image.open(imagepath).convert("RGB")
    draw = ImageDraw.Draw(image)
    for roi_result in data["results"]:
        for detection in roi_result["detections"]:
            for contour in detection["contours"]:
                points = contour["points"]
                draw.polygon(points, outline="red")
    image.save("test_output/infer_result.jpg")