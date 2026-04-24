from fastapi import FastAPI, HTTPException

from service.inference import YOLOMaskService
from service.schemas import InferenceRequest, InferenceResponse, ROI

app = FastAPI(
    title="SubStation AI Mask Inference Service",
    description="基于YOLO分割模型的ROI裁剪推理服务，返回原图坐标下的 mask 轮廓。",
)

_model_cache = {}


def get_model(weights_path: str, device: str, conf_threshold: float, img_size: int) -> YOLOMaskService:
    key = (weights_path, device, conf_threshold, img_size)
    if key not in _model_cache:
        _model_cache[key] = YOLOMaskService(
            weights_path=weights_path,
            device=device,
            conf_threshold=conf_threshold,
            img_size=img_size,
        )
    return _model_cache[key]


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    try:
        image_rgb = YOLOMaskService.decode_base64_image(request.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    normalized_rois = request.rois or [ROI(x1=0, y1=0, x2=image_rgb.shape[1], y2=image_rgb.shape[0])]

    try:
        service = get_model(
            weights_path=request.weights_path,
            device=request.device,
            conf_threshold=request.conf_threshold,
            img_size=request.img_size,
        )
        result = service.predict(image_rgb=image_rgb, rois=normalized_rois)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result
