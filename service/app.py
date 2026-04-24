from fastapi import FastAPI, HTTPException

from service.inference import YOLOMaskService
from service.logger import get_logger
from service.schemas import InferenceRequest, InferenceResponse, ROI

app = FastAPI(
    title="SubStation AI Mask Inference Service",
    description="基于YOLO分割模型的ROI裁剪推理服务，返回原图坐标下的 mask 轮廓。",
)

# 初始化日志记录器
logger = get_logger(name="service", log_dir="logs", prefix="service")

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
        logger.info(f"收到推理请求: {len(request.rois)} 个ROI, 权重: {request.weights_path}")

        image_rgb = YOLOMaskService.decode_base64_image(request.image_base64)
        logger.debug(f"图像解码成功: {image_rgb.shape}")

    except ValueError as exc:
        logger.error(f"请求参数错误: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))

    normalized_rois = request.rois or [ROI(x1=0, y1=0, x2=image_rgb.shape[1], y2=image_rgb.shape[0])]

    try:
        service = get_model(
            weights_path=request.weights_path,
            device=request.device,
            conf_threshold=request.conf_threshold,
            img_size=request.img_size,
        )

        logger.debug("开始模型推理")
        result = service.predict(image_rgb=image_rgb, rois=normalized_rois)
        logger.info(f"推理完成: 检测到 {sum(len(r['detections']) for r in result['results'])} 个目标")

    except Exception as exc:
        logger.exception(f"推理过程出错: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return result
