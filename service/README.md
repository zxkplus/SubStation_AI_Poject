# SubStation AI 服务模块

本目录包含 SubStation AI 项目的推理服务和客户端应用。

## 📁 目录结构

```
service/
├── app.py                  # FastAPI 推理服务主应用
├── inference.py            # YOLO 推理核心逻辑
├── schemas.py              # Pydantic 数据模型定义
├── logger.py               # 日志记录模块
├── run.py                  # 服务启动脚本
├── client_app.py           # GUI 客户端应用（新增）
├── client_example.py       # 客户端使用示例（新增）
├── launch_client.py        # 客户端快速启动脚本（新增）
├── client_requirements.txt # 客户端依赖（新增）
└── CLIENT_README.md        # 客户端详细文档（新增）
```

## 🚀 快速开始

### 1. 启动推理服务

```bash
# 方法一：使用 uvicorn
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000

# 方法二：使用启动脚本
python service/run.py
```

服务启动后，可以访问以下地址：
- API 文档: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc

### 2. 运行 GUI 客户端

```bash
# 方法一：直接运行
cd service
python client_app.py

# 方法二：使用启动脚本（推荐，自动检查依赖）
cd service
python launch_client.py
```

## 📖 功能说明

### 推理服务 (app.py)

提供 RESTful API 接口用于实例分割推理：

**端点**: `POST /infer`

**请求体**:
```json
{
  "image_base64": "base64编码的图片字符串",
  "rois": [
    {"x1": 100, "y1": 100, "x2": 500, "y2": 500}
  ],
  "weights_path": "yolov8n-seg.pt",
  "conf_threshold": 0.25,
  "img_size": 640,
  "device": "cpu"
}
```

**响应**:
```json
{
  "image_width": 800,
  "image_height": 600,
  "results": [
    {
      "roi": {"x1": 100, "y1": 100, "x2": 500, "y2": 500},
      "detections": [
        {
          "bbox": [150, 150, 300, 350],
          "confidence": 0.95,
          "class_id": 0,
          "contours": [
            {"points": [[150, 150], [300, 150], [300, 350], [150, 350]]}
          ]
        }
      ]
    }
  ]
}
```

### GUI 客户端 (client_app.py)

图形界面应用，提供以下功能：

✅ **图片选择与显示**
- 支持 JPG、PNG、BMP 格式
- 自动缩放以适应窗口

✅ **交互式 ROI 划定**
- 鼠标拖拽绘制矩形区域
- 实时显示 ROI 边框
- 支持清除和重绘

✅ **可配置推理参数**
- 模型权重路径
- 置信度阈值 (0.0-1.0)
- 图像尺寸 (320-1920)
- 设备选择 (cpu/cuda)

✅ **结果可视化**
- 彩色半透明 mask 填充
- mask 轮廓线
- 检测边界框
- 类别标签和置信度

### 日志模块 (logger.py)

线程安全的日志记录系统：

```python
from service.logger import get_logger

logger = get_logger(name="service", log_dir="logs", prefix="service")
logger.info("推理请求已接收")
logger.error("发生错误", exc_info=True)
```

特性：
- 线程安全
- 按日期分割日志文件
- 自动捕获异常
- 支持多个日志级别

## 🔧 安装依赖

### 服务端依赖

已在项目根目录的 `requirements.txt` 中定义。

### 客户端依赖

```bash
cd service
pip install -r client_requirements.txt
```

或手动安装：
```bash
pip install Pillow requests numpy
```

**注意**: tkinter 是 Python 标准库，通常无需安装。

## 💻 编程方式使用

除了 GUI 界面，还可以以编程方式调用：

```python
from service.client_app import InferenceClient, ResultRenderer
from PIL import Image

# 创建客户端
client = InferenceClient(base_url="http://localhost:8000")

# 执行推理
result = client.predict(
    image_path="test.jpg",
    roi={"x1": 100, "y1": 100, "x2": 500, "y2": 500},
    weights_path="yolov8n-seg.pt",
    conf_threshold=0.25,
    img_size=640,
    device="cpu"
)

# 可视化结果
image = Image.open("test.jpg")
result_image = ResultRenderer.draw_results_on_image(image, result)
result_image.save("result.jpg")
```

更多示例请查看 `client_example.py`。

## 📊 架构设计

```
┌─────────────┐         HTTP POST          ┌──────────────┐
│             │  ──────────────────────►   │              │
│  GUI Client │                            │ FastAPI App  │
│             │  ◄──────────────────────   │              │
└─────────────┘     JSON Response          └──────┬───────┘
                                                  │
                                                  │ 调用
                                                  ▼
                                          ┌──────────────┐
                                          │ YOLO Model   │
                                          │ (Ultralytics)│
                                          └──────────────┘
```

## 🎯 使用场景

### 场景 1: 交互式标注
使用 GUI 客户端对单张图片进行交互式分析和标注。

### 场景 2: 批量处理
使用编程接口批量处理大量图片：

```python
from pathlib import Path
from service.client_app import InferenceClient

client = InferenceClient()
image_dir = Path("dataset/images")

for image_path in image_dir.glob("*.jpg"):
    result = client.predict(
        image_path=str(image_path),
        roi={"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}
    )
    # 处理结果...
```

### 场景 3: 集成到其他系统
将推理服务集成到 Web 应用、移动应用或其他系统中。

## ❓ 常见问题

### Q: 如何修改服务器端口？

A: 启动时指定端口：
```bash
python -m uvicorn service.app:app --port 8080
```

### Q: 客户端连接失败？

A: 检查：
1. 服务是否正在运行
2. 服务器地址是否正确
3. 防火墙是否阻止连接
4. 网络是否正常

### Q: 如何提高推理速度？

A: 
- 使用 GPU: `device="cuda"`
- 减小图像尺寸: `img_size=320`
- 提高置信度阈值: `conf_threshold=0.5`
- 使用更小的模型: `yolov8n-seg.pt`

### Q: 如何自定义检测类别颜色？

A: 在 `ResultRenderer.draw_results_on_image()` 中传入 `class_colors` 参数：

```python
custom_colors = {
    0: (255, 0, 0),    # 红色
    1: (0, 255, 0),    # 绿色
}
result_image = ResultRenderer.draw_results_on_image(
    image, result, class_colors=custom_colors
)
```

## 📝 开发指南

### 添加新功能

1. **新的 API 端点**: 在 `app.py` 中添加路由
2. **新的数据模型**: 在 `schemas.py` 中定义 Pydantic 模型
3. **客户端功能扩展**: 在 `client_app.py` 中添加 UI 组件

### 测试

```bash
# 测试 API
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d @test_request.json

# 运行客户端测试
python service/client_example.py
```

## 📚 相关文档

- [CLIENT_README.md](CLIENT_README.md) - 客户端详细使用指南
- [README_logger.md](README_logger.md) - 日志模块文档
- [../references/training_guide.md](../references/training_guide.md) - 训练指南
- [../notebooks/inference_example.ipynb](../notebooks/inference_example.ipynb) - 推理示例 Notebook

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

遵循项目主许可证。
