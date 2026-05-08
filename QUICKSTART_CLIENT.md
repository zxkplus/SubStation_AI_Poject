# SubStation AI 客户端应用 - 快速开始

## 🎯 功能概述

新增了一个图形界面客户端应用，可以：
1. ✅ 选择并显示图片
2. ✅ 在图片上交互式划定ROI区域
3. ✅ 调用推理服务进行实例分割
4. ✅ 在原图上可视化展示分割结果（mask轮廓、边界框）

## 🚀 快速启动（3步）

### 步骤1: 安装客户端依赖

```bash
cd service
pip install -r client_requirements.txt
```

### 步骤2: 启动推理服务

```bash
# 在项目根目录执行
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
```

保持此终端运行。

### 步骤3: 启动客户端

打开**新终端**：

```bash
cd service
python launch_client.py
```

或

```bash
python client_app.py
```

## 📖 使用流程

1. **配置服务器地址**: 默认为 `http://localhost:8000`
2. **选择图片**: 点击"选择图片"按钮
3. **划定ROI**: 在图片上拖拽鼠标绘制矩形区域
4. **调整参数** (可选): 模型权重、置信度阈值等
5. **执行推理**: 点击"执行推理"按钮
6. **查看结果**: 自动在原图上显示检测结果

## 💻 编程方式使用

```python
from service.client_app import InferenceClient, ResultRenderer
from PIL import Image

# 创建客户端
client = InferenceClient("http://localhost:8000")

# 执行推理
result = client.predict(
    image_path="test.jpg",
    roi={"x1": 100, "y1": 100, "x2": 500, "y2": 500},
    weights_path="yolov8n-seg.pt",
    conf_threshold=0.25,
    img_size=640,
    device="cpu"
)

# 可视化
image = Image.open("test.jpg")
result_image = ResultRenderer.draw_results_on_image(image, result)
result_image.save("result.jpg")
```

## 📁 相关文件

- `service/client_app.py` - 主客户端应用
- `service/launch_client.py` - 快速启动脚本
- `service/client_example.py` - 编程示例
- `service/CLIENT_README.md` - 详细文档
- `service/IMPLEMENTATION_SUMMARY.md` - 实现总结

## ❓ 遇到问题？

查看详细文档：[service/CLIENT_README.md](service/CLIENT_README.md)

或运行测试验证：
```bash
cd service
python test_client.py
```
