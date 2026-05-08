# SubStation AI 客户端应用 - 实现总结

## 📦 已创建的文件

### 核心文件

1. **`service/client_app.py`** (约450行)
   - 主客户端GUI应用程序
   - 包含以下核心类：
     - `InferenceClient`: HTTP客户端，调用推理API
     - `ROIDrawer`: ROI区域绘制交互处理器
     - `ResultRenderer`: 结果渲染器，在图片上绘制mask和bbox
     - `SubstationClientApp`: 主应用控制器

2. **`service/client_requirements.txt`**
   - 客户端依赖包列表
   - 包含: Pillow, requests, numpy

3. **`service/launch_client.py`**
   - 快速启动脚本
   - 自动检查依赖和服务状态

4. **`service/client_example.py`**
   - 编程方式使用客户端的示例代码
   - 包含4个示例场景

### 文档文件

5. **`service/CLIENT_README.md`**
   - 详细的客户端使用指南
   - 包含安装、使用步骤、常见问题等

6. **`service/README.md`**
   - service模块的综合说明文档
   - 涵盖服务端和客户端

### 测试文件

7. **`service/test_client.py`**
   - 完整的模块测试脚本
   - 测试导入、实例化、功能等

8. **`service/quick_test.py`**
   - 快速导入测试脚本

## ✨ 主要功能

### 1. 图形界面 (GUI)

基于 Tkinter 构建的用户友好界面：

```
┌─────────────────────────────────────────────┐
│ [服务器地址] [选择图片] [执行推理] [清除ROI]│
├─────────────────────────────────────────────┤
│ 推理参数配置面板                             │
├─────────────────────────────────────────────┤
│                                             │
│        图片显示与ROI绘制区域                 │
│        (鼠标拖拽划定ROI)                     │
│                                             │
├─────────────────────────────────────────────┤
│ 状态栏                                      │
└─────────────────────────────────────────────┘
```

### 2. 交互式 ROI 划定

- 鼠标左键拖拽绘制矩形区域
- 实时显示红色边框
- 自动验证ROI有效性（最小10x10像素）
- 支持清除和重绘

### 3. 推理参数配置

可调整的推理参数：
- **模型权重**: 自定义YOLO模型路径
- **置信度阈值**: 0.0-1.0 (默认0.25)
- **图像尺寸**: 320-1920 (默认640)
- **设备选择**: CPU 或 CUDA

### 4. 结果可视化

在原图上叠加显示：
- 🎨 **彩色半透明Mask填充**: 不同类别使用不同颜色
- 📐 **Mask轮廓线**: 精确的分割边界
- ⬜ **检测边界框**: 目标的矩形框
- 🏷️ **类别标签**: 显示类别ID和置信度

默认颜色映射：
- 类别0: 红色 (255, 0, 0)
- 类别1: 绿色 (0, 255, 0)
- 类别2: 蓝色 (0, 0, 255)
- 类别3: 黄色 (255, 255, 0)
- 类别4: 紫色 (255, 0, 255)
- 类别5: 青色 (0, 255, 255)

## 🚀 使用方法

### 方法一：GUI界面（推荐新手）

```bash
# 1. 确保推理服务正在运行
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000

# 2. 启动客户端（新终端）
cd service
python launch_client.py
# 或
python client_app.py
```

**操作步骤**：
1. 确认服务器地址（默认 http://localhost:8000）
2. 点击"选择图片"按钮加载图片
3. 在图片上拖拽鼠标划定ROI区域
4. 调整推理参数（可选）
5. 点击"执行推理"按钮
6. 查看可视化结果

### 方法二：编程方式（推荐开发者）

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

# 打印检测结果
for roi_result in result['results']:
    for detection in roi_result['detections']:
        print(f"类别: {detection['class_id']}, "
              f"置信度: {detection['confidence']:.2f}")
```

更多示例请查看 `client_example.py`。

## 📋 依赖安装

```bash
cd service
pip install -r client_requirements.txt
```

所需依赖：
- **Pillow** >= 9.0.0: 图像处理
- **requests** >= 2.28.0: HTTP请求
- **numpy** >= 1.19.0: 数值计算
- **tkinter**: Python标准库（通常无需安装）

如果 tkinter 缺失：
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install python3-tkinter
```

## 🔍 测试验证

```bash
# 运行完整测试
cd service
python test_client.py

# 或快速测试
python quick_test.py
```

测试内容包括：
- ✅ 依赖包版本检查
- ✅ 模块导入测试
- ✅ InferenceClient 实例化
- ✅ ResultRenderer 功能测试

## 📊 技术架构

### 数据流

```
用户操作 → GUI事件 → InferenceClient → HTTP POST /infer
                                            ↓
                                    FastAPI Service
                                            ↓
                                    YOLO Model
                                            ↓
                                   JSON Response
                                            ↓
                                  ResultRenderer
                                            ↓
                                Canvas显示结果图片
```

### 组件关系

```
SubstationClientApp (主控制器)
    ├── InferenceClient (API调用)
    ├── ROIDrawer (ROI交互)
    ├── ResultRenderer (结果绘制)
    └── Tkinter UI (界面组件)
```

## 🎯 应用场景

### 1. 单图交互式分析
使用GUI界面对单张图片进行交互式分析和标注。

### 2. 批量处理
```python
from pathlib import Path
from service.client_app import InferenceClient

client = InferenceClient()
for img_path in Path("dataset").glob("*.jpg"):
    result = client.predict(str(img_path), roi={...})
    # 处理结果...
```

### 3. 集成到其他系统
将 `InferenceClient` 集成到Web应用、数据处理pipeline等系统中。

## 💡 高级技巧

### 自定义类别颜色

```python
custom_colors = {
    0: (255, 0, 0),      # 绝缘子 - 红色
    1: (0, 255, 0),      # 导线 - 绿色
    2: (0, 0, 255),      # 塔架 - 蓝色
}

result_image = ResultRenderer.draw_results_on_image(
    image, result, class_colors=custom_colors
)
```

### 中文标签支持

```python
from PIL import ImageFont

# Windows
font = ImageFont.truetype("simhei.ttf", 16)
# Linux
# font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 16)

# 修改 ResultRenderer 中的文本绘制
draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
```

### 性能优化

```python
# GPU加速
client.predict(..., device="cuda")

# 减小图像尺寸
client.predict(..., img_size=320)

# 提高置信度阈值
client.predict(..., conf_threshold=0.5)
```

## ❓ 常见问题

### Q1: 连接被拒绝？
**A**: 确保推理服务正在运行：
```bash
python -m uvicorn service.app:app --port 8000
```

### Q2: ROI无法划定？
**A**: 确保：
- 图片已成功加载
- 鼠标在Canvas区域内
- ROI区域大于10x10像素

### Q3: 结果显示不完整？
**A**: 检查：
- 推理结果格式是否正确
- Mask轮廓点是否有效
- 图片尺寸是否匹配

### Q4: 如何提高FPS？
**A**: 
- 使用GPU (`device="cuda"`)
- 降低图像分辨率
- 减少检测类别数量
- 使用更小的模型 (yolov8n vs yolov8x)

## 📚 相关文档

- [CLIENT_README.md](CLIENT_README.md) - 详细使用指南
- [README.md](README.md) - 服务模块总览
- [client_example.py](client_example.py) - 编程示例
- [../references/training_guide.md](../references/training_guide.md) - 训练指南

## 🔄 后续改进方向

- [ ] 支持批量图片处理
- [ ] 添加结果保存功能
- [ ] 支持视频流实时推理
- [ ] 添加性能监控（FPS、延迟）
- [ ] 支持自定义标注工具
- [ ] 添加历史记录功能
- [ ] 支持导出标注结果（COCO、VOC格式）

## 📝 更新日志

### v1.0.0 (2026-05-08)
- ✅ 初始版本发布
- ✅ 实现GUI客户端应用
- ✅ 支持交互式ROI划定
- ✅ 实现结果可视化
- ✅ 提供编程接口
- ✅ 完整文档和示例

---

**作者**: SubStation AI Team  
**许可证**: 遵循项目主许可证  
**反馈**: 欢迎提交 Issue 和 Pull Request
