# SubStation AI 客户端应用使用指南

## 概述

这是一个基于 Tkinter 的图形界面客户端应用，用于调用 SubStation AI 推理服务进行实例分割。用户可以通过可视化界面选择图片、划定ROI区域、执行推理并查看结果。

## 功能特性

- ✅ 可视化图片选择与显示
- ✅ 交互式ROI区域划定（鼠标拖拽）
- ✅ 可配置的推理参数（模型权重、置信度阈值、图像尺寸、设备）
- ✅ 实时调用推理服务API
- ✅ 在原图上展示分割结果（mask轮廓、边界框、类别标签）

## 安装依赖

### 方法一：使用 pip

```bash
cd service
pip install -r client_requirements.txt
```

### 方法二：手动安装

```bash
pip install Pillow requests numpy
```

**注意**: `tkinter` 是 Python 标准库，通常无需单独安装。如果遇到导入错误：
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **CentOS/RHEL**: `sudo yum install python3-tkinter`
- **macOS**: 通常已内置
- **Windows**: 通常已内置

## 启动服务

在运行客户端之前，确保推理服务已经启动：

```bash
# 在项目根目录启动服务
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
```

或者使用已有的启动脚本：

```bash
python service/run.py
```

## 运行客户端

```bash
cd service
python client_app.py
```

## 使用步骤

### 1. 配置服务器地址

在顶部输入框中设置推理服务的地址，默认为 `http://localhost:8000`

### 2. 选择图片

点击 **"选择图片"** 按钮，从文件系统中选择一张图片（支持 JPG、PNG、BMP 格式）

### 3. 划定ROI区域

- 在图片上按住鼠标左键并拖拽，绘制矩形ROI区域
- ROI区域会以红色边框显示
- 如果绘制的区域太小（小于10x10像素），系统会提示重新划定
- 如需重新划定，点击 **"清除ROI"** 按钮

### 4. 配置推理参数

在"推理参数"面板中可以调整以下参数：

- **模型权重**: YOLO模型文件路径（如 `yolov8n-seg.pt`）
- **置信度阈值**: 检测置信度阈值（0.0-1.0，默认0.25）
- **图像尺寸**: 推理时的图像尺寸（320-1920，默认640）
- **设备**: 推理设备选择（cpu 或 cuda）

### 5. 执行推理

点击 **"执行推理"** 按钮，系统将：
1. 编码图片为base64格式
2. 发送POST请求到 `/infer` 接口
3. 接收推理结果
4. 在图片上绘制检测结果

### 6. 查看结果

推理完成后，原图上会显示：
- **彩色半透明mask区域**: 表示实例分割的掩码
- **彩色轮廓线**: mask的边界轮廓
- **边界框**: 检测目标的矩形框
- **标签**: 显示类别ID和置信度

不同类别使用不同颜色标识：
- 类别0: 红色
- 类别1: 绿色
- 类别2: 蓝色
- 类别3: 黄色
- 类别4: 紫色
- 类别5: 青色

## 界面说明

```
┌─────────────────────────────────────────────────────┐
│ 服务器地址: [http://localhost:8000] [选择图片]      │
│ [执行推理] [清除ROI]                                │
├─────────────────────────────────────────────────────┤
│ 推理参数                                            │
│ 模型权重: [yolov8n-seg.pt    ] 置信度阈值: [0.25]  │
│ 图像尺寸: [640              ] 设备:       [cpu  ▼] │
├─────────────────────────────────────────────────────┤
│                                                     │
│                  [图片显示区域]                      │
│           （在此处拖拽鼠标划定ROI）                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│ 状态: 已加载图片: example.jpg                       │
└─────────────────────────────────────────────────────┘
```

## 常见问题

### Q1: 点击"执行推理"后没有反应？

**A**: 检查以下几点：
1. 确认推理服务是否正在运行
2. 检查服务器地址是否正确
3. 查看终端是否有错误信息
4. 确认网络连接正常

### Q2: 推理结果显示错误？

**A**: 可能的原因：
1. 模型权重文件路径不正确
2. 图片格式不支持
3. ROI区域超出图片范围
4. 服务端返回错误响应

### Q3: 如何修改检测类别的颜色？

**A**: 编辑 `client_app.py` 中的 `ResultRenderer` 类，修改 `class_colors` 字典：

```python
class_colors = {
    0: (255, 0, 0),     # 红色
    1: (0, 255, 0),     # 绿色
    # ... 添加更多类别
}
```

### Q4: 中文标签显示乱码？

**A**: PIL默认字体不支持中文。需要加载中文字体：

```python
from PIL import ImageFont

# 加载中文字体
font = ImageFont.truetype("simhei.ttf", 16)  # Windows
# 或
font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 16)  # Linux

# 在绘制文本时指定字体
draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
```

### Q5: 如何提高推理速度？

**A**: 
1. 使用GPU加速：将设备设置为 `cuda`
2. 减小图像尺寸：降低 `img_size` 参数
3. 提高置信度阈值：减少检测目标数量
4. 使用更小的模型：如 `yolov8n-seg.pt` 而非 `yolov8x-seg.pt`

## 技术架构

### 核心组件

1. **InferenceClient**: HTTP客户端，负责调用推理API
2. **ROIDrawer**: ROI绘制交互处理器
3. **ResultRenderer**: 结果渲染器，在图片上绘制mask和bbox
4. **SubstationClientApp**: 主应用控制器，管理UI和业务流程

### 数据流

```
用户选择图片 → 加载并显示 → 划定ROI → 配置参数 → 执行推理 
    ↓
编码图片(base64) → POST /infer → 接收结果 → 绘制mask/bbox → 显示结果
```

## 扩展开发

### 添加自定义功能

1. **批量处理**: 修改 `_select_image` 支持多选，添加队列处理逻辑
2. **结果保存**: 添加保存按钮，将结果图片保存到文件
3. **历史记录**: 记录推理历史，支持回溯查看
4. **性能监控**: 显示推理耗时、FPS等指标

### API集成示例

```python
# 在其他Python脚本中使用InferenceClient
from service.client_app import InferenceClient

client = InferenceClient(base_url="http://localhost:8000")

result = client.predict(
    image_path="test.jpg",
    roi={"x1": 100, "y1": 100, "x2": 500, "y2": 500},
    weights_path="yolov8n-seg.pt",
    conf_threshold=0.25,
    img_size=640,
    device="cpu"
)

print(f"检测到 {len(result['results'][0]['detections'])} 个目标")
```

## 许可证

本项目遵循原项目许可证。

## 联系方式

如有问题或建议，请联系开发团队。
