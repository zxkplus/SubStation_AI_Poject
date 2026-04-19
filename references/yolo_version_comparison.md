# YOLO版本对比与选择指南

## 支持的YOLO版本

本系统支持以下YOLO版本的实例分割训练：

- **YOLOv8** (推荐) - Ultralytics官方维护
- **YOLOv26** - Ultralytics最新版本
- **YOLOv6** - 美团开源

## 版本对比

### YOLOv8

**特点**：
- ✅ 最成熟稳定的版本
- ✅ 广泛的社区支持
- ✅ 丰富的文档和教程
- ✅ 良好的兼容性
- ✅ 支持多种部署格式

**模型尺寸**：
- `yolov8n-seg.pt` - Nano (最小)
- `yolov8s-seg.pt` - Small (平衡)
- `yolov8m-seg.pt` - Medium (更精确)
- `yolov8l-seg.pt` - Large (高精度)
- `yolov8x-seg.pt` - X-Large (最高精度)

**适用场景**：
- 生产环境部署
- 需要稳定性
- 团队协作项目
- 需要丰富文档

**训练命令**：
```bash
python scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --yolo_version yolov8 \
  --model_size s \
  --epochs 300
```

### YOLOv26

**特点**：
- ✅ Ultralytics最新版本
- ✅ 性能优化
- ✅ 更先进的架构
- ⚠️ 相对较新，社区经验较少
- ✅ 支持最新特性

**模型尺寸**：
- `yolo26n.pt` - Nano
- `yolo26s.pt` - Small
- `yolo26m.pt` - Medium
- `yolo26l.pt` - Large
- `yolo26x.pt` - X-Large

**适用场景**：
- 追求最新性能
- 实验性项目
- 需要最新特性
- 性能优化需求

**训练命令**：
```bash
python scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --yolo_version yolo26 \
  --model_size s \
  --epochs 300
```

### YOLOv6

**特点**：
- ✅ 美团开源
- ✅ 工业界验证
- ⚠️ 使用命令行接口（非Python API）
- ✅ 针对工业场景优化
- ⚠️ 安装相对复杂

**模型尺寸**：
- `yolov6n` - Nano
- `yolov6s` - Small
- `yolov6m` - Medium
- `yolov6l` - Large

**适用场景**：
- 工业界应用
- 需要稳定性
- 已有YOLOv6经验
- 批量训练需求

**训练命令**：
```bash
python scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --yolo_version yolov6 \
  --epochs 300
```

## 性能对比

### 速度与精度

| 版本 | 模型 | 参数量 | 速度 (FPS) | 精度 (mAP) |
|------|------|--------|------------|-----------|
| YOLOv8 | s | 11.2M | ~230 | ~48.6 |
| YOLOv26 | s | 12.0M | ~245 | ~49.2 |
| YOLOv6 | s | 12.0M | ~220 | ~48.0 |

*注：以上数据为参考值，实际性能取决于硬件和数据集*

### 训练速度

| 版本 | 训练速度 | 内存占用 | GPU需求 |
|------|----------|----------|---------|
| YOLOv8 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 6GB+ |
| YOLOv26 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB+ |
| YOLOv6 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB+ |

## 选择建议

### 场景1: 生产环境部署

**推荐**: YOLOv8

**原因**：
- 成熟稳定
- 广泛验证
- 丰富的部署支持
- 完善的文档

```bash
--yolo_version yolov8 --model_size m
```

### 场景2: 追求最佳性能

**推荐**: YOLOv26

**原因**：
- 最新优化
- 更高的精度和速度
- 先进的架构

```bash
--yolo_version yolo26 --model_size l
```

### 场景3: 资源受限环境

**推荐**: YOLOv8-Nano 或 YOLO26-Nano

```bash
--yolo_version yolov8 --model_size n
```

### 场景4: 快速原型开发

**推荐**: YOLOv8-Small

**原因**：
- 平衡的速度和精度
- 快速训练和验证
- 易于调试

```bash
--yolo_version yolov8 --model_size s
```

## API对比

### Ultralytics系列 (YOLOv8 / YOLOv26)

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8s-seg.pt')

# 训练
model.train(data='data.yaml', epochs=300)

# 验证
results = model.val()

# 推理
results = model.predict('image.jpg')

# 导出
model.export(format='onnx')
```

**优点**：
- 统一的Python API
- 易于集成
- 代码简洁

### YOLOv6

```python
from yolov6.train import train
from yolov6.deploy import export

# 训练（命令行风格）
train.run(...)

# 导出
export.run(...)
```

**优点**：
- 工业优化
- 批量处理

**缺点**：
- 接口复杂
- 命令行依赖

## 安装对比

### YOLOv8 / YOLOv26

```bash
pip install ultralytics>=8.0.0
```

### YOLOv6

```bash
pip install git+https://github.com/meituan/YOLOv6.git@main
```

## 迁移指南

### 从YOLOv8迁移到YOLOv26

**步骤**：
1. 修改`--yolo_version`参数
2. 更新模型名称（自动完成）
3. 其他参数保持不变

```bash
# 原命令
python scripts/train_yolo.py --yolo_version yolov8 --model_size s ...

# 修改为
python scripts/train_yolo.py --yolo_version yolo26 --model_size s ...
```

### 从YOLOv6迁移到YOLOv8

**注意事项**：
1. 模型权重不兼容（需要重新训练）
2. 训练参数可能需要微调
3. 验证输出格式略有不同

## 常见问题

### Q: 哪个版本精度最高？

A: 对于相同尺寸的模型，YOLOv26通常略高于YOLOv8，YOLOv8高于YOLOv6。但实际精度取决于数据集和训练参数。

### Q: 哪个版本训练最快？

A: YOLOv26通常训练速度最快，其次是YOLOv8。

### Q: 可以混用不同版本吗？

A: 不建议。每个版本的数据格式和权重格式不同，应该统一使用一个版本。

### Q: 如何选择模型尺寸？

A:
- 移动端/嵌入式：n (nano)
- 边缘设备：s (small)
- 服务器：m (medium)
- 高精度需求：l (large) 或 x (xlarge)

### Q: YOLOv26和YOLOv8的权重可以互换吗？

A: 不可以。它们是不同的模型架构，权重不兼容。

## 版本更新

### YOLOv8
- 最新版本: 8.0.x
- 更新频率: 定期更新
- 稳定性: 高

### YOLOv26
- 最新版本: 8.4.x
- 更新频率: 频繁更新
- 稳定性: 中

### YOLOv6
- 最新版本: 3.x
- 更新频率: 较慢
- 稳定性: 高

## 推荐配置

### 通用场景（推荐）
```bash
python scripts/train_yolo.py \
  --yolo_version yolov8 \
  --model_size s \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```

### 高精度场景
```bash
python scripts/train_yolo.py \
  --yolo_version yolov26 \
  --model_size l \
  --epochs 500 \
  --batch_size 16 \
  --img_size 1280
```

### 快速原型
```bash
python scripts/train_yolo.py \
  --yolo_version yolov8 \
  --model_size n \
  --epochs 100 \
  --batch_size 64 \
  --img_size 640
```

## 总结

| 需求 | 推荐版本 |
|------|----------|
| 生产环境 | YOLOv8 |
| 最佳性能 | YOLOv26 |
| 快速开发 | YOLOv8 |
| 资源受限 | YOLOv8-n |
| 工业界应用 | YOLOv6 |

**最终建议**：对于大多数应用场景，**YOLOv8-Small**是最佳选择，在速度、精度和稳定性之间取得了很好的平衡。
