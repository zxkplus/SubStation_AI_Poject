# Mask可视化问题排查指南

## 问题描述

训练完成后，在验证目录中生成的`val_batch1_pred.jpg`文件中：
- ✓ 可以看到准确的外接框（bounding box）
- ✗ 但是看不到mask的分割效果

## 可能的原因

### 原因1: Ultralytics默认验证可视化不显示mask

**最可能的原因**：Ultralytics框架的默认验证可视化主要关注检测框，对于实例分割的mask可能不会在`val_batch1_pred.jpg`中显示。

**验证方法**：
```bash
python scripts/diagnose_mask.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml
```

### 原因2: 模型任务类型错误

**可能原因**：模型被错误地训练为检测模型而非分割模型。

**验证方法**：运行诊断脚本，检查`model.task`是否为`segment`

## 解决方案

### 方案1: 使用自定义验证脚本（推荐）

使用专门设计的验证脚本进行mask可视化：

```bash
python scripts/validate_with_mask.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml \
  --output_dir ./validation_vis \
  --num_samples 10 \
  --conf 0.25
```

**参数说明**：
- `--weights`: 模型权重路径
- `--data`: 数据集配置文件路径
- `--output_dir`: 输出目录
- `--num_samples`: 可视化样本数量
- `--conf`: 置信度阈值

**输出**：
- 每张图片会生成一个可视化结果
- 同时显示边界框和mask
- Mask以半透明彩色叠加在原图上

### 方案2: 仅显示mask

如果只想看mask效果：

```bash
python scripts/validate_with_mask.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml \
  --output_dir ./validation_vis \
  --no_bbox \
  --mask_alpha 0.7
```

### 方案3: 调整验证可视化参数

在训练时添加更多可视化参数：

修改`train_configs/yolov26_seg_config.yaml`或直接在命令行添加：

```bash
python scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --yolo_version yolo26 \
  --model_size s \
  --plots true \
  --save true
```

## 诊断流程

### 步骤1: 运行诊断脚本

```bash
python scripts/diagnose_mask.py \
  /path/to/weights.pt \
  /path/to/data.yaml
```

**检查输出**：
1. 模型任务是否为`segment`？
2. 推理结果中是否有`masks`属性？
3. Mask数据是否正常？

### 步骤2: 检查模型权重

```bash
# 查看模型信息
python -c "from ultralytics import YOLO; model = YOLO('weights.pt'); print(model.task); print(model.names)"
```

### 步骤3: 手动推理测试

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('weights.pt')
results = model('test_image.jpg', verbose=False)

if results[0].masks is not None:
    print("✓ Mask输出正常")
    # 可视化第一个mask
    mask = results[0].masks.data[0].cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.savefig('mask_check.png')
else:
    print("✗ Mask输出异常")
```

## 常见问题

### Q1: 诊断脚本显示"未发现mask相关模块"

**A**: 模型可能被错误地训练为检测模型。

**解决**：
1. 检查训练时的`task`参数
2. 确保使用的是分割任务的预训练权重
3. 重新训练模型

### Q2: 诊断脚本显示"结果中有masks属性，但为None"

**A**: 模型推理时没有产生mask输出。

**可能原因**：
- 置信度阈值过高
- 输入图片格式不正确
- 模型权重损坏

**解决**：
1. 降低置信度阈值：`--conf 0.1`
2. 检查输入图片格式和尺寸
3. 重新训练模型

### Q3: Mask可见但效果很差

**A**: 可能是训练不充分或数据质量问题。

**解决**：
1. 增加训练轮数
2. 检查数据集标注质量
3. 调整学习率和数据增强参数
4. 使用更大的模型（如`--model_size m`或`l`）

### Q4: `val_batch1_pred.jpg`中仍然没有mask

**A**: 这是正常的，Ultralytics的默认验证可视化就是这样设计的。

**解决**：使用`validate_with_mask.py`脚本进行自定义可视化。

## 代码位置说明

### 需要检查的关键代码位置

1. **训练脚本**（`scripts/trainers/yolov26_trainer.py`）:
   - 第141行：`'task': 'segment'` - 确保设置为分割任务

2. **验证参数**（`scripts/trainers/yolov26_trainer.py`）:
   - 第239行：`val_args` - 验证参数配置

3. **数据集配置**（`data.yaml`）:
   - `nc`: 类别数量
   - `names`: 类别名称
   - `task`: 应该为`segment`

### Ultralytics内部代码

如果您想深入了解Ultralytics的验证可视化逻辑：

```python
# Ultralytics验证器位置
# ultralytics/engine/validator.py
# ultralytics/utils/plotting.py
```

但这些代码通常不需要修改，建议使用我们提供的自定义可视化脚本。

## 最佳实践

1. **训练前**：运行诊断脚本，确认预训练权重正确
2. **训练中**：定期检查训练日志，确认loss正常下降
3. **训练后**：使用`validate_with_mask.py`进行详细的mask可视化
4. **调试时**：使用`diagnose_mask.py`快速定位问题

## 相关脚本

- `scripts/validate_with_mask.py` - 自定义mask可视化脚本
- `scripts/diagnose_mask.py` - 模型诊断脚本
- `scripts/trainers/yolov26_trainer.py` - YOLO26训练器
