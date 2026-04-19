# AttributeError: 'str' object has no attribute 'shape' 错误解决方案

## 错误描述

运行mask可视化脚本时出现以下错误：

```
File "/home/industai/anaconda3/envs/y8_env/lib/python3.10/site-packages/ultralytics/utils/ops.py", line 501, in process_mask
    c, mh, mw = protos.shape  # CHW
AttributeError: 'str' object has no attribute 'shape'
```

## 错误原因

这个错误发生在Ultralytics内部处理mask原型（prototypes）时。`protos`参数被错误地传递为字符串而不是张量。

**可能的原因**：
1. Ultralytics版本兼容性问题
2. 模型权重文件格式问题
3. 推理参数传递方式不正确

## 解决方案

### 方案1: 使用修复版脚本（推荐）

使用新创建的修复版脚本：

```bash
python scripts/validate_with_mask_fixed.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml \
  --output_dir ./validation_vis \
  --num_samples 10 \
  --conf 0.25
```

**修复内容**：
1. 使用更稳定的`model.predict()`方法
2. 添加多层异常处理
3. 提供多种推理方法的fallback

### 方案2: 使用简化测试脚本

先用简化脚本测试模型是否正常：

```bash
python scripts/test_mask_vis.py \
  /path/to/weights.pt \
  /path/to/data.yaml
```

这个脚本会：
1. 加载模型
2. 运行推理
3. 检查结果中是否有mask
4. 如果有mask，可视化并保存到`/tmp/mask_test_result.jpg`

### 方案3: 检查Ultralytics版本

检查当前安装的Ultralytics版本：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

如果版本较旧，升级到最新版本：

```bash
pip install --upgrade ultralytics
```

### 方案4: 使用不同的推理方式

在Python代码中尝试不同的推理方式：

```python
from ultralytics import YOLO

model = YOLO('weights.pt')

# 方式1: predict方法
results = model.predict('image.jpg', conf=0.25)

# 方式2: 直接调用
results = model('image.jpg', conf=0.25)

# 方式3: 使用source参数
results = model.predict(source='image.jpg', conf=0.25, verbose=False)
```

### 方案5: 检查模型权重

检查模型权重文件是否正确：

```python
from ultralytics import YOLO

try:
    model = YOLO('weights.pt')
    print(f"模型任务: {model.task}")
    print(f"模型设备: {model.device}")

    # 尝试推理
    results = model.predict('test_image.jpg')
    print(f"推理成功，结果数量: {len(results)}")

except Exception as e:
    print(f"模型加载或推理失败: {e}")
```

## 诊断流程

### 步骤1: 检查模型信息

```bash
python -c "from ultralytics import YOLO; model = YOLO('weights.pt'); print('Task:', model.task); print('Names:', model.names)"
```

**期望输出**：
```
Task: segment
Names: {0: 'class1', 1: 'class2', ...}
```

如果`Task`不是`segment`，说明模型被错误地加载为检测模型。

### 步骤2: 测试简化脚本

```bash
python scripts/test_mask_vis.py weights.pt data.yaml
```

检查输出中是否有：
- ✓ Mask可视化成功！
- 或 Masks: None（说明模型没有输出mask）

### 步骤3: 检查数据集配置

确保`data.yaml`中的配置正确：

```yaml
path: /path/to/dataset
train: train
val: val
nc: 10  # 类别数量
names:
  0: class1
  1: class2
  # ...
```

### 步骤4: 验证图片路径

确保验证集图片路径正确：

```bash
ls -la /path/to/dataset/val/images/*.jpg | head -5
```

## 常见问题

### Q1: 简化脚本显示"Masks: None"

**A**: 模型没有输出mask。

**可能原因**：
1. 模型被训练为检测模型而非分割模型
2. 置信度阈值太高
3. 模型权重损坏

**解决方案**：
1. 检查训练时的`task`参数是否为`segment`
2. 降低置信度阈值：`--conf 0.1`
3. 重新训练模型

### Q2: 简化脚本可以，但原脚本不行

**A**: 说明模型本身是正常的，是推理参数的问题。

**解决方案**：使用修复版脚本`validate_with_mask_fixed.py`

### Q3: 所有脚本都失败

**A**: 可能是Ultralytics版本问题或模型权重问题。

**解决方案**：
1. 升级Ultralytics：`pip install --upgrade ultralytics`
2. 重新训练模型
3. 检查模型权重文件是否完整

### Q4: AttributeError仍然出现

**A**: 这是Ultralytics内部的bug。

**临时解决方案**：
1. 使用简化脚本`test_mask_vis.py`
2. 或者手动可视化（见下方）

## 手动可视化方法

如果所有脚本都失败，可以手动编写简单的可视化代码：

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('weights.pt')
image = cv2.imread('test_image.jpg')

# 推理
results = model.predict(image, conf=0.25)

# 检查mask
if results and results[0].masks:
    mask = results[0].masks.data[0].cpu().numpy()

    # 可视化
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask_binary * 255

    vis_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite('mask_vis.jpg', vis_image)
    print("保存结果: mask_vis.jpg")
else:
    print("没有mask输出")
```

## 相关脚本

- `scripts/validate_with_mask.py` - 原始脚本（可能有问题）
- `scripts/validate_with_mask_fixed.py` - 修复版脚本（推荐）
- `scripts/test_mask_vis.py` - 简化测试脚本
- `scripts/diagnose_mask.py` - 模型诊断脚本

## 下一步

1. **先运行简化测试脚本**：确认模型是否正常输出mask
2. **如果mask正常**：使用修复版脚本进行批量可视化
3. **如果mask不正常**：检查训练配置，重新训练模型

## 技术细节

**错误位置**：
- `ultralytics/utils/ops.py`, line 501
- `process_mask` 函数

**问题**：
`protos`参数应该是形状为`[C, H, W]`的张量，但实际收到的是字符串。

**修复思路**：
1. 使用更稳定的`model.predict()`方法
2. 明确指定推理参数
3. 添加异常处理和fallback机制
