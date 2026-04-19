# 修复说明：直接传递numpy数组

## 问题

之前使用 `str(image_path)` 传递图片路径给模型，导致：
1. Ultralytics内部路径解析可能出现问题
2. `protos` 参数被错误传递为字符串
3. 出现 `AttributeError: 'str' object has no attribute 'shape'` 错误

## 解决方案

### 修改内容

**1. 直接传递numpy数组**
```python
# 修改前
results = model.predict(str(image_path), conf=0.25)

# 修改后
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
results = model.predict(source=image_rgb, conf=0.25)
```

**2. 处理OpenCV通道问题**
```python
# OpenCV读取图片：BGR格式（Blue-Green-Red）
image = cv2.imread(image_path)

# 转换为RGB格式（Red-Green-Blue），Ultralytics期望RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 传递给模型
results = model.predict(source=image_rgb, ...)
```

### 优点

1. **避免路径解析问题**：直接传递图像数组，不需要文件路径
2. **避免字符串错误**：不会出现`protos`参数被错误传递为字符串的问题
3. **更好的控制**：可以在传递前对图像进行预处理
4. **更稳定**：不依赖于文件系统路径

### 修改的文件

1. `scripts/validate_with_mask.py`
2. `scripts/validate_with_mask_fixed.py`
3. `scripts/test_mask_vis.py`

### 关键代码示例

```python
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 使用cv2读取图片（BGR格式）
image = cv2.imread('test_image.jpg')

# 2. 转换为RGB格式（Ultralytics期望）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 直接传递numpy数组给模型
model = YOLO('weights.pt')
results = model.predict(source=image_rgb, conf=0.25)

# 4. 处理结果
for result in results:
    if result.masks:
        mask = result.masks.data[0].cpu().numpy()
        # 处理mask...
```

### OpenCV通道说明

**BGR vs RGB**：

```python
# OpenCV读取：BGR格式
image_bgr = cv2.imread('image.jpg')

# 转换为RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 转换回BGR（保存时需要）
image_bgr_again = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.jpg', image_bgr_again)
```

**为什么需要转换**：

- OpenCV：默认使用BGR格式（历史原因）
- Ultralytics：期望RGB格式（主流格式）
- matplotlib：期望RGB格式
- PIL/Pillow：期望RGB格式

### 验证方法

检查图像格式是否正确：

```python
import cv2
import numpy as np

image = cv2.imread('test.jpg')
print(f"图像形状: {image.shape}")  # (H, W, C)
print(f"图像类型: {image.dtype}")  # uint8

# 检查通道顺序
print(f"前3个像素值: {image[0, 0]}")  # BGR: [Blue, Green, Red]

# 转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"RGB前3个像素值: {image_rgb[0, 0]}")  # RGB: [Red, Green, Blue]
```

### 使用方法

**运行修复后的脚本**：

```bash
python scripts/validate_with_mask.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml \
  --output_dir ./validation_vis \
  --num_samples 10 \
  --conf 0.25
```

**或使用简化测试脚本**：

```bash
python scripts/test_mask_vis.py \
  ./runs/train/exp/weights/best.pt \
  ./yolo_dataset/data.yaml
```

### 注意事项

1. **内存占用**：直接传递numpy数组会增加内存占用（但通常可以忽略）
2. **预处理**：可以在传递前进行额外的预处理（如resize、normalize）
3. **通道一致性**：确保始终使用RGB格式传递给模型
4. **保存结果**：保存可视化结果时需要转换回BGR

### 性能对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| 传递路径字符串 | 内存占用小 | 可能出现路径解析错误 |
| 传递numpy数组 | 更稳定，避免路径问题 | 内存占用稍大 |

### 完整示例

```python
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def visualize_with_numpy_array(model, image_path, output_path):
    """使用numpy数组进行推理和可视化"""

    # 1. 读取图片（BGR）
    image_bgr = cv2.imread(str(image_path))

    # 2. 转换为RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. 推理（传递numpy数组）
    results = model.predict(source=image_rgb, conf=0.25)

    # 4. 可视化
    vis_image = image_bgr.copy()  # 使用BGR版本进行绘制

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (vis_image.shape[1], vis_image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                mask_colored = np.zeros_like(vis_image)
                mask_colored[:, :, 1] = mask_binary * 255  # 绿色
                vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)

    # 5. 保存（已经是BGR格式）
    cv2.imwrite(str(output_path), vis_image)

# 使用
model = YOLO('weights.pt')
visualize_with_numpy_array(model, 'test.jpg', 'output.jpg')
```

## 总结

通过直接传递numpy数组而不是文件路径，我们：
1. ✅ 解决了`AttributeError`错误
2. ✅ 避免了路径解析问题
3. ✅ 正确处理了OpenCV的BGR/RGB通道问题
4. ✅ 提高了代码的稳定性和可维护性
