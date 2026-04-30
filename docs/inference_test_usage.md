# 推理测试脚本使用说明

推理测试脚本 (`scripts/inference_test.py`) 是一个用于处理单张图片或整个文件夹图片的工具，可以完成实例分割并展示结果。

## 功能特点

- 支持处理单张图片或整个文件夹的图片
- 完成实例分割任务并可视化结果
- 显示边界框、类别标签和分割轮廓
- 可以显示检测到的目标数量
- 支持自定义模型权重、设备、置信度阈值等参数

## 使用方法

### 处理单张图片

```bash
python scripts/inference_test.py \
    --weights /path/to/your/model.pt \
    --image /path/to/your/image.jpg \
    --device 0 \
    --conf-threshold 0.25 \
    --img-size 640 \
    --data-config /path/to/your/data.yaml
```

### 处理整个文件夹

```bash
python scripts/inference_test.py \
    --weights /path/to/your/model.pt \
    --folder /path/to/your/image/folder \
    --device 0 \
    --conf-threshold 0.25 \
    --img-size 640 \
    --data-config /path/to/your/data.yaml
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--weights` | str | 是 | - | 模型权重文件路径 |
| `--image` | str | 否 | - | 单张图片路径（与 `--folder` 二选一） |
| `--folder` | str | 否 | - | 图片文件夹路径（与 `--image` 二选一） |
| `--device` | str | 否 | "0" | 推理设备（如 "0" 表示第一张GPU，"cpu" 表示CPU） |
| `--conf-threshold` | float | 否 | 0.25 | 置信度阈值 |
| `--img-size` | int | 否 | 640 | 输入图像尺寸 |
| `--data-config` | str | 否 | - | 数据配置文件路径，用于加载类别名称 |

## 输出结果

脚本会为每张图片显示两个子图：

1. **原始图像** - 显示输入的原始图片
2. **分割结果** - 显示带有检测框和分割掩码的处理结果

在控制台中还会显示：

- 处理的图片名称
- 检测到的目标数量
- 每个目标的类别和置信度

## 注意事项

- 确保提供的权重文件路径正确且模型文件存在
- 支持的图片格式包括：JPG, JPEG, PNG, BMP, TIFF
- 如果提供了 `--data-config` 参数，类别将显示为配置文件中定义的名称，否则显示为类别ID
- 使用GPU时确保CUDA环境正确配置

## 示例

处理单张图片：

```bash
python scripts/inference_test.py \
    --weights runs/segment/runs/train/exp/weights/best.pt \
    --image test_images/sample.jpg \
    --device 0 \
    --conf-threshold 0.3
```

处理文件夹中的所有图片：

```bash
python scripts/inference_test.py \
    --weights runs/segment/runs/train/exp/weights/best.pt \
    --folder test_images/ \
    --device 0 \
    --conf-threshold 0.3 \
    --data-config runs/train/data.yaml
```