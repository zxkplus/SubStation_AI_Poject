# YOLO训练使用指南

## 概述

本Skill提供了完整的YOLO实例分割训练流程，支持数据集转换、模型训练、验证、测试和导出。

## 支持的YOLO版本

- **YOLO26** (推荐): Ultralytics框架，使用Python API，支持n/s/m/l/x五种尺寸
- **YOLOv6**: 美团开源，使用命令行接口，支持实例分割

## 完整训练流程

### 第一步：准备数据集

确保你的数据集已经转换为YOLO格式：

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path /path/to/original/dataset \
  --mode yolo \
  --output_yolo_path /path/to/yolo_dataset \
  --expand_ratio 0.1 \
  --min_size 64 \
  --num_workers 8
```

输出目录结构：
```
yolo_dataset/
├── classes.txt          # 类别映射文件
├── train/               # 训练集（自动生成）
│   ├── images/
│   └── labels/
├── val/                 # 验证集（自动生成）
│   ├── images/
│   └── labels/
├── category1/           # 原始类别数据
│   ├── images/
│   └── labels/
└── category2/
    ├── images/
    └── labels/
```

### 第二步：验证数据集

在训练前验证数据集的标注质量：

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/yolo_validator.py \
  --yolo_path /path/to/yolo_dataset \
  --samples_per_class 10 \
  --output_path /path/to/validation_output
```

### 第三步：训练模型

#### 训练YOLO26（推荐）

**基础训练（使用yolo26s模型）**：
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolo26 \
  --model_size s \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```

**使用不同模型尺寸**：
```bash
# nano模型（最小最快）
python scripts/train_yolo.py --yolo_version yolo26 --model_size n ...

# medium模型（更高精度）
python scripts/train_yolo.py --yolo_version yolo26 --model_size m ...

# large模型（高精度）
python scripts/train_yolo.py --yolo_version yolo26 --model_size l ...
```

#### 训练YOLOv6

**基础训练**：
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolov6 \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```

#### 高级训练选项

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolo26 \
  --model_size s \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640 \
  --device 0 \
  --workers 8 \
  --train_ratio 0.8 \
  --name exp001
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | YOLO格式数据集路径 | 必需 |
| `--output_dir` | 输出目录 | `./runs/train` |
| `--yolo_version` | YOLO版本 | `yolov6` |
| `--model_size` | YOLO26模型尺寸 (n/s/m/l/x) | `s` |
| `--epochs` | 训练轮数 | 300 |
| `--batch_size` | 批次大小 | 32 |
| `--img_size` | 输入图片尺寸 | 640 |
| `--device` | 设备ID (0=GPU0, -1=CPU) | 0 |
| `--workers` | 数据加载线程数 | 8 |
| `--train_ratio` | 训练集比例 | 0.8 |
| `--name` | 实验名称 | `exp` |

#### 恢复训练

如果训练中断，可以从checkpoint恢复：

**YOLO26**：
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolo26 \
  --resume ./runs/train/exp/weights/last.pt
```

**YOLOv6**：
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolov6 \
  --resume ./runs/train/exp/weights/last_ckpt.pt
```

### 第四步：验证模型

训练完成后，使用验证集评估模型：

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --mode val \
  --dataset_path /path/to/yolo_dataset \
  --weights ./runs/train/exp/weights/best.pt \
  --batch_size 32 \
  --img_size 640
```

### 第五步：测试模型

使用测试集评估最终性能：

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --mode test \
  --dataset_path /path/to/yolo_dataset \
  --weights ./runs/train/exp/weights/best.pt \
  --batch_size 32 \
  --img_size 640
```

### 第六步：导出模型

将训练好的模型导出为部署格式：

#### 导出为ONNX格式

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --mode export \
  --weights ./runs/train/exp/weights/best.pt \
  --export_format onnx \
  --img_size 640
```

#### 导出为其他格式

**TorchScript**：
```bash
python scripts/train_yolo.py --mode export --weights best.pt --export_format torchscript
```

**TensorRT (engine)**：
```bash
python scripts/train_yolo.py --mode export --weights best.pt --export_format engine
```

**CoreML**：
```bash
python scripts/train_yolo.py --mode export --weights best.pt --export_format coreml
```

**TFLite**：
```bash
python scripts/train_yolo.py --mode export --weights best.pt --export_format tflite
```

## 输出文件说明

训练完成后，输出目录结构：

```
runs/train/exp/
├── weights/              # 模型权重
│   ├── best_ckpt.pt      # 最佳模型（验证集最高mAP）
│   └── last_ckpt.pt      # 最后一个epoch的模型
├── results.png           # 训练曲线图
├── results.txt           # 训练结果文本
└── args.yaml             # 训练参数配置
```

## 训练技巧

### 1. 调整批次大小

根据GPU显存调整batch_size：

```bash
# 8GB显存
--batch_size 16

# 16GB显存
--batch_size 32

# 24GB显存
--batch_size 64
```

### 2. 使用多GPU训练

使用CUDA_VISIBLE_DEVICES指定GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_yolo.py \
  --dataset_path /path/to/yolo_dataset \
  --device 0,1 \
  --batch_size 64
```

### 3. 学习率调整

可以通过修改配置文件调整学习率：

```yaml
# train_configs/yolov6_seg_config.yaml
training:
  optimizer: 'AdamW'
  lr: 0.001  # 调整学习率
```

### 4. 数据增强

调整数据增强参数：

```yaml
# train_configs/data_template.yaml
mosaic: 1.0
mixup: 0.0
flipud: 0.0
fliplr: 0.5
```

### 5. 训练监控

训练过程中会实时显示：
- 损失值（box_loss, cls_loss, mask_loss）
- mAP指标
- 学习率变化
- 训练速度

## 常见问题

### Q1: 训练时显存不足

**A**: 减小batch_size或img_size

```bash
--batch_size 16 --img_size 512
```

### Q2: 训练速度慢

**A**: 增加workers数量

```bash
--workers 16
```

### Q3: 模型不收敛

**A**:
1. 检查数据集质量
2. 调整学习率
3. 增加训练轮数
4. 使用预训练权重

### Q4: 如何切换YOLO版本

**A**: 修改--yolo_version参数

```bash
# 当前支持
--yolo_version yolov6

# 未来支持
--yolo_version yolov8
```

### Q5: 自定义配置文件

**A**: 使用--data_config和--model_config参数

```bash
python scripts/train_yolo.py \
  --data_config ./custom_data.yaml \
  --model_config ./custom_config.yaml \
  --dataset_path /path/to/dataset
```

## 扩展支持其他YOLO版本

如需支持其他YOLO版本（如YOLOv8），参考以下步骤：

1. 在`scripts/trainers/`目录下创建新的训练器类

```python
# scripts/trainers/yolov8_trainer.py
from .base_trainer import BaseTrainer

class YOLOv8Trainer(BaseTrainer):
    def train(self, **kwargs):
        # 实现YOLOv8训练逻辑
        pass

    def validate(self, **kwargs):
        # 实现YOLOv8验证逻辑
        pass

    # 实现其他方法...
```

2. 在`scripts/train_yolo.py`中注册新版本

```python
TRAINER_REGISTRY = {
    'yolov6': YOLOv6Trainer,
    'yolov8': YOLOv8Trainer,  # 添加新版本
}
```

3. 更新SKILL.md文档

## 参考资源

- [YOLOv6官方文档](https://github.com/meituan/YOLOv6)
- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [数据集格式规范](./data_format.md)
