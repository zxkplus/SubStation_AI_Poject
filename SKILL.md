---
name: substation-segmentation-dataset
description: 变电站设备分割数据集管理与训练；支持YOLOv6/YOLOv8/YOLO26实例分割模型训练
dependency:
  python:
    - opencv-python>=4.5.0
    - numpy>=1.19.0
    - matplotlib>=3.3.0
    - pillow>=8.0.0
    - pyyaml>=5.4.0
---

# 变电站设备分割数据集管理 Skill

## 任务目标
- 本 Skill 用于：变电站设备分割数据集的统计分析与YOLO格式转换
- 能力包含：数据集加载、类别统计、Mask可视化、YOLO格式转换
- 触发条件：用户需要了解数据集分布、查看标注质量、准备训练数据时

## 前置准备

### 依赖安装
```bash
pip install opencv-python numpy matplotlib pillow pyyaml

# 如果需要训练YOLOv6
pip install git+https://github.com/meituan/YOLOv6.git@main

# 如果需要训练YOLO26（Ultralytics）
pip install ultralytics>=8.4.0
```

### 数据集目录结构要求
```
dataset_root/
├── category_1/
│   ├── image_1.jpg
│   ├── image_1.json
│   ├── image_2.jpg
│   └── image_2.json
├── category_2/
│   └── ...
└── category_n/
    └── ...
```

**数据格式规范**：
- 第一层目录：分割类别（如 `transformer`、`insulator`）
- 第二层目录：图片文件（支持 jpg/png/bmp）和对应的JSON标注文件
- JSON与图片名称一致（仅扩展名不同）

## 前置准备

### 环境安装

**方法1: 自动安装脚本（推荐）**
```bash
cd /workspace/projects/substation-segmentation-dataset
./scripts/install_conda_env.sh
```

**方法2: 快速安装**
```bash
cd /workspace/projects/substation-segmentation-dataset
./scripts/install_conda_env_quick.sh
```

**方法3: 使用requirements.txt**
```bash
# 创建conda环境
conda create -n substation_seg python=3.10 -y
conda activate substation_seg

# 安装PyTorch（根据GPU情况选择）
# 有GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 仅CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

详细安装说明请参考 [references/install_guide.md](references/install_guide.md)

### 依赖安装
```bash
# 核心依赖（已在安装脚本中包含）
pip install opencv-python numpy matplotlib pillow pyyaml

# YOLO26（Ultralytics）
pip install ultralytics>=8.4.0

# YOLOv6（可选）
pip install git+https://github.com/meituan/YOLOv6.git@main
```

### 数据集目录结构要求

### 标准流程

1. **统计数据集分布**
   ```bash
   python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
     --dataset_path /path/to/dataset \
     --mode stats
   ```
   - 智能体会调用 `scripts/statistics.py` 计算各类别样本数量
   - 生成统计报告并在终端输出

2. **可视化标注示例**
   ```bash
   python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
     --dataset_path /path/to/dataset \
     --mode visualize \
     --samples_per_class 3
   ```
   - 智能体会调用 `scripts/visualization.py` 加载标注
   - 在每个类别中随机选择指定数量的样本
   - 将mask叠加到原图并通过弹窗展示

3. **完整分析（统计+可视化）**
   ```bash
   python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
     --dataset_path /path/to/dataset \
     --mode full \
     --samples_per_class 2
   ```

### 可选分支

- **当仅查看统计结果**：使用 `--mode stats` 跳过可视化
- **当需要自定义采样数量**：调整 `--samples_per_class` 参数
- **当数据集较大**：建议先运行 `stats` 模式快速预览，再进行可视化
- **当训练后检查mask效果**：使用mask可视化脚本（详见下方）
- **当需要YOLO格式训练数据**：使用 `--mode yolo` 进行转换（详见下方）

### YOLO格式转换

将分割标注数据转换为YOLO目标检测格式：

```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path /path/to/dataset \
  --mode yolo \
  --output_yolo_path /path/to/yolo_output \
  --expand_ratio 0.0 \
  --min_size 32
```

**转换流程**：
1. 根据每个mask的外接矩形裁剪原图
2. 将标注数据坐标系转换为裁剪后小图的坐标
3. 生成符合YOLO Polygon规范的标注文件（每行：`class_id x1 y1 x2 y2 x3 y3 ...`）
4. 自动生成类别映射文件 `classes.txt`
5. 统计各类别图像尺寸分布（最小、最大、平均值）

**可选参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_yolo_path` | YOLO输出目录 | 输入目录的 `yolo_output` |
| `--samples_per_class` | 每类采样数量 | 全部转换 |
| `--expand_ratio` | 边界框扩展比例 | 0.0 |
| `--min_size` | 最小裁剪尺寸(像素) | 32 |
| `--num_workers` | 并行线程数 | 8 |

## 资源索引

- **核心脚本**：
  - [scripts/data_loader.py](scripts/data_loader.py) - 数据加载器，遍历目录结构并加载图片与标注
  - [scripts/statistics.py](scripts/statistics.py) - 统计模块，计算类别分布和数据集信息
  - [scripts/visualization.py](scripts/visualization.py) - 可视化模块，处理mask叠加和图片输出
  - [scripts/yolo_converter.py](scripts/yolo_converter.py) - YOLO格式转换器，支持多线程并行转换，保留polygon轮廓信息，统计图像尺寸分布
  - [scripts/yolo_validator.py](scripts/yolo_validator.py) - YOLO数据集验证器，绘制mask验证转换逻辑
  - [scripts/train_yolo.py](scripts/train_yolo.py) - YOLO训练统一入口，支持YOLOv6/YOLOv8/YOLO26版本
  - [scripts/validate_with_mask.py](scripts/validate_with_mask.py) - Mask可视化验证脚本，专门用于检查mask预测效果
  - [scripts/diagnose_mask.py](scripts/diagnose_mask.py) - 模型诊断脚本，排查mask输出问题
  - [scripts/test_mask_vis.py](scripts/test_mask_vis.py) - 简化版mask可视化测试脚本
  - [scripts/trainers/base_trainer.py](scripts/trainers/base_trainer.py) - 基础训练器接口
  - [scripts/trainers/yolov6_trainer.py](scripts/trainers/yolov6_trainer.py) - YOLOv6实例分割训练器实现
  - [scripts/trainers/yolov8_trainer.py](scripts/trainers/yolov8_trainer.py) - Ultralytics YOLOv8实例分割训练器实现
  - [scripts/trainers/yolov26_trainer.py](scripts/trainers/yolov26_trainer.py) - Ultralytics YOLO26实例分割训练器实现
  - [scripts/main.py](scripts/main.py) - 主入口，整合所有功能并提供命令行接口

- **参考文档**：
  - [references/data_format.md](references/data_format.md) - JSON标注格式规范与YOLO格式说明
  - [references/training_guide.md](references/training_guide.md) - YOLO训练完整指南
  - [references/install_guide.md](references/install_guide.md) - Conda环境安装指南
  - [references/mask_visualization_guide.md](references/mask_visualization_guide.md) - Mask可视化问题排查指南

- **训练配置**：
  - [train_configs/data_template.yaml](train_configs/data_template.yaml) - 数据集配置模板
  - [train_configs/yolov6_seg_config.yaml](train_configs/yolov6_seg_config.yaml) - YOLOv6实例分割配置
  - [train_configs/yolov8_seg_config.yaml](train_configs/yolov8_seg_config.yaml) - YOLOv8实例分割配置
  - [train_configs/yolov26_seg_config.yaml](train_configs/yolov26_seg_config.yaml) - YOLO26实例分割配置

## 模块扩展指南

### 扩展新功能

1. **添加新的统计维度**：
   - 在 `scripts/statistics.py` 的 `DatasetStats` 类中添加新方法
   - 例如：图片尺寸分布、mask面积统计、标注完整性检查

2. **支持新的标注格式**：
   - 在 `scripts/data_loader.py` 中扩展JSON解析逻辑
   - 参考 [references/data_format.md](references/data_format.md) 中的格式定义

3. **添加新的可视化方式**：
   - 在 `scripts/visualization.py` 中扩展 `MaskVisualizer` 类
   - 例如：添加轮廓绘制、热力图、对比视图

### 模块化设计原则

- **数据加载器**（data_loader.py）：只负责文件IO和数据结构化，不处理统计或可视化
- **统计模块**（statistics.py）：只处理数值计算和报告生成，不涉及图像操作
- **可视化模块**（visualization.py）：只处理图像渲染和输出，不依赖统计逻辑
- **YOLO转换器**（yolo_converter.py）：独立的数据格式转换模块，不依赖其他模块
- **主入口**（main.py）：协调各模块，处理命令行参数，组织执行流程

## 注意事项

- 确保数据集目录结构符合要求，否则会触发错误提示
- JSON文件必须与对应的图片文件同名（仅扩展名不同）
- 可视化输出需要GUI环境，在服务器环境中会保存为图片文件
- 对于大型数据集，建议先运行统计模式了解数据分布
- YOLO转换会跳过小于 `min_size` 的目标，可根据实际需求调整
- `expand_ratio` 用于增加裁剪区域，避免目标被裁剪边缘截断

## 使用示例

### 示例1：快速统计数据集
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path ./datasets/substation \
  --mode stats
```
**输出**：各类别图片数量统计表

### 示例2：查看标注质量
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path ./datasets/substation \
  --mode visualize \
  --samples_per_class 5
```
**输出**：弹窗展示每类5张样本的mask叠加效果

### 示例3：完整分析
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path ./datasets/substation \
  --mode full \
  --samples_per_class 3
```
**输出**：统计报告 + 可视化图片

### 示例4：转换为YOLO格式
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path ./datasets/substation \
  --mode yolo \
  --output_yolo_path ./yolo_dataset
```
**输出**：YOLO格式数据集（images/、labels/、classes.txt）

### 示例5：带边界扩展的YOLO转换
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/main.py \
  --dataset_path ./datasets/substation \
  --mode yolo \
  --output_yolo_path ./yolo_dataset \
  --expand_ratio 0.1 \
  --min_size 64
```
**说明**：边界框扩展10%，跳过小于64像素的目标

### 示例6：验证YOLO数据集
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/yolo_validator.py \
  --yolo_path ./yolo_dataset \
  --samples_per_class 10 \
  --output_path ./validation_output
```
**说明**：每类随机选择10张图片，将YOLO标注转换为mask并以半透明彩色叠加，每张图片单独保存到对应类别子目录

### 示例7：训练YOLOv8实例分割模型（推荐）
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolov8 \
  --model_size s \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```
**说明**：使用Ultralytics YOLOv8训练实例分割模型，支持n/s/m/l/x五种尺寸

### 示例8：训练YOLO26实例分割模型
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolo26 \
  --model_size s \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```
**说明**：使用Ultralytics YOLO26训练实例分割模型，支持n/s/m/l/x五种尺寸

### 示例9：训练后检查Mask效果
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/validate_with_mask.py \
  --weights ./runs/train/exp/weights/best.pt \
  --data ./yolo_dataset/data.yaml \
  --output_dir ./validation_vis \
  --num_samples 10 \
  --conf 0.25
```
**说明**：专门用于可视化mask预测效果，解决默认验证图中不显示mask的问题

### 示例10：诊断模型Mask输出
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/diagnose_mask.py \
  /path/to/weights.pt \
  /path/to/data.yaml
```
**说明**：诊断模型是否正确输出mask，排查mask不可见问题

### 示例11：训练YOLOv6实例分割模型
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --dataset_path ./yolo_dataset \
  --output_dir ./runs/train \
  --yolo_version yolov6 \
  --epochs 300 \
  --batch_size 32 \
  --img_size 640
```
**说明**：使用YOLOv6训练实例分割模型，自动划分训练集和验证集

### 示例12：验证训练好的模型
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --mode val \
  --dataset_path ./yolo_dataset \
  --weights ./runs/train/exp/weights/best.pt
```
**说明**：使用验证集评估模型性能

### 示例13：导出模型为ONNX格式
```bash
python /workspace/projects/substation-segmentation-dataset/scripts/train_yolo.py \
  --mode export \
  --weights ./runs/train/exp/weights/best.pt \
  --export_format onnx
```
**说明**：将训练好的模型导出为ONNX格式，便于部署
