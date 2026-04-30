# 数据处理模式说明

本项目提供两种数据处理模式：`convert` 和 `yolo`，它们具有不同的功能和用途。

## 模式对比

### `convert` 模式

**目的**：将原始数据集中的每个对象单独裁剪成图片，保持JSON格式标注。

**功能**：
- 根据每个polygon的外接矩形裁剪图片
- 保留mask轮廓信息，坐标变换适应裁剪后图片
- 生成JSON格式标注文件（保持原有分割轮廓信息）
- 处理相交的polygon，确保相交部分也被保留
- 多线程并行处理加速转换

**输入**：原始数据集（图片+JSON标注）

**输出**：
- 裁剪后的图片
- JSON格式标注文件（与原始标注格式相同）

**适用场景**：
- 数据预处理，准备特定任务的裁剪数据
- 需要保留原始分割轮廓信息
- 对单个对象进行独立分析

**命令示例**：
```bash
python scripts/main.py \
  --dataset_path /path/to/dataset \
  --mode convert \
  --output_yolo_path /path/to/convert_output \
  --expand_ratio 0.1 \
  --min_size 64 \
  --num_workers 8
```

### `yolo` 模式

**目的**：将标注格式转换为YOLO标准格式，用于YOLO模型训练。

**功能**：
- 将JSON格式转换为YOLO格式（.txt文件）
- 生成类别映射文件（classes.txt）
- 数据集划分（训练/验证/测试集）
- 生成YOLO配置文件（data.yaml）
- 将polygon坐标转换为YOLO格式（归一化坐标）
- 多线程并行处理加速转换

**输入**：原始数据集（图片+JSON标注）

**输出**：
- YOLO格式标注文件（.txt格式）
- 训练/验证/测试数据集划分
- YOLO配置文件（data.yaml）
- 类别映射文件（classes.txt）

**适用场景**：
- 准备用于YOLO模型训练的标准数据集
- 需要标准化的数据集格式
- 模型训练前的数据准备

**命令示例**：
```bash
python scripts/main.py \
  --dataset_path /path/to/dataset \
  --mode yolo \
  --output_yolo_path /path/to/yolo_output \
  --num_workers 8
```

## 参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | 输入数据集路径 | 必需 |
| `--mode` | 运行模式（convert/yolo/stats/visualize/full） | full |
| `--num_workers` | 并行处理的线程数 | 8 |

### `convert` 模式特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--expand_ratio` | 边界框扩展比例（0-1），增加裁剪区域 | 0.0 |
| `--min_size` | 最小裁剪尺寸，低于此尺寸的目标将被跳过 | 32 |
| `--samples_per_class` | 每个类别采样的样本数量 | 100 |

### `yolo` 模式特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_ratio` | 训练集比例 | 0.8 |
| `--val_ratio` | 验证集比例 | 0.1 |

## 输出目录结构

### `convert` 模式输出

```
convert_output/
├── category1/           # 按类别分目录
│   ├── image_001_1.jpg  # 裁剪图片（第一个polygon）
│   ├── image_001_2.jpg  # 裁剪图片（第二个polygon）
│   ├── image_001_1.json # JSON标注文件
│   └── image_001_2.json # JSON标注文件
├── category2/           # 另一个类别
│   ├── ...
└── ...
```

### `yolo` 模式输出

```
yolo_output/
├── classes.txt          # 类别映射文件
├── data.yaml            # YOLO数据集配置文件
├── images/
│   ├── train/           # 训练集图片
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/             # 验证集图片
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/            # 测试集图片
│       ├── img3.jpg
│       └── ...
└── labels/
    ├── train/           # 训练集标签
    │   ├── img1.txt
    │   └── ...
    ├── val/             # 验证集标签
    │   ├── img2.txt
    │   └── ...
    └── test/            # 测试集标签
        ├── img3.txt
        └── ...
```

## 使用建议

1. **`convert` 模式**：当你需要对单个对象进行独立处理，或者需要保持原始JSON格式标注时使用。

2. **`yolo` 模式**：当你准备用于YOLO模型训练的数据集时使用，它会生成YOLO训练所需的标准格式。

3. **处理相交对象**：在`convert`模式中，如果多个polygon在裁剪区域相交，这些相交部分会被保留在同一个裁剪图像中。

4. **性能考虑**：两个模式都支持多线程处理，可以根据CPU核心数调整`--num_workers`参数。

## 常见问题

**Q: 何时使用`convert`模式？**
A: 当你需要对每个对象单独裁剪并保留原始JSON格式标注时，例如进行对象级别的分析或预处理。

**Q: 何时使用`yolo`模式？**
A: 当你准备训练YOLO模型时，它会生成YOLO训练所需的标准格式数据集。

**Q: 两个模式可以链式使用吗？**
A: 不建议链式使用，每个模式都有其特定的用途和输出格式。