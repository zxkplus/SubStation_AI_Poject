# JSON标注格式规范

## 目录结构

```
dataset_root/
├── transformer/              # 变压器类别
│   ├── image_001.jpg
│   ├── image_001.json
│   ├── image_002.jpg
│   └── image_002.json
├── insulator/                # 绝缘子类别
│   └── ...
└── breaker/                  # 断路器类别
    └── ...
```

## 文件命名规则

- 图片文件和JSON标注文件必须同名（仅扩展名不同）
- 支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 标注文件格式：`.json`

## JSON标注格式

本工具支持以下常见的分割标注格式：

### 格式1: COCO格式

```json
{
  "image_width": 640,
  "image_height": 480,
  "segmentation": [
    [[x1, y1, x2, y2, x3, y3, ...]],  // 多边形1
    [[x1, y1, x2, y2, ...]]           // 多边形2（可选）
  ]
}
```

**字段说明**：
- `image_width`: 图像宽度（像素）
- `image_height`: 图像高度（像素）
- `segmentation`: 分割多边形列表，每个多边形为 [x1, y1, x2, y2, ...] 格式

### 格式2: LabelMe格式

```json
{
  "imageWidth": 640,
  "imageHeight": 480,
  "shapes": [
    {
      "label": "transformer",
      "points": [[x1, y1], [x2, y2], [x3, y3]],
      "shape_type": "polygon"
    }
  ]
}
```

**字段说明**：
- `imageWidth`: 图像宽度
- `imageHeight`: 图像高度
- `shapes`: 标注形状列表
- `label`: 标注类别名称
- `points`: 多边形顶点坐标列表 [[x1, y1], [x2, y2], ...]
- `shape_type`: 形状类型，支持 "polygon"

### 格式3: 直接Mask格式

```json
{
  "image_width": 640,
  "image_height": 480,
  "mask": [
    [0, 0, 1, 1, ...],
    [0, 1, 1, 1, ...],
    ...
  ]
}
```

**字段说明**：
- `mask`: 二值mask数组，0表示背景，1表示前景

## 坐标系统

- **原点**: 图像左上角
- **X轴**: 从左到右
- **Y轴**: 从上到下
- **单位**: 像素

## 完整示例

### 示例1: 单个多边形标注

```json
{
  "image_width": 640,
  "image_height": 480,
  "segmentation": [
    [100, 100, 200, 100, 200, 200, 100, 200]
  ]
}
```

对应的图片：`image_001.jpg`

### 示例2: 多个多边形标注

```json
{
  "image_width": 640,
  "image_height": 480,
  "shapes": [
    {
      "label": "transformer_body",
      "points": [[50, 50], [150, 50], [150, 150], [50, 150]],
      "shape_type": "polygon"
    },
    {
      "label": "transformer_top",
      "points": [[100, 20], [130, 50], [70, 50]],
      "shape_type": "polygon"
    }
  ]
}
```

## 扩展性说明

本工具支持以下扩展：

1. **添加新的标注格式**：
   - 在 `scripts/data_loader.py` 中修改 `parse_json_mask` 方法
   - 添加对新格式的解析逻辑

2. **自定义mask颜色**：
   - 在 `scripts/visualization.py` 中修改 `COLORS` 列表

3. **调整可视化透明度**：
   - 在调用 `MaskVisualizer` 时传入 `alpha` 参数（0-1之间）

## YOLO格式说明

当使用 `--mode yolo` 进行数据集转换时，工具会生成符合YOLO格式的数据集。

### 输出目录结构

默认情况下，转换器会按原始类别分文件夹保存：

```
yolo_output/
├── transformer/           # 按类别分目录
│   ├── images/            # 该类别的裁剪图片
│   │   ├── image_001_1.jpg
│   │   └── ...
│   └── labels/            # 该类别的YOLO标注文件
│       ├── image_001_1.txt
│       └── ...
├── insulator/            # 另一个类别
│   ├── images/
│   └── labels/
├── breaker/              # 更多类别...
│   ├── images/
│   └── labels/
└── classes.txt          # 类别映射文件（全局）
```

### YOLO标注格式

每张裁剪图片对应一个同名的 `.txt` 文件，支持两种格式：

#### 格式1: YOLO Polygon格式（推荐，保留轮廓信息）

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

- `class_id`: 类别ID（整数，从0开始）
- `x1, y1, x2, y2, ...`: 多边形顶点坐标（归一化到0-1）

示例：
```
0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9
```

#### 格式2: YOLO Bounding Box格式（兼容性）

```
<class_id> <x_center> <y_center> <width> <height>
```

示例：
```
0 0.5 0.5 0.8 0.6
```

**注意**：本工具默认使用Polygon格式，完整保留了原始分割标注的轮廓信息。

### 示例

假设有一个目标：
- 类别ID: 0
- 在100x100的图片中，边界框为 (x=20, y=30, w=40, h=50)

YOLO格式的标注行为：
```
0 0.40 0.55 0.40 0.50
```

计算过程：
- x_center = (20 + 40/2) / 100 = 0.40
- y_center = (30 + 50/2) / 100 = 0.55
- width = 40 / 100 = 0.40
- height = 50 / 100 = 0.50

### classes.txt 格式

每行定义一个类别，格式为：
```
<class_id> <class_name>
```

示例：
```
0 transformer_body
1 insulator
2 breaker
```

### 转换参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_yolo_path` | YOLO格式输出目录 | 输入目录的同级 `yolo_output` |
| `--samples_per_class` | 每个类别采样的数量 | 全部转换 |
| `--expand_ratio` | 边界框扩展比例（0-1） | 0.0 |
| `--min_size` | 最小裁剪尺寸 | 32像素 |
| `--num_workers` | 并行处理的线程数 | 8 |

### 并行处理

YOLO转换器支持多线程并行处理以加快转换速度：

- **类别级并行**：多个类别同时处理
- **图片级并行**：每个类别内的多张图片并行处理
- **线程安全**：使用锁机制保护共享资源（类别映射、统计信息）

默认使用8个线程，可根据CPU核心数调整。

### 使用示例

```bash
# 完整转换整个数据集
python scripts/main.py --dataset_path ./datasets --mode yolo --output_yolo_path ./yolo_dataset

# 采样转换（每类最多100张）
python scripts/main.py --dataset_path ./datasets --mode yolo --output_yolo_path ./yolo_dataset --samples_per_class 100

# 带边界扩展的转换（扩展10%）
python scripts/main.py --dataset_path ./datasets --mode yolo --output_yolo_path ./yolo_dataset --expand_ratio 0.1

# 跳过过小目标
python scripts/main.py --dataset_path ./datasets --mode yolo --output_yolo_path ./yolo_dataset --min_size 64

# 使用16线程并行处理
python scripts/main.py --dataset_path ./datasets --mode yolo --output_yolo_path ./yolo_dataset --num_workers 16

### 验证YOLO数据集

转换完成后，可以使用验证脚本检查裁剪和标注的正确性：

```bash
python scripts/yolo_validator.py \
  --yolo_path /path/to/yolo_output \
  --samples_per_class 10 \
  --output_path ./validation_output
```

**验证脚本功能**：
- 每个类别随机选择指定数量的图片
- 支持YOLO Polygon和Bounding Box两种格式
- 将YOLO标注转换为mask并半透明叠加到原图
- 绘制白色轮廓和类别名称
- 每张图片单独保存到对应类别子目录
- 生成详细的验证报告

**输出内容**：
```
validation_output/
├── transformer/                    # 变压器类别
│   ├── test1_1_validated.jpg
│   └── test2_1_validated.jpg
├── insulator/                      # 绝缘子类别
│   ├── test1_1_validated.jpg
│   └── test2_1_validated.jpg
├── breaker/                        # 断路器类别
│   ├── test1_1_validated.jpg
│   └── test2_1_validated.jpg
└── validation_report.txt          # 验证报告
```

## 错误处理

- 如果JSON文件格式不正确，工具会输出警告并跳过该样本
- 如果找不到对应的JSON文件，工具会输出警告
- 建议在正式使用前先运行统计模式检查数据完整性
