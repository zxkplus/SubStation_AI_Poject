# SubStation AI - 数据集处理工具集

`main.py` 是变电站设备分割数据集处理的主入口脚本，整合了**统计分析**、**可视化**、**YOLO 格式转换**和**数据集裁剪**四类功能。

## 依赖

```
numpy, opencv-python (cv2), pycocotools (可选)
```

确保 `scripts/` 目录下的所有模块文件完整（`data_loader.py`, `statistics.py`, `visualization.py`, `yolo_formatter.py`, `dataset_cropper.py`）。

## 运行模式概览

| 模式 | 说明 |
|------|------|
| `stats` | 加载数据集并生成统计报告（类别分布、样本数等） |
| `visualize` | 加载数据集并对标注进行可视化，生成带 mask 叠加的图片 |
| `full` | 统计 + 可视化（默认模式） |
| `yolo` | 将数据集转换为 YOLO 分割格式，自动划分 train/val |
| `convert` | 按照目标裁剪图片，保留 JSON 标注格式（多边形坐标变换到裁剪图坐标系） |

---

## 1. stats / visualize / full 模式

用于**快速浏览和检查数据集质量**，要求数据集按类别目录组织：

```
dataset_root/
├── 类别A/
│   ├── img_001.jpg
│   ├── img_001.json    # LabelMe 格式标注
│   ├── img_002.jpg
│   └── img_002.json
├── 类别B/
│   └── ...
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_path` | 是 | - | 数据集根目录路径（不支持 .txt） |
| `--mode` | 否 | `full` | `stats` / `visualize` / `full` |
| `--samples_per_class` | 否 | `2` | 可视化时每类随机采样的样本数 |
| `--output_report` | 否 | 无 | 统计报告输出文件路径，不指定则只打印到终端 |
| `--output_visualization` | 否 | `./visualization_output` | 可视化图片输出目录 |

### 示例

```bash
# 统计 + 可视化
python scripts/main.py --dataset_path /data/substation --mode full --samples_per_class 5

# 仅统计，保存报告
python scripts/main.py --dataset_path /data/substation --mode stats --output_report report.txt

# 仅可视化
python scripts/main.py --dataset_path /data/substation --mode visualize --samples_per_class 10
```

---

## 2. yolo 模式

将数据集转换为 **YOLO 实例分割格式**，自动划分训练集/验证集。

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_path` | 是 | - | 数据集根目录（支持 .txt 批量文件，但仅处理第一个） |
| `--mode` | 是 | - | `yolo` |
| `--output_yolo_path` | 否 | 输入目录同级 `yolo_dataset/` | YOLO 数据集输出路径 |
| `--num_workers` | 否 | `8` | 并行线程数 |
| `--ignore_classes` | 否 | 空 | 要忽略的类别名列表，支持逗号分隔 |

### 输出结构

```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/       # YOLO 分割格式 .txt
│   └── val/
└── data.yaml        # 数据集配置
```

### 示例

```bash
# 单数据集
python scripts/main.py --dataset_path /data/substation --mode yolo --output_yolo_path /data/yolo_out

# 忽略特定类别
python scripts/main.py --dataset_path /data/substation --mode yolo --ignore_classes 背景 杂质,虚化

# 注意: --ignore_classes 支持空格分隔或逗号分隔（或混用），逗号处会自动拆分
```

---

## 3. convert 模式

将数据集的每个标注目标**单独裁剪**为独立图片，保留 JSON 标注格式（多边形坐标变换到裁剪图片坐标系）。

### 核心逻辑

1. 递归加载数据集目录结构，以父目录名作为类别名
2. 每类随机采样（默认 100 张），不足则全取
3. 对每个 Polygon 计算外接矩形并裁剪，合并相交的多边形到同一裁剪图中
4. `ignore_classes` 中的类别作为"附属物"处理：
   - 若与主目标重叠，标签被替换为包围它的主类别标签
   - 若孤立存在，则直接跳过
5. 尺寸小于 `min_size` 的目标会被跳过
6. 坐标全部变换为相对于裁剪图的坐标

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_path` | 是 | - | 数据集根目录（支持 .txt 批量处理多个数据集） |
| `--mode` | 是 | - | `convert` |
| `--output_yolo_path` | 否 | 输入目录同级 `converted_dataset/` | 裁剪后的输出路径 |
| `--samples_per_class` | 否 | `100`（convert 模式默认值） | 每类采样数 |
| `--expand_ratio` | 否 | `0.0` | bbox 扩展比例（0~1），增加裁剪边距 |
| `--min_size` | 否 | `32` | 最小裁剪尺寸（像素），低于此值的目标跳过 |
| `--num_workers` | 否 | `8` | 并行线程数 |
| `--ignore_classes` | 否 | 空 | 忽略的类别列表，不会被单独裁剪 |
| `--class_mapping_file` | 否 | 无 | 类别映射文件，用于中文→英文类别名转换 |
| `--enable_rectangle` | 否 | `False` | 启用矩形优先策略（见下方说明） |

### 矩形优先策略 (`--enable_rectangle`)

默认关闭。开启后：
- 优先以 `shape_type: rectangle` 的标注作为裁剪边界
- 矩形内包含的所有多边形标注会作为 shapes 一并保留
- 已处理的多边形不再单独裁剪

关闭时（默认）：忽略所有矩形标注，仅按多边形外接矩形裁剪。

### 输出结构

**单个数据集**时保留类别目录：
```
converted_dataset/
├── 类别A/
│   ├── 类别A_img_001_1.jpg
│   ├── 类别A_img_001_1.json
│   └── ...
├── 类别B/
│   └── ...
```

**多个数据集**时（.txt 输入）不保留类别目录，所有输出平铺到根目录以避免冲突。

### 示例

```bash
# 基本转换
python scripts/main.py --dataset_path /data/substation --mode convert

# 自定义采样数和裁剪边距
python scripts/main.py --dataset_path /data/substation --mode convert \
    --samples_per_class 200 --expand_ratio 0.1 --min_size 48

# 多数据集批量处理（通过 .txt 文件）
python scripts/main.py --dataset_path datasets.txt --mode convert --output_yolo_path /data/cropped

# 类别映射 + 忽略类别 + 矩形优先
python scripts/main.py --dataset_path /data/substation --mode convert \
    --class_mapping_file class_mapping.txt \
    --ignore_classes 背景 杂质 \
    --enable_rectangle
```

### 类别映射文件格式

```
# 注释行以 # 开头
中文名A:english_name_a
中文名B:english_name_b
```

---

## 批量处理 .txt 文件格式

`convert` 和 `yolo` 模式支持通过 `--dataset_path` 传入一个 `.txt` 文件，每行一个数据集路径：

```
# 注释行
/path/to/dataset_1
/path/to/dataset_2
/path/to/dataset_3
```
