---
name: substation-segmentation-dataset
description: 变电站设备分割数据集管理与训练；本地数据处理 + 服务器训练全流水线
dependency:
  python:
    - opencv-python>=4.5.0
    - numpy>=1.19.0
    - matplotlib>=3.3.0
    - pillow>=8.0.0
    - pyyaml>=5.4.0
    - paramiko>=2.7.0
---

# 变电站设备分割数据集管理与训练 Skill

## 概述

本 Skill 覆盖变电站设备分割的完整流水线：

```
原始数据(raw_data) → 解压整理(extract_and_copy.sh) → 数据裁剪转换(main.py convert)
→ YOLO格式转换(main.py yolo) → 上传服务器(server_helper.py upload)
→ 检查GPU → 启动训练(nohup) → 监控训练 → 模型评估 → 下载模型
```

**本地操作**在 `/home/industai/workspace/SubStation_AI_Poject` 执行。
**服务器操作**通过 `scripts/server_helper.py` 远程执行，目标服务器 172.20.60.10。

---

## 前置条件

### 本地环境

```bash
# 核心依赖
pip install opencv-python numpy matplotlib pillow pyyaml paramiko

# YOLO26（本地测试用）
pip install ultralytics>=8.4.0
```

### 服务器配置

服务器连接信息存储在 `.server_config.yaml`（已加入 `.gitignore`，不提交 Git）：

```yaml
# .server_config.yaml 结构
servers:
  "172.20.60.10":
    host: "172.20.60.10"
    user: "industai"
    password: "xxx"
    port: 22
    project_path: "/home/industai/zxk/SubStation_AI_Poject"
    conda_env: "ultralytics-8.4.84"
    data_path: "/data/zxk/dataset"

local:
  project_path: "/home/industai/workspace/SubStation_AI_Poject"
  raw_data_path: "/media/industai/data11/SEG_DATA/raw_data"
  roi_data_path: "/media/industai/data11/SEG_DATA/roi自动分割"
  converted_data_path: "/media/industai/data11/SEG_DATA/converted_dataset"
  yolo_data_path: "/media/industai/data11/SEG_DATA/yolo_data"

training_defaults:
  yolo_version: "yolo26"
  model_size: "l"
  model_config: "train_configs/yolov26_l_seg_config.yaml"
  weights: "weights/yolo26l-seg.pt"
  epochs: 60
  batch_size: 3
  img_size: 1024
  ignore_classes: [...]
```

### 代码同步

服务器代码通过 Git 同步（**不是 rsync**）。本地修改代码后：
```bash
git add -A && git commit -m "描述"
git push origin main
```
然后在服务器上拉取：
```bash
python3 scripts/server_helper.py -s 172.20.60.10 exec "cd /home/industai/zxk/SubStation_AI_Poject && git pull"
```

---

## 流水线：第一阶段 — 本地数据处理

### 步骤 1：解压原始数据

当用户说"有新数据来了，在 xxx 目录"时，第一步是运行 `extract_and_copy.sh` 提取图片和 JSON 标注。

```bash
cd /home/industai/workspace/SubStation_AI_Poject
./scripts/extract_and_copy.sh \
  /media/industai/data11/SEG_DATA/raw_data/<批次目录> \
  /media/industai/data11/SEG_DATA/roi自动分割
```

**输入**：`raw_data/<batch>/<category>/<N>/` 下的 `jpg` 文件夹（图片）和 `json.zip`（标注）
**输出**：`roi自动分割/` 下混合存放所有 `*.jpg` 和 `*.json` 文件

**注意**：
- 如果 `roi自动分割/` 已有数据，新数据会追加进去（不覆盖同名文件）
- 先确认目标目录有写入权限（`roi自动分割/` 有时为只读挂载）

### 步骤 2：数据裁剪与格式转换

将图片+JSON标注裁剪为单目标小图，统一到 `converted_dataset/` 目录。

```bash
cd /home/industai/workspace/SubStation_AI_Poject
python3 ./scripts/main.py \
  --dataset_path /media/industai/data11/SEG_DATA/roi自动分割 \
  --class_mapping_file class_mapping.txt \
  --mode "convert" \
  --samples_per_class 10000 \
  --output_report /media/industai/data11/SEG_DATA/converted_dataset \
  --expand_ratio 0.0 \
  --min_size 200 \
  --enable_rectangle \
  --ignore_classes bei_jin,bei_jing
```

**关键参数说明**：
| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--samples_per_class` | 每类最多裁剪图片数 | 10000（全量） |
| `--expand_ratio` | 裁剪框扩展比例 | 0.0～0.1 |
| `--min_size` | 最小目标尺寸(px) | 200 |
| `--enable_rectangle` | 矩形优先策略 | 已启用 |
| `--ignore_classes` | 跳过的类别（逗号分隔） | bei_jin,bei_jing 等 |

### 步骤 3：YOLO 格式转换

将裁剪后的数据转为 YOLO 分割格式（train/val 划分 + polygon 标注）。

```bash
cd /home/industai/workspace/SubStation_AI_Poject
python3 ./scripts/main.py \
  --dataset_path /media/industai/data11/SEG_DATA/converted_dataset \
  --mode "yolo" \
  --output_yolo_path /media/industai/data11/SEG_DATA/yolo_data
```

**输出**：`yolo_data/` 下生成 `train/`、`val/`、`classes.txt`、`data.yaml`

---

## 流水线：第二阶段 — 上传到服务器

### 步骤 4：同步训练数据到服务器

使用 `server_helper.py upload` 做增量同步（只传变化的文件）。

```bash
cd /home/industai/workspace/SubStation_AI_Poject
python3 scripts/server_helper.py -s 172.20.60.10 upload \
  /media/industai/data11/SEG_DATA/yolo_data \
  /data/zxk/dataset/yolo_data
```

**建议先 dry-run** 查看将要传输的文件：
```bash
python3 scripts/server_helper.py -s 172.20.60.10 upload --dry-run \
  /media/industai/data11/SEG_DATA/yolo_data \
  /data/zxk/dataset/yolo_data
```

### 步骤 5：同步代码到服务器

Git push/pull 方式同步项目代码。

```bash
# 本地：确保已提交并推送
cd /home/industai/workspace/SubStation_AI_Poject
git status

# 服务器：拉取最新代码
python3 scripts/server_helper.py -s 172.20.60.10 exec \
  "cd /home/industai/zxk/SubStation_AI_Poject && git pull"
```

---

## 流水线：第三阶段 — 服务器训练

### 步骤 6：检查 GPU 状态

**必须在启动训练前检查**，因为服务器多人共享。

```bash
python3 scripts/server_helper.py -s 172.20.60.10 gpu
```

**判断逻辑**：
- 显存占用 < 15% 且利用率 < 15% → 空闲（显示 "✓ 空闲"）
- 挑选空闲 GPU 编号，传给训练的 `--device` 参数
- 如果所有 GPU 都在使用，告诉用户等待或选择占用最低的 GPU

### 步骤 7：启动训练（nohup 后台）

```bash
python3 scripts/server_helper.py -s 172.20.60.10 train \
  --workdir /home/industai/zxk/SubStation_AI_Poject \
  --logfile /home/industai/zxk/SubStation_AI_Poject/runs/train_$(date +%Y%m%d_%H%M%S).log \
  "conda activate ultralytics-8.4.84 && CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_yolo.py \
    --dataset_path /data/zxk/dataset/yolo_data \
    --output_dir ./runs/train \
    --yolo_version yolo26 \
    --model_config train_configs/yolov26_l_seg_config.yaml \
    --weights weights/yolo26l-seg.pt \
    --model_size l \
    --ignore_classes bei_jin,bei_jing,sheng_gao_zuo,you_zhen,mu_xian,dian_lan_zhong_duan,dian_kang_qi,you_wei,jie_xian_he,mo_ping,dian_rang_qi,二次接线盒 \
    --device 0,1,2,3"
```

**重要**：
- `--device` 参数必须与步骤 6 中确认的空闲 GPU 一致
- 日志路径包含时间戳，方便后续查找
- 记录下这个日志路径，后续监控需要

### 步骤 8：查看训练进度

```bash
# 查看最新日志
python3 scripts/server_helper.py -s 172.20.60.10 log \
  /home/industai/zxk/SubStation_AI_Poject/runs/train_YYYYMMDD_HHMMSS.log

# 查看最后 100 行
python3 scripts/server_helper.py -s 172.20.60.10 log \
  /home/industai/zxk/SubStation_AI_Poject/runs/train_YYYYMMDD_HHMMSS.log \
  --tail 100

# 检查训练进程是否还在运行
python3 scripts/server_helper.py -s 172.20.60.10 exec \
  "ps aux | grep train_yolo | grep -v grep"
```

---

## 流水线：第四阶段 — 模型评估与下载

### 步骤 9：模型验证

训练完成后，在服务器上对 best.pt 做验证：

```bash
python3 scripts/server_helper.py -s 172.20.60.10 exec \
  --workdir /home/industai/zxk/SubStation_AI_Poject \
  "python3 scripts/train_yolo.py \
    --mode val \
    --dataset_path /data/zxk/dataset/yolo_data \
    --weights ./runs/train/weights/best.pt \
    --device 0"
```

### 步骤 10：下载模型与训练产物

```bash
# 下载模型权重
python3 scripts/server_helper.py -s 172.20.60.10 download \
  /home/industai/zxk/SubStation_AI_Poject/runs/train/weights \
  ./runs/server_download/weights

# 下载完整训练产物（含图表、日志）
python3 scripts/server_helper.py -s 172.20.60.10 download \
  /home/industai/zxk/SubStation_AI_Poject/runs/train \
  ./runs/server_download

# 下载训练日志
python3 scripts/server_helper.py -s 172.20.60.10 download \
  /home/industai/zxk/SubStation_AI_Poject/runs/train_YYYYMMDD_HHMMSS.log \
  ./runs/
```

---

## 训练结果分析与讨论

训练完成后，agent 应主动分析关键指标并和用户讨论：

### 需要关注的指标

1. **mAP@0.5 和 mAP@0.5:0.95**
   - > 0.7 基准良好，> 0.85 优秀
   - 如果某个类别 AP 明显偏低，提示用户检查该类别标注质量

2. **Box Loss vs Mask Loss**
   - Box loss 远大于 mask loss → 定位不准，可能需要调整 anchor 或检查标注框
   - Mask loss 远大于 box loss → 分割轮廓不准，可能需要检查 polygon 标注精度

3. **训练/验证 Loss 曲线**
   - 训练 loss 持续下降但验证 loss 上升 → 过拟合，建议增加数据增强或减少 epochs
   - 两个 loss 都很高且不下降 → 学习率可能有问题

4. **类别分布**
   - 检查 `label_statistics_*.txt`，长尾类别（样本 < 100）建议增加数据

### 讨论要点

- 哪些类别表现差？是否需要补充标注数据？
- 是否需要调整 `ignore_classes`（太少的类别先忽略）？
- 数据增强参数是否需要调整？
- 是否需要换模型尺寸（如 l → x 提高精度）？

---

## 本地操作快捷参考

### 数据集统计

```bash
python3 ./scripts/main.py \
  --dataset_path /media/industai/data11/SEG_DATA/roi自动分割 \
  --mode stats
```

### 标注可视化

```bash
python3 ./scripts/main.py \
  --dataset_path /media/industai/data11/SEG_DATA/roi自动分割 \
  --mode visualize \
  --samples_per_class 3
```

### 本地快速训练测试

```bash
python3 scripts/train_yolo.py \
  --dataset_path /media/industai/data11/SEG_DATA/yolo_data \
  --output_dir ./runs/test \
  --yolo_version yolo26 \
  --model_size s \
  --epochs 5 \
  --batch_size 8 \
  --device 0
```

---

## 数据集目录结构参考

```
raw_data/<batch>/<category>/<N>/
├── <jpg文件夹>/
│   ├── img_001.jpg
│   └── ...
└── <N>-json.zip
    ├── img_001.json
    └── ...

roi自动分割/
├── img_001.jpg
├── img_001.json
└── ...

converted_dataset/
├── category_1/
│   ├── img_001.jpg
│   ├── img_001.json
│   └── ...
└── category_n/

yolo_data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── classes.txt
└── data.yaml
```

---

## 脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/extract_and_copy.sh` | 解压原始数据（zip→jpg+json） |
| `scripts/main.py` | 数据统计/可视化/YOLO转换主入口 |
| `scripts/dataset_cropper.py` | convert 模式的裁剪逻辑 |
| `scripts/yolo_formatter.py` | yolo 模式的格式转换逻辑 |
| `scripts/train_yolo.py` | YOLO 训练统一入口 |
| **`scripts/server_helper.py`** | **服务器 GPU 检查/远程命令/文件传输** |
| `statistic/check_servers.py` | 批量检查所有服务器 GPU 状态 |

---

## 安全约束（必须遵守）

1. **服务器操作限制**：`server_helper.py` 只能在指定的两个目录操作：
   - `/home/industai/zxk/SubStation_AI_Poject`（代码+模型）
   - `/data/zxk/dataset/`（训练数据）
   - **绝对禁止**操作服务器上其他路径的文件

2. **删除操作必须通知用户**：任何 `rm`、删除目录等操作前，agent 必须先向用户说明将要删除的内容并等待确认。

3. **GPU 冲突避免**：启动训练前必须检查 GPU 状态，不得在已被占用的 GPU 上启动训练。

4. **数据备份意识**：训练数据上传前确认本地 yolo_data 是最新的；下载模型时不要覆盖本地已有的同名模型。

5. **配置文件安全**：`.server_config.yaml` 含密码，已加入 `.gitignore`，禁止提交到 Git。
