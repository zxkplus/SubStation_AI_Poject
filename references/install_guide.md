# Conda环境安装指南

本文档提供了多种安装变电站设备分割训练环境的方法。

## 方法1: 完整自动安装脚本（推荐）

适用于全新安装，会自动检查并安装所有依赖。

```bash
cd /workspace/projects/substation-segmentation-dataset
chmod +x scripts/install_conda_env.sh
./scripts/install_conda_env.sh
```

**脚本功能**：
- 自动检查并安装Conda/Miniconda
- 创建独立的Conda环境
- 安装系统依赖
- 自动检测GPU并安装对应版本的PyTorch
- 安装所有训练所需的Python包
- 验证安装结果

**配置选项**：
- 环境名称: `substation_seg`
- Python版本: `3.10`
- CUDA版本: 自动检测

## 方法2: 快速安装脚本

适用于已有Conda/Mamba的快速安装。

```bash
cd /workspace/projects/substation-segmentation-dataset
chmod +x scripts/install_conda_env_quick.sh
./scripts/install_conda_env_quick.sh
```

## 方法3: 手动安装

### 步骤1: 安装Conda（如果未安装）

```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 添加到PATH
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

### 步骤2: 创建Conda环境

```bash
conda create -n substation_seg python=3.10 -y
conda activate substation_seg
```

### 步骤3: 安装PyTorch

**有GPU（推荐）**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**仅CPU**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 步骤4: 安装其他依赖

```bash
# 使用requirements.txt安装
pip install -r requirements.txt

# 或手动安装
pip install numpy opencv-python matplotlib pillow pyyaml ultralytics
```

### 步骤5: 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

## 方法4: 使用Docker（可选）

### 构建Docker镜像

```bash
cd /workspace/projects/substation-segmentation-dataset
docker build -t substation-seg-train:latest .
```

### 运行Docker容器

**使用docker-compose（推荐）**:
```bash
# 启动训练容器
docker-compose up train

# 启动TensorBoard（可选）
docker-compose --profile tensorboard up
```

**使用docker run**:
```bash
docker run --gpus all -it \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/runs:/workspace/runs \
  -v $(pwd):/workspace/substation-segmentation-dataset \
  -p 6006:6006 \
  substation-seg-train:latest
```

## 环境管理

### 激活环境

```bash
conda activate substation_seg
```

### 退出环境

```bash
conda deactivate
```

### 删除环境

```bash
conda deactivate  # 先退出环境
conda env remove -n substation_seg -y
```

### 导出环境

```bash
conda env export > environment.yml
```

### 从导出文件创建环境

```bash
conda env create -f environment.yml
```

## 常见问题

### Q1: 安装后找不到conda命令

**A**: 需要将conda添加到PATH

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Q2: GPU不可用

**A**: 检查NVIDIA驱动和CUDA

```bash
# 检查GPU
nvidia-smi

# 检查CUDA
nvcc --version

# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q3: 内存不足

**A**: 减小训练时的batch_size

```bash
python scripts/train_yolo.py --batch_size 16  # 减小到16或更小
```

### Q4: 网络连接慢

**A**: 使用国内镜像源

```bash
# 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### Q5: Docker无法访问GPU

**A**: 确保安装了NVIDIA Container Toolkit

```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 使用训练环境

安装完成后，可以开始训练：

```bash
# 激活环境
conda activate substation_seg

# 转换数据集
python scripts/main.py --dataset_path /path/to/dataset --mode yolo --output_yolo_path /path/to/yolo_dataset

# 训练YOLO26模型
python scripts/train_yolo.py --dataset_path /path/to/yolo_dataset --yolo_version yolo26 --model_size s

# 训练YOLOv6模型
python scripts/train_yolo.py --dataset_path /path/to/yolo_dataset --yolo_version yolov6
```

## 系统要求

### 最低要求
- **操作系统**: Ubuntu 20.04或更高
- **Python**: 3.10
- **内存**: 8GB RAM
- **存储**: 20GB可用空间
- **GPU**: 无（CPU模式）

### 推荐配置
- **操作系统**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **内存**: 32GB RAM
- **存储**: 100GB SSD
- **GPU**: NVIDIA RTX 3060或更高（8GB+ VRAM）

### 性能优化
- **GPU**: NVIDIA RTX 4090 / A100
- **内存**: 64GB+
- **存储**: NVMe SSD
- **多GPU**: 支持多GPU训练

## 参考资料

- [Conda官方文档](https://docs.conda.io/)
- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [Ultralytics文档](https://docs.ultralytics.com/)
- [Docker文档](https://docs.docker.com/)
