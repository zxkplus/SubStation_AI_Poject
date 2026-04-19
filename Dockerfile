# 变电站设备分割训练环境 - Docker镜像
# 基于Ubuntu 22.04

FROM ubuntu:22.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    python3.10 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 升级pip
RUN python -m pip install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt /tmp/requirements.txt

# 安装Python依赖
RUN pip install -r /tmp/requirements.txt

# 安装PyTorch（CUDA版本）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 创建工作目录
RUN mkdir -p /workspace/substation-segmentation-dataset

# 设置环境变量
ENV PYTHONPATH=/workspace/substation-segmentation-dataset:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# 默认命令
CMD ["/bin/bash"]
