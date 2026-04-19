#!/bin/bash
################################################################################
# 快速安装脚本（适用于已有Conda/Mamba的情况）
################################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}快速安装Conda环境${NC}"
echo -e "${BLUE}======================================${NC}"

# 配置
ENV_NAME="substation_seg"
PYTHON_VERSION="3.10"

# 激活conda
eval "$(conda shell.bash hook)" || eval "$(mamba shell.bash hook)"

# 创建环境
echo "创建环境: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# 激活环境
conda activate "${ENV_NAME}"

# 安装依赖
echo "安装PyTorch（自动检测CUDA）"
pip install torch torchvision torchaudio

echo "安装其他依赖"
pip install numpy opencv-python matplotlib pillow pyyaml ultralytics

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"

echo -e "${GREEN}安装完成！${NC}"
echo "激活环境: conda activate ${ENV_NAME}"
