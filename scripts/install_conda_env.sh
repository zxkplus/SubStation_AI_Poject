#!/bin/bash
################################################################################
# 自动安装变电站设备分割训练所需的Conda环境
# 支持Ubuntu系统
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 配置变量
ENV_NAME="substation_seg"  # Conda环境名称
PYTHON_VERSION="3.10"      # Python版本
CUDA_VERSION="12.1"        # CUDA版本（可选）

print_info "=========================================="
print_info "变电站设备分割训练环境安装脚本"
print_info "=========================================="
print_info "环境名称: ${ENV_NAME}"
print_info "Python版本: ${PYTHON_VERSION}"
print_info "=========================================="
echo ""

# 步骤1: 检查系统
print_info "步骤 1/7: 检查系统环境..."

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "此脚本仅支持Linux/Ubuntu系统"
    exit 1
fi

if ! command_exists apt; then
    print_error "未找到apt包管理器"
    exit 1
fi

print_success "系统检查通过"

# 步骤2: 检查并安装conda/miniconda
print_info "步骤 2/7: 检查Conda安装..."

if command_exists conda; then
    print_success "Conda已安装: $(conda --version)"
elif command_exists mamba; then
    print_success "Mamba已安装（使用Mamba代替Conda）"
    # 创建conda命令别名
    alias conda=mamba
else
    print_warning "未找到Conda/Mamba，正在安装Miniconda..."

    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="$HOME/miniconda.sh"

    print_info "下载Miniconda安装程序..."
    wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"

    print_info "安装Miniconda..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"

    # 添加到PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"

    rm "$MINICONDA_INSTALLER"

    print_success "Miniconda安装完成"
fi

# 初始化conda
if command_exists conda; then
    eval "$(conda shell.bash hook)"
elif command_exists mamba; then
    eval "$(mamba shell.bash hook)"
fi

# 步骤3: 检查并创建conda环境
print_info "步骤 3/7: 检查Conda环境..."

if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除现有环境..."
        conda env remove -n "$ENV_NAME" -y
        print_success "已删除现有环境"
    else
        print_info "使用现有环境"
        conda activate "$ENV_NAME"
    fi
else
    print_info "创建新的Conda环境..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    print_success "环境创建完成"
fi

# 激活环境
print_info "激活环境: ${ENV_NAME}"
conda activate "$ENV_NAME"

# 步骤4: 安装系统依赖
print_info "步骤 4/7: 安装系统依赖..."

print_info "更新apt包列表..."
sudo apt update

print_info "安装系统依赖..."
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

print_success "系统依赖安装完成"

# 步骤5: 安装CUDA相关（可选）
print_info "步骤 5/7: 检查CUDA支持..."

if command_exists nvidia-smi; then
    print_success "检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    # 检查CUDA
    if command_exists nvcc; then
        CUDA_INSTALLED=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_success "CUDA已安装: ${CUDA_INSTALLED}"
    else
        print_warning "未找到CUDA编译器，建议安装CUDA Toolkit"
        print_info "参考: https://developer.nvidia.com/cuda-downloads"
    fi

    # 安装PyTorch CUDA版本
    print_info "将安装PyTorch CUDA版本..."
    INSTALL_PYTORCH_CUDA=true
else
    print_warning "未检测到NVIDIA GPU，将安装CPU版本PyTorch"
    INSTALL_PYTORCH_CUDA=false
fi

# 步骤6: 安装Python依赖包
print_info "步骤 6/7: 安装Python依赖包..."

print_info "升级pip..."
pip install --upgrade pip setuptools wheel

# 创建requirements.txt
cat > /tmp/requirements.txt << EOF
# 核心依赖
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
pillow>=8.0.0
pyyaml>=5.4.0

# 深度学习框架
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# YOLO训练框架
ultralytics>=8.4.0

# 工具包
tqdm>=4.62.0
scipy>=1.7.0
scikit-learn>=0.24.0
pandas>=1.3.0
seaborn>=0.11.0

# 可选：更好的可视化（可选）
tensorboard>=2.10.0
EOF

print_info "安装PyTorch..."
if [ "$INSTALL_PYTORCH_CUDA" = true ]; then
    print_info "安装PyTorch CUDA版本..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_info "安装PyTorch CPU版本..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

print_info "安装其他依赖包..."
pip install -r /tmp/requirements.txt

# 步骤7: 验证安装
print_info "步骤 7/7: 验证安装..."

print_info "检查Python版本..."
python --version

print_info "检查关键包..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"

if [ "$INSTALL_PYTORCH_CUDA" = true ]; then
    print_info "检查CUDA可用性..."
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count()}')"
fi

print_success "=========================================="
print_success "安装完成！"
print_success "=========================================="
echo ""
print_info "使用方法："
print_info "  1. 激活环境: conda activate ${ENV_NAME}"
print_info "  2. 运行训练: python scripts/train_yolo.py --dataset_path <数据集路径> --yolo_version yolo26"
echo ""
print_info "环境信息:"
print_info "  环境名称: ${ENV_NAME}"
print_info "  Python: $(python --version)"
print_info "  PyTorch: $(python -c "import torch; print(torch.__version__)")"
if [ "$INSTALL_PYTORCH_CUDA" = true ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        print_info "  GPU数量: ${GPU_COUNT}"
    fi
fi
echo ""
print_info "如需卸载环境，运行:"
print_info "  conda deactivate"
print_info "  conda env remove -n ${ENV_NAME}"
echo ""
