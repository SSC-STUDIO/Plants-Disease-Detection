#!/bin/bash
# Plants Disease Detection - Remote Server Setup Script
# 用于在远程服务器上配置训练环境

set -e

echo "=========================================="
echo "Plants Disease Detection - 环境配置"
echo "=========================================="

# 检查Python版本
echo "[1/7] 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 建议Python 3.10+
if [[ $(echo "$python_version" | awk -F. '{print $1}') -lt 3 ]] || \
   [[ $(echo "$python_version" | awk -F. '{print $1}') -eq 3 && $(echo "$python_version" | awk -F. '{print $2}') -lt 10 ]]; then
    echo "警告: 建议使用Python 3.10+以获得最佳性能"
fi

# 检查CUDA版本
echo "[2/7] 检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "CUDA版本: $cuda_version"
else
    echo "警告: 未检测到CUDA，训练将在CPU上进行（速度较慢）"
fi

# 创建虚拟环境（可选）
read -p "是否创建新的虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "[3/7] 创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# 升级pip
echo "[4/7] 升级pip和setuptools..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch
echo "[5/7] 安装PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装主要依赖
echo "[6/7] 安装项目依赖..."
pip install \
    timm>=0.9.0 \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    Pillow \
    matplotlib \
    seaborn \
    opencv-python \
    albumentations \
    scikit-image \
    tensorboard \
    tensorboardX \
    psutil \
    PyYAML \
    requests \
    python-dotenv \
    efficientnet-pytorch \
    torch_optimizer \
    rarfile

# 安装额外的高级依赖
echo "[7/7] 安装高级依赖..."
pip install \
    einops \
    flash_attn

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"

# 检查CUDA可用性
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA可用: {torch.cuda.get_device_name(0)}')
    print(f'CUDA版本: {torch.version.cuda}')
else:
    print('CUDA不可用，将在CPU上运行')
"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "接下来可以运行训练："
echo "  python main.py train --epochs 50 --model eva02_large"
echo ""
