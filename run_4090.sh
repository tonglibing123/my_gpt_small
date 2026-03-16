#!/bin/bash
# ============================================
# MiniMind RTX 4090 优化训练脚本
# ============================================
# 此脚本针对 RTX 4090 (24GB VRAM) 优化
# 使用更大的 batch size 以充分利用 GPU 性能

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

print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
    echo ""
}

# 检查 GPU
check_gpu() {
    print_header "检查 GPU 环境"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未找到 nvidia-smi，请确认已安装 NVIDIA 驱动"
        exit 1
    fi
    
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    
    print_info "检测到 GPU: $GPU_NAME"
    print_info "显存大小: ${GPU_MEMORY}MB"
    
    if [[ ! "$GPU_NAME" =~ "4090" ]]; then
        print_warning "检测到的 GPU 不是 RTX 4090"
        print_warning "此脚本针对 RTX 4090 优化，其他 GPU 可能需要调整参数"
        read -p "是否继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "GPU 检查通过"
}

# 检查 Python 环境
check_python() {
    print_header "检查 Python 环境"
    
    if ! command -v python &> /dev/null; then
        print_error "未找到 Python"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python 版本: $PYTHON_VERSION"
    
    # 检查 PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_error "未安装 PyTorch"
        print_info "请运行: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        exit 1
    fi
    
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    
    print_info "PyTorch 版本: $TORCH_VERSION"
    print_info "CUDA 可用: $CUDA_AVAILABLE"
    
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        print_error "CUDA 不可用，请检查 PyTorch 安装"
        exit 1
    fi
    
    print_success "Python 环境检查通过"
}

# 检查依赖
check_dependencies() {
    print_header "检查依赖"
    
    REQUIRED_PACKAGES=("deepspeed" "transformers" "tokenizers" "tqdm" "yaml")
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "缺少以下依赖: ${MISSING_PACKAGES[*]}"
        print_info "正在安装..."
        pip install -r requirements.txt
    fi
    
    print_success "依赖检查通过"
}

# 检查数据文件
check_data() {
    print_header "检查数据文件"
    
    if [ ! -f "minimind_dataset/pretrain_hq.jsonl" ]; then
        print_error "未找到数据文件: minimind_dataset/pretrain_hq.jsonl"
        print_info "请从以下位置下载数据:"
        print_info "  ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind-dataset/files"
        print_info "  HuggingFace: https://huggingface.co/datasets/jingyaogong"
        exit 1
    fi
    
    print_success "数据文件检查通过"
}

# 训练配置
TOTAL_STEPS=5000
BATCH_SIZE=32  # RTX 4090 优化
SAVE_STEPS=500
LOG_INTERVAL=10

print_header "MiniMind RTX 4090 优化训练"
print_info "Batch Size: $BATCH_SIZE (针对 RTX 4090 优化)"
print_info "Total Steps: $TOTAL_STEPS"
print_info "预计训练时间: 1.5-2 小时"

# 执行检查
check_gpu
check_python
check_dependencies
check_data

# 步骤1: 训练 Tokenizer
print_header "步骤 1/5: 训练 Tokenizer (3-5分钟)"
if [ -d "my_tokenizer" ]; then
    print_warning "检测到已有 tokenizer，跳过训练"
else
    python src/training/train_tokenizer.py
    print_success "Tokenizer 训练完成"
fi

# 步骤2: 数据预处理
print_header "步骤 2/5: 数据预处理 (8-12分钟)"
if [ -f "data/train.bin" ]; then
    print_warning "检测到已有预处理数据，跳过"
else
    python src/data/pretokenize.py
    print_success "数据预处理完成"
fi

# 步骤3: 模型训练
print_header "步骤 3/5: 模型训练 (1.5-2小时)"
print_info "使用 RTX 4090 优化配置: batch_size=$BATCH_SIZE"

deepspeed --num_gpus 1 src/training/train_model.py \
    --vocab_size 6400 \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
    --block_size 256 \
    --total_steps $TOTAL_STEPS \
    --batch_size $BATCH_SIZE \
    --data_dir data \
    --deepspeed configs/deepspeed_zero2.json \
    --save_steps $SAVE_STEPS \
    --log_interval $LOG_INTERVAL

print_success "模型训练完成"

# 步骤4: 模型评估
print_header "步骤 4/5: 模型评估 (5分钟)"
if [ -f "tools/model_evaluator.py" ]; then
    python tools/model_evaluator.py \
        --checkpoint ckpt/pretrain/final \
        --max_samples 1000
    print_success "模型评估完成"
else
    print_warning "未找到评估脚本，跳过评估"
fi

# 步骤5: 推理测试
print_header "步骤 5/5: 推理测试"
python src/inference/infer.py \
    --checkpoint ckpt/pretrain/final \
    --prompt "人工智能在未来" \
    --max_new_tokens 100 \
    --temperature 0.8

print_success "推理测试完成"

# 完成
print_header "训练完成！"
print_success "所有步骤已完成"
print_info "模型保存在: ckpt/pretrain/final"
print_info ""
print_info "下一步:"
print_info "  1. 查看训练日志: tensorboard --logdir=runs"
print_info "  2. 快速评估模型: python tools/quick_eval.py --checkpoint ckpt/pretrain/final"
print_info "  3. 多轮对话测试: python src/inference/chat.py --checkpoint ckpt/pretrain/final"
print_info "  4. RLHF 训练: 参考 docs/RLHF训练完整指南.md"
print_info ""
print_info "性能统计:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

echo ""
print_success "🎉 恭喜！MiniMind 训练成功完成！"
