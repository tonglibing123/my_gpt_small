#!/bin/bash
# MiniMind 一键运行脚本 (Linux/Mac)
# 自动执行完整的训练流程
# 默认配置: RTX 4090 GPU

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
    echo ""
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检测 GPU 并推荐配置
check_gpu() {
    if command_exists nvidia-smi; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        
        print_info "检测到 GPU: $GPU_NAME"
        print_info "显存大小: ${GPU_MEMORY}MB"
        
        if [[ "$GPU_NAME" =~ "4090" ]]; then
            print_success "检测到 RTX 4090，使用优化配置 (batch_size=32)"
            BATCH_SIZE=32
            print_info "推荐使用: bash run_4090.sh (专用优化脚本)"
        elif [[ "$GPU_NAME" =~ "3090" ]]; then
            print_warning "检测到 RTX 3090，使用标准配置 (batch_size=16)"
            BATCH_SIZE=16
        else
            print_warning "未检测到 RTX 4090/3090，使用保守配置 (batch_size=8)"
            BATCH_SIZE=8
        fi
    else
        print_warning "未检测到 nvidia-smi，使用默认配置 (batch_size=16)"
        BATCH_SIZE=16
    fi
}

# 主函数
main() {
    print_header "MiniMind 一键运行脚本"
    
    # 检查 GPU
    check_gpu
    echo ""
    
    # 检查 Python
    if ! command_exists python && ! command_exists python3; then
        print_error "未找到 Python，请先安装 Python 3.10+"
        exit 1
    fi
    
    # 使用 python3 或 python
    PYTHON_CMD="python3"
    if ! command_exists python3; then
        PYTHON_CMD="python"
    fi
    
    print_info "使用 Python: $PYTHON_CMD"
    
    # 步骤0: 环境检查
    print_header "步骤0: 环境检查"
    if $PYTHON_CMD tools/check_env.py; then
        print_success "环境检查通过"
    else
        print_error "环境检查失败，请先解决环境问题"
        exit 1
    fi
    
    # 询问运行模式
    echo ""
    print_info "请选择运行模式:"
    echo "  1) 快速模式 (1000 steps, RTX 4090: ~30分钟, RTX 3090: ~50分钟)"
    echo "  2) 完整模式 (5000 steps, RTX 4090: ~1.5小时, RTX 3090: ~2.5小时)"
    echo "  3) 仅推理 (使用已有模型)"
    read -p "请输入选项 [1-3]: " mode
    
    case $mode in
        1)
            TOTAL_STEPS=1000
            MODE_NAME="快速模式"
            ;;
        2)
            TOTAL_STEPS=5000
            MODE_NAME="完整模式"
            ;;
        3)
            print_info "跳过训练，直接进入推理"
            run_inference
            exit 0
            ;;
        *)
            print_warning "无效选项，使用快速模式"
            TOTAL_STEPS=1000
            MODE_NAME="快速模式"
            ;;
    esac
    
    print_info "运行模式: $MODE_NAME ($TOTAL_STEPS steps)"
    print_info "Batch Size: $BATCH_SIZE (根据 GPU 自动配置)"
    
    # 步骤1: 训练 Tokenizer
    print_header "步骤1: 训练 Tokenizer"
    if [ -d "my_tokenizer" ]; then
        print_warning "检测到已有 tokenizer，跳过训练"
    else
        print_info "开始训练 Tokenizer..."
        $PYTHON_CMD src/training/train_tokenizer.py
        print_success "Tokenizer 训练完成"
    fi
    
    # 步骤2: 数据预处理
    print_header "步骤2: 数据预处理"
    if [ -f "data/train.bin" ]; then
        print_warning "检测到已有预处理数据，跳过预处理"
    else
        print_info "开始数据预处理..."
        $PYTHON_CMD src/data/pretokenize.py
        print_success "数据预处理完成"
    fi
    
    # 步骤3: 分词器分析（可选）
    print_header "步骤3: 分词器分析 (可选)"
    read -p "是否运行分词器分析? [y/N]: " run_analyze
    if [[ $run_analyze =~ ^[Yy]$ ]]; then
        print_info "运行分词器分析..."
        $PYTHON_CMD tools/tokenizer_analyzer.py
        print_success "分词器分析完成"
    else
        print_info "跳过分词器分析"
    fi
    
    # 步骤4: 模型训练
    print_header "步骤4: 模型训练"
    print_info "开始训练模型 ($TOTAL_STEPS steps)..."
    print_warning "这可能需要较长时间，请耐心等待..."
    
    if command_exists deepspeed; then
        deepspeed --num_gpus 1 src/training/train_model.py \
            --total_steps $TOTAL_STEPS \
            --batch_size $BATCH_SIZE \
            --save_steps 500 \
            --log_interval 10
        print_success "模型训练完成"
    else
        print_error "未找到 deepspeed 命令"
        print_info "尝试使用 Python 直接运行..."
        $PYTHON_CMD src/training/train_model.py \
            --total_steps $TOTAL_STEPS \
            --batch_size $BATCH_SIZE \
            --save_steps 500 \
            --log_interval 10
        print_success "模型训练完成"
    fi
    
    # 步骤5: 模型评估（可选）
    print_header "步骤5: 模型评估 (可选)"
    read -p "是否运行模型评估? [y/N]: " run_eval
    if [[ $run_eval =~ ^[Yy]$ ]]; then
        print_info "运行模型评估..."
        $PYTHON_CMD tools/model_evaluator.py \
            --checkpoint ckpt/pretrain/final \
            --max_samples 1000
        print_success "模型评估完成"
    else
        print_info "跳过模型评估"
    fi
    
    # 步骤6: 可视化（可选）
    print_header "步骤6: 训练可视化 (可选)"
    read -p "是否生成可视化报告? [y/N]: " run_viz
    if [[ $run_viz =~ ^[Yy]$ ]]; then
        print_info "生成可视化报告..."
        $PYTHON_CMD tools/visualizer.py
        print_success "可视化报告已生成: training_report.html"
    else
        print_info "跳过可视化"
    fi
    
    # 步骤7: 推理测试
    run_inference
    
    # 完成
    print_header "🎉 完整流程运行完成！"
    print_success "所有步骤已完成"
    echo ""
    print_info "下一步建议:"
    echo "  1. 查看训练报告: open training_report.html"
    echo "  2. 启动多轮对话: $PYTHON_CMD src/inference/chat.py --checkpoint ckpt/pretrain/final"
    echo "  3. 查看 TensorBoard: tensorboard --logdir=runs"
    echo ""
}

# 推理函数
run_inference() {
    print_header "步骤7: 推理测试"
    
    if [ ! -d "ckpt/pretrain/final" ]; then
        print_error "未找到训练好的模型"
        return
    fi
    
    print_info "运行推理测试..."
    $PYTHON_CMD src/inference/infer.py \
        --checkpoint ckpt/pretrain/final \
        --prompt "人工智能在未来" \
        --max_new_tokens 50 \
        --temperature 0.8
    
    print_success "推理测试完成"
    
    echo ""
    read -p "是否启动多轮对话? [y/N]: " run_chat
    if [[ $run_chat =~ ^[Yy]$ ]]; then
        print_info "启动多轮对话..."
        $PYTHON_CMD src/inference/chat.py \
            --checkpoint ckpt/pretrain/final \
            --max_new_tokens 50
    fi
}

# 错误处理
trap 'print_error "脚本执行失败"; exit 1' ERR

# 运行主函数
main

exit 0
