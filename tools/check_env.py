#!/usr/bin/env python3
"""
环境检查脚本
检查运行 MiniMind 所需的环境配置
"""
import sys
import os

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_item(status, message):
    """打印检查项"""
    icon = "✅" if status else "❌"
    print(f"{icon} {message}")

def check_python_version():
    """检查 Python 版本"""
    print_header("Python 环境")
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        print_item(False, f"Python 版本过低: {version.major}.{version.minor}.{version.micro}")
        print(f"   需要: Python {required_major}.{required_minor}+")
        print(f"   建议: 升级到 Python 3.10+")
        return False
    
    print_item(True, f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_packages():
    """检查必要的包"""
    print_header("Python 包")
    
    required_packages = {
        'torch': ('2.0.0', 'PyTorch 深度学习框架'),
        'transformers': ('4.30.0', 'HuggingFace Transformers'),
        'deepspeed': ('0.9.0', 'DeepSpeed 分布式训练'),
        'tokenizers': ('0.13.0', 'Fast Tokenizers'),
        'numpy': ('1.20.0', '数值计算'),
        'tqdm': ('4.60.0', '进度条'),
    }
    
    optional_packages = {
        'tensorboard': ('2.10.0', 'TensorBoard 训练监控'),
        'matplotlib': ('3.5.0', '可视化工具'),
    }
    
    all_ok = True
    missing_packages = []
    
    # 检查必需包
    for package, (min_version, description) in required_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print_item(True, f"{package:15s} {version:10s} - {description}")
        except ImportError:
            print_item(False, f"{package:15s} {'未安装':10s} - {description}")
            missing_packages.append(package)
            all_ok = False
    
    # 检查可选包
    print("\n可选包:")
    for package, (min_version, description) in optional_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print_item(True, f"{package:15s} {version:10s} - {description}")
        except ImportError:
            print_item(False, f"{package:15s} {'未安装':10s} - {description} (可选)")
    
    if missing_packages:
        print("\n💡 安装缺失的包:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    return all_ok

def check_cuda():
    """检查 CUDA 和 GPU"""
    print_header("GPU 和 CUDA")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            
            print_item(True, f"CUDA {cuda_version}")
            print_item(True, f"检测到 {gpu_count} 个 GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                if gpu_memory < 8:
                    print(f"   ⚠️  显存较小，建议至少 8GB")
            
            return True
        else:
            print_item(False, "CUDA 不可用")
            print("   ⚠️  将使用 CPU 训练（速度会很慢）")
            print("   💡 建议:")
            print("      1. 检查是否安装了 CUDA")
            print("      2. 安装 GPU 版本的 PyTorch")
            print("      3. 或使用云 GPU 服务（Colab, Kaggle 等）")
            return False
    except Exception as e:
        print_item(False, f"无法检查 CUDA: {e}")
        return False

def check_disk_space():
    """检查磁盘空间"""
    print_header("磁盘空间")
    
    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        
        if free_gb < 10:
            print_item(False, f"磁盘空间不足: {free_gb:.1f} GB 可用")
            print(f"   总空间: {total_gb:.1f} GB")
            print(f"   已使用: {used_gb:.1f} GB")
            print(f"   建议: 至少需要 20GB 可用空间")
            return False
        elif free_gb < 20:
            print_item(True, f"磁盘空间: {free_gb:.1f} GB 可用（建议至少 20GB）")
            print(f"   ⚠️  空间较紧张，建议清理磁盘")
        else:
            print_item(True, f"磁盘空间: {free_gb:.1f} GB 可用")
        
        return True
    except Exception as e:
        print_item(False, f"无法检查磁盘空间: {e}")
        return False

def check_files():
    """检查必要的文件"""
    print_header("项目文件")
    
    required_files = [
        ('src/training/train_tokenizer.py', '训练 Tokenizer'),
        ('src/data/pretokenize.py', '数据预处理'),
        ('src/training/train_model.py', '模型训练'),
        ('src/inference/infer.py', '模型推理'),
        ('src/inference/chat.py', '多轮对话'),
        ('src/models/model.py', '模型定义'),
        ('src/models/components.py', '通用组件'),
        ('src/data/dataset.py', '数据集'),
        ('tools/utils.py', '工具函数'),
        ('requirements.txt', '依赖列表'),
        ('configs/deepspeed_zero2.json', 'DeepSpeed 配置'),
    ]
    
    all_ok = True
    for filename, description in required_files:
        if os.path.exists(filename):
            print_item(True, f"{filename:40s} - {description}")
        else:
            print_item(False, f"{filename:40s} - {description} (缺失)")
            all_ok = False
    
    if not all_ok:
        print("\n   ⚠️  部分文件缺失，请检查项目完整性")
    
    return all_ok

def check_data():
    """检查数据目录"""
    print_header("数据目录")
    
    data_dirs = [
        ('minimind_dataset', '原始数据集', False),
        ('data', '预处理数据', False),
        ('ckpt', 'Checkpoint 目录', False),
    ]
    
    for dirname, description, required in data_dirs:
        if os.path.exists(dirname):
            if os.path.isdir(dirname):
                file_count = len(os.listdir(dirname))
                print_item(True, f"{dirname:20s} - {description} ({file_count} 个文件)")
            else:
                print_item(False, f"{dirname:20s} - 不是目录")
        else:
            if required:
                print_item(False, f"{dirname:20s} - {description} (缺失)")
            else:
                print_item(True, f"{dirname:20s} - {description} (将自动创建)")
    
    return True

def print_summary(results):
    """打印总结"""
    print_header("检查总结")
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        print_item(passed, check_name)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 环境检查通过！可以开始训练")
        print("\n💡 下一步:")
        print("   1. 运行一键脚本: bash run_all.sh")
        print("   2. 或查看快速开始: cat 快速开始.md")
    else:
        print("❌ 环境检查失败，请先解决上述问题")
        print("\n💡 常见解决方案:")
        print("   1. 安装缺失的包: pip install -r requirements.txt")
        print("   2. 升级 Python: 使用 Python 3.8+")
        print("   3. 安装 CUDA: 访问 https://developer.nvidia.com/cuda-downloads")
        print("   4. 清理磁盘空间")
    print("="*60)
    
    return all_passed

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  MiniMind 环境检查工具")
    print("  检查运行所需的环境配置")
    print("="*60)
    
    results = {}
    
    # 执行所有检查
    results['Python 版本'] = check_python_version()
    results['Python 包'] = check_packages()
    results['GPU 和 CUDA'] = check_cuda()
    results['磁盘空间'] = check_disk_space()
    results['项目文件'] = check_files()
    results['数据目录'] = check_data()
    
    # 打印总结
    all_passed = print_summary(results)
    
    # 返回状态码
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  检查被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
