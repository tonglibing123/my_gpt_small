"""
错误处理工具
提供友好的错误提示和解决方案
"""

import sys
import traceback
from typing import Optional, Callable
from functools import wraps


class MiniMindError(Exception):
    """MiniMind 基础异常类"""
    
    def __init__(self, message: str, solution: Optional[str] = None):
        self.message = message
        self.solution = solution
        super().__init__(self.message)
    
    def __str__(self):
        error_msg = f"\n{'=' * 60}\n"
        error_msg += f"❌ 错误: {self.message}\n"
        if self.solution:
            error_msg += f"\n💡 解决方案:\n{self.solution}\n"
        error_msg += f"{'=' * 60}\n"
        return error_msg


class ConfigError(MiniMindError):
    """配置错误"""
    pass


class DataError(MiniMindError):
    """数据错误"""
    pass


class ModelError(MiniMindError):
    """模型错误"""
    pass


class CheckpointError(MiniMindError):
    """检查点错误"""
    pass


class EnvironmentError(MiniMindError):
    """环境错误"""
    pass


def handle_file_not_found(filepath: str, file_type: str = "文件") -> str:
    """处理文件未找到错误"""
    solution = f"""
1. 检查文件路径是否正确: {filepath}
2. 确保文件存在
3. 检查文件权限
4. 如果是数据文件，请先运行数据准备脚本
"""
    raise DataError(f"{file_type}不存在: {filepath}", solution)


def handle_cuda_error() -> str:
    """处理 CUDA 错误"""
    solution = """
1. 检查 CUDA 是否正确安装:
   nvidia-smi
   
2. 检查 PyTorch CUDA 版本:
   python -c "import torch; print(torch.cuda.is_available())"
   
3. 如果没有 GPU，可以使用 CPU 模式:
   添加参数 --device cpu
   
4. 如果显存不足，尝试:
   - 减小 batch_size
   - 减小模型大小
   - 使用梯度累积
"""
    raise EnvironmentError("CUDA 不可用或出现错误", solution)


def handle_out_of_memory() -> str:
    """处理显存不足错误"""
    solution = """
1. 减小 batch_size:
   --batch_size 8  # 或更小
   
2. 使用梯度累积:
   修改 DeepSpeed 配置中的 gradient_accumulation_steps
   
3. 减小模型大小:
   --n_layer 4 --n_embd 256
   
4. 使用 DeepSpeed ZeRO:
   已默认启用 ZeRO-2，可以尝试 ZeRO-3
   
5. 清理 GPU 缓存:
   torch.cuda.empty_cache()
"""
    raise EnvironmentError("GPU 显存不足", solution)


def handle_checkpoint_error(checkpoint_path: str) -> str:
    """处理检查点错误"""
    solution = f"""
1. 检查检查点路径: {checkpoint_path}

2. 确保检查点文件完整:
   - pytorch_model.bin 或 model.pt
   - config.json（如果有）
   
3. 如果检查点损坏，尝试使用其他检查点:
   --checkpoint ckpt/pretrain/step_XXXX
   
4. 如果是首次训练，不要指定 --resume_from
"""
    raise CheckpointError(f"无法加载检查点: {checkpoint_path}", solution)


def handle_tokenizer_error(tokenizer_path: str) -> str:
    """处理 Tokenizer 错误"""
    solution = f"""
1. 检查 Tokenizer 路径: {tokenizer_path}

2. 确保已训练 Tokenizer:
   python src/training/train_tokenizer.py
   
3. 检查 Tokenizer 文件:
   - tokenizer.json
   - vocab.txt 或 merges.txt
   
4. 如果 Tokenizer 损坏，重新训练:
   rm -rf {tokenizer_path}
   python src/training/train_tokenizer.py
"""
    raise DataError(f"无法加载 Tokenizer: {tokenizer_path}", solution)


def handle_data_error(data_path: str) -> str:
    """处理数据错误"""
    solution = f"""
1. 检查数据文件: {data_path}

2. 确保数据格式正确:
   - JSONL 格式
   - 每行一个 JSON 对象
   - 包含 "text" 字段
   
3. 如果数据未预处理，运行:
   python src/data/pretokenize.py
   
4. 检查数据文件大小:
   ls -lh {data_path}
"""
    raise DataError(f"数据文件错误: {data_path}", solution)


def handle_import_error(module_name: str) -> str:
    """处理导入错误"""
    solution = f"""
1. 安装缺失的包:
   pip install {module_name}
   
2. 或安装所有依赖:
   pip install -r requirements.txt
   
3. 检查 Python 环境:
   python --version
   pip list
   
4. 如果使用虚拟环境，确保已激活:
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate     # Windows
"""
    raise EnvironmentError(f"缺少依赖包: {module_name}", solution)


def safe_execute(func: Callable, error_handler: Optional[Callable] = None):
    """
    安全执行函数，捕获并处理错误
    
    Args:
        func: 要执行的函数
        error_handler: 自定义错误处理函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到: {e.filename}")
            print("\n💡 解决方案:")
            print("1. 检查文件路径是否正确")
            print("2. 确保文件存在")
            print("3. 运行相应的准备脚本")
            sys.exit(1)
        
        except ImportError as e:
            module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print(f"\n❌ 缺少依赖包: {module_name}")
            print("\n💡 解决方案:")
            print(f"pip install {module_name}")
            print("或: pip install -r requirements.txt")
            sys.exit(1)
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "cuda" in error_msg or "gpu" in error_msg:
                if "out of memory" in error_msg:
                    print("\n❌ GPU 显存不足")
                    print("\n💡 解决方案:")
                    print("1. 减小 batch_size")
                    print("2. 使用梯度累积")
                    print("3. 减小模型大小")
                else:
                    print("\n❌ CUDA 错误")
                    print("\n💡 解决方案:")
                    print("1. 检查 CUDA 安装: nvidia-smi")
                    print("2. 使用 CPU 模式: --device cpu")
            else:
                print(f"\n❌ 运行时错误: {e}")
                if error_handler:
                    error_handler(e)
            
            sys.exit(1)
        
        except MiniMindError as e:
            print(str(e))
            sys.exit(1)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断执行")
            print("程序已安全退出")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n❌ 未预期的错误: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("\n详细错误信息:")
            traceback.print_exc()
            
            if error_handler:
                error_handler(e)
            
            sys.exit(1)
    
    return wrapper


def check_file_exists(filepath: str, file_type: str = "文件"):
    """检查文件是否存在"""
    from pathlib import Path
    if not Path(filepath).exists():
        handle_file_not_found(filepath, file_type)


def check_cuda_available():
    """检查 CUDA 是否可用"""
    try:
        import torch
        if not torch.cuda.is_available():
            handle_cuda_error()
    except ImportError:
        handle_import_error("torch")


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "torch",
        "transformers",
        "tokenizers",
        "deepspeed",
        "tqdm",
        "yaml",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        solution = f"""
缺少以下依赖包: {', '.join(missing)}

请运行以下命令安装:
pip install {' '.join(missing)}

或安装所有依赖:
pip install -r requirements.txt
"""
        raise EnvironmentError("缺少必需的依赖包", solution)


# 示例用法
if __name__ == "__main__":
    # 测试错误处理
    @safe_execute
    def test_function():
        raise FileNotFoundError("test.txt")
    
    test_function()
