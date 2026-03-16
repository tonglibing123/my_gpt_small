# utils.py - 工具函数
import os
import sys
import importlib.util


def _get_project_root():
    """获取项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # tools/utils.py -> 项目根目录
    return os.path.dirname(current_dir)


def _load_module_from_path(module_path: str, module_name: str = None):
    """从绝对路径加载模块"""
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"文件不存在: {module_path}")
    
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_module(base_dir: str = None):
    """加载模型模块 (src/models/model.py)"""
    if base_dir is None:
        base_dir = _get_project_root()
    model_path = os.path.join(base_dir, "src", "models", "model.py")
    return _load_module_from_path(model_path, "model")


def load_common_module(base_dir: str = None):
    """加载通用组件模块 (src/models/components.py)"""
    if base_dir is None:
        base_dir = _get_project_root()
    components_path = os.path.join(base_dir, "src", "models", "components.py")
    return _load_module_from_path(components_path, "components")


def load_data_module(base_dir: str = None):
    """加载数据模块 (src/data/dataset.py)"""
    if base_dir is None:
        base_dir = _get_project_root()
    dataset_path = os.path.join(base_dir, "src", "data", "dataset.py")
    return _load_module_from_path(dataset_path, "dataset")


def get_model_components(base_dir: str = None):
    """获取模型相关的所有组件"""
    model_module = load_model_module(base_dir)
    common_module = load_common_module(base_dir)
    return {
        'MyGPT': model_module.MyGPT,
        'CheckpointManager': common_module.CheckpointManager,
    }


def create_model(vocab_size: int = 6400, **kwargs):
    """便捷函数：创建模型"""
    return load_model_module().MyGPT(vocab_size=vocab_size, **kwargs)
