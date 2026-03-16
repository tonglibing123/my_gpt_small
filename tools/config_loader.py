"""
配置文件加载工具
支持从 YAML 文件加载配置，并提供命令行参数覆盖功能
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """配置类，支持点号访问"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if '.' in key:
                # 支持嵌套更新，如 "model.n_layer"
                keys = key.split('.')
                obj = self
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
            else:
                if hasattr(self, key) and isinstance(getattr(self, key), Config):
                    if isinstance(value, dict):
                        getattr(self, key).update(value)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 configs/config.yaml
    
    Returns:
        Config 对象
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = "configs/config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确保配置文件位于正确的位置。"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"配置文件格式错误: {config_path}\n"
            f"错误信息: {e}\n"
            f"请检查 YAML 文件格式是否正确。"
        )
    
    return Config(config_dict)


def merge_args_with_config(config: Config, args) -> Config:
    """
    将命令行参数合并到配置中
    命令行参数优先级高于配置文件
    
    Args:
        config: 配置对象
        args: argparse 解析的参数
    
    Returns:
        更新后的配置对象
    """
    args_dict = vars(args)
    
    # 过滤掉 None 值（未指定的参数）
    updates = {k: v for k, v in args_dict.items() if v is not None}
    
    # 更新配置
    config.update(updates)
    
    return config


def save_config(config: Config, save_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, allow_unicode=True, default_flow_style=False)


def print_config(config: Config, indent: int = 0):
    """
    打印配置信息
    
    Args:
        config: 配置对象
        indent: 缩进级别
    """
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


# 示例用法
if __name__ == "__main__":
    # 加载配置
    config = load_config()
    
    # 打印配置
    print("=" * 60)
    print("配置信息:")
    print("=" * 60)
    print_config(config)
    
    # 访问配置
    print("\n" + "=" * 60)
    print("访问示例:")
    print("=" * 60)
    print(f"模型层数: {config.model.n_layer}")
    print(f"学习率: {config.pretrain.learning_rate}")
    print(f"批次大小: {config.pretrain.batch_size}")
    
    # 更新配置
    print("\n" + "=" * 60)
    print("更新配置:")
    print("=" * 60)
    config.update({"model.n_layer": 12, "pretrain.batch_size": 32})
    print(f"更新后的层数: {config.model.n_layer}")
    print(f"更新后的批次大小: {config.pretrain.batch_size}")
