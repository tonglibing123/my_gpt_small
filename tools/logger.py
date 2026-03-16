"""
日志工具
提供统一的日志记录功能，支持彩色输出、文件保存、TensorBoard
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


# ANSI 颜色代码
class Colors:
    """终端颜色"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 前景色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.BRIGHT_BLUE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BOLD + Colors.BRIGHT_RED,
    }
    
    def format(self, record):
        # 保存原始 levelname
        levelname = record.levelname
        
        # 添加颜色
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"
        
        # 格式化消息
        result = super().format(record)
        
        # 恢复原始 levelname
        record.levelname = levelname
        
        return result


def setup_logger(
    name: str = "MiniMind",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_color: bool = True,
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名（可选）
        log_dir: 日志目录
        use_color: 是否使用彩色输出
    
    Returns:
        配置好的 Logger 对象
    """
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if use_color and sys.stdout.isatty():
        # 使用彩色格式化器
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file:
        # 创建日志目录
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir_path / f"{log_file}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 文件不使用颜色
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file_path}")
    
    return logger


def log_config(logger: logging.Logger, config):
    """
    记录配置信息
    
    Args:
        logger: Logger 对象
        config: 配置对象
    """
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info("=" * 60)
    
    def _log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                _log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    if hasattr(config, 'to_dict'):
        _log_dict(config.to_dict())
    elif isinstance(config, dict):
        _log_dict(config)
    else:
        logger.info(str(config))
    
    logger.info("=" * 60)


def log_model_info(logger: logging.Logger, model):
    """
    记录模型信息
    
    Args:
        logger: Logger 对象
        model: 模型对象
    """
    logger.info("=" * 60)
    logger.info("模型信息:")
    logger.info("=" * 60)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    logger.info("=" * 60)


def log_training_step(
    logger: logging.Logger,
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    elapsed_time: float,
    **kwargs
):
    """
    记录训练步骤信息
    
    Args:
        logger: Logger 对象
        step: 当前步数
        total_steps: 总步数
        loss: 损失值
        lr: 学习率
        elapsed_time: 已用时间（秒）
        **kwargs: 其他要记录的指标
    """
    progress = step / total_steps * 100
    
    # 估算剩余时间
    if step > 0:
        avg_time_per_step = elapsed_time / step
        remaining_steps = total_steps - step
        eta = avg_time_per_step * remaining_steps
        eta_str = format_time(eta)
    else:
        eta_str = "N/A"
    
    msg = (
        f"Step [{step}/{total_steps}] ({progress:.1f}%) | "
        f"Loss: {loss:.4f} | LR: {lr:.2e} | "
        f"Time: {format_time(elapsed_time)} | ETA: {eta_str}"
    )
    
    # 添加其他指标
    if kwargs:
        extra = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                           for k, v in kwargs.items()])
        msg += f" | {extra}"
    
    logger.info(msg)


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# 便捷函数
def get_logger(name: str = "MiniMind", level: str = "INFO") -> logging.Logger:
    """获取或创建 logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name, level)
    return logger


# 示例用法
if __name__ == "__main__":
    # 创建 logger
    logger = setup_logger(
        name="MiniMind",
        level="DEBUG",
        log_file="test",
        use_color=True
    )
    
    # 测试不同级别的日志
    logger.debug("这是一条 DEBUG 消息")
    logger.info("这是一条 INFO 消息")
    logger.warning("这是一条 WARNING 消息")
    logger.error("这是一条 ERROR 消息")
    logger.critical("这是一条 CRITICAL 消息")
    
    # 测试训练日志
    import time
    for step in range(1, 11):
        time.sleep(0.1)
        log_training_step(
            logger,
            step=step,
            total_steps=10,
            loss=1.0 / step,
            lr=1e-4,
            elapsed_time=step * 0.1,
            accuracy=0.9 + step * 0.01
        )
