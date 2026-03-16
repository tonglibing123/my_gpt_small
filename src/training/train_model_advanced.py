#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级预训练脚本 (Advanced Version)

⚠️ 注意: 这是高级版本，适合有经验的用户
对于初学者，建议使用 train_model.py

高级特性:
- 配置文件管理 (YAML)
- 结构化日志系统
- 错误处理和恢复
- 进度条显示
- 更详细的训练监控

使用方法:
    deepspeed --num_gpus 1 src/training/train_model_advanced.py --config configs/config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import deepspeed
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入新工具
from tools.config_loader import load_config, merge_args_with_config, save_config
from tools.logger import setup_logger, log_config, log_model_info, log_training_step
from tools.error_handler import (
    safe_execute, check_file_exists, check_cuda_available,
    handle_checkpoint_error, handle_data_error
)

# 导入模型和数据
from tools.utils import load_data_module, get_model_components

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MiniMind 预训练脚本（改进版）")
    
    # 配置文件
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    # 训练参数（可覆盖配置文件）
    parser.add_argument("--total_steps", type=int, help="总训练步数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--save_steps", type=int, help="保存间隔")
    parser.add_argument("--log_interval", type=int, help="日志间隔")
    
    # 模型参数
    parser.add_argument("--n_layer", type=int, help="Transformer 层数")
    parser.add_argument("--n_embd", type=int, help="嵌入维度")
    
    # 系统参数
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="设备")
    parser.add_argument("--log_level", type=str, 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="DeepSpeed local rank")
    parser.add_argument("--deepspeed_config", type=str,
                       help="DeepSpeed 配置文件")
    
    return parser.parse_args()


@safe_execute
def main():
    """主函数"""
    # 1. 解析参数
    args = get_args()
    
    # 2. 加载配置
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)
    
    # 3. 设置日志
    logger = setup_logger(
        name="MiniMind-Train",
        level=config.logging.level,
        log_file="train",
        log_dir=config.logging.log_dir,
        use_color=True
    )
    
    logger.info("=" * 60)
    logger.info("MiniMind 预训练脚本（改进版）")
    logger.info("=" * 60)
    
    # 4. 记录配置
    log_config(logger, config)
    
    # 5. 初始化 DeepSpeed
    try:
        deepspeed.init_distributed()
        logger.info("DeepSpeed 初始化成功")
    except Exception as e:
        logger.warning(f"DeepSpeed 初始化失败: {e}")
        logger.info("将使用单 GPU 或 CPU 模式")
    
    # 6. 检查环境
    if config.system.device == "cuda":
        check_cuda_available()
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.info("使用 CPU 模式")
    
    # 7. 设置随机种子
    torch.manual_seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.system.seed)
    logger.info(f"随机种子: {config.system.seed}")
    
    # 8. 加载数据
    logger.info("=" * 60)
    logger.info("加载数据...")
    logger.info("=" * 60)
    
    try:
        data_module = load_data_module()
        MMapDataset = data_module.MMapDataset
        
        # 检查数据文件
        train_data_path = os.path.join(config.data.output_dir, "train.bin")
        check_file_exists(train_data_path, "训练数据")
        
        # 加载完整数据集（注意：MMapDataset 第一个参数是列表）
        full_dataset = MMapDataset([train_data_path], block_size=config.model.block_size)
        
        # 划分训练集和验证集（5%验证集）
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * 0.05))
        n_train = n_total - n_val
        
        # 固定随机种子确保划分一致
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
        train_indices, val_indices = indices[:n_train], indices[n_train:]
        
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        logger.info(f"总样本数: {n_total:,}")
        logger.info(f"训练样本数: {len(train_dataset):,}")
        logger.info(f"验证样本数: {len(val_dataset):,}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.pretrain.batch_size,
            shuffle=True,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.pretrain.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory
        )
        
    except Exception as e:
        handle_data_error(str(e))
    
    # 9. 创建模型
    logger.info("=" * 60)
    logger.info("创建模型...")
    logger.info("=" * 60)
    
    try:
        model_components = get_model_components()
        MyGPT = model_components["MyGPT"]
        
        model = MyGPT(
            vocab_size=config.model.vocab_size,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            n_kv_head=config.model.n_kv_head,
            n_embd=config.model.n_embd,
            block_size=config.model.block_size,
            dropout=config.model.dropout,
            bias=config.model.bias,
            rope_theta=config.model.rope_theta
        )
        
        log_model_info(logger, model)
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        sys.exit(1)
    
    # 10. 初始化 DeepSpeed
    logger.info("=" * 60)
    logger.info("初始化 DeepSpeed...")
    logger.info("=" * 60)
    
    try:
        # 检查 DeepSpeed 配置文件
        ds_config = config.pretrain.deepspeed_config
        check_file_exists(ds_config, "DeepSpeed 配置")
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        
        logger.info("DeepSpeed 初始化成功")
        logger.info(f"使用配置: {ds_config}")
        
    except Exception as e:
        logger.error(f"DeepSpeed 初始化失败: {e}")
        sys.exit(1)
    
    # 11. 设置 TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE and config.logging.tensorboard:
        writer = SummaryWriter(log_dir=config.logging.tensorboard_dir)
        logger.info(f"TensorBoard 日志: {config.logging.tensorboard_dir}")
    
    # 12. 训练循环
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60)
    
    model_engine.train()
    global_step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    # 创建进度条
    pbar = tqdm(
        total=config.pretrain.total_steps,
        desc="训练进度",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    try:
        while global_step < config.pretrain.total_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if global_step >= config.pretrain.total_steps:
                    break
                
                # 数据移到设备
                x = x.to(model_engine.device)
                y = y.to(model_engine.device)
                
                # 前向传播
                _, loss, _ = model_engine(x, targets=y)
                
                # 反向传播
                model_engine.backward(loss)
                model_engine.step()
                
                global_step += 1
                pbar.update(1)
                
                # 记录日志
                if global_step % config.pretrain.log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = optimizer.param_groups[0]['lr']
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{lr:.2e}'
                    })
                    
                    # 记录到日志
                    log_training_step(
                        logger,
                        step=global_step,
                        total_steps=config.pretrain.total_steps,
                        loss=loss.item(),
                        lr=lr,
                        elapsed_time=elapsed
                    )
                    
                    # 记录到 TensorBoard
                    if writer:
                        writer.add_scalar('train/loss', loss.item(), global_step)
                        writer.add_scalar('train/lr', lr, global_step)
                
                # 验证
                if global_step % config.pretrain.eval_steps == 0:
                    logger.info("运行验证...")
                    val_loss = evaluate(model_engine, val_loader, logger)
                    
                    logger.info(f"验证损失: {val_loss:.4f}")
                    
                    if writer:
                        writer.add_scalar('val/loss', val_loss, global_step)
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(config.pretrain.checkpoint_dir, "best")
                        save_model(model_engine, save_path, config, logger)
                        logger.info(f"✅ 保存最佳模型 (val_loss: {val_loss:.4f})")
                
                # 保存检查点
                if global_step % config.pretrain.save_steps == 0:
                    save_path = os.path.join(
                        config.pretrain.checkpoint_dir,
                        f"step_{global_step}"
                    )
                    save_model(model_engine, save_path, config, logger)
                    logger.info(f"💾 保存检查点: {save_path}")
    
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    
    finally:
        pbar.close()
    
    # 13. 保存最终模型
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    
    final_path = os.path.join(config.pretrain.checkpoint_dir, "final")
    save_model(model_engine, final_path, config, logger)
    logger.info(f"✅ 最终模型已保存: {final_path}")
    
    # 14. 训练统计
    total_time = time.time() - start_time
    logger.info(f"总训练时间: {total_time / 3600:.2f} 小时")
    logger.info(f"平均每步时间: {total_time / global_step:.2f} 秒")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    
    if writer:
        writer.close()
    
    logger.info("🎉 训练成功完成！")


@torch.no_grad()
def evaluate(model_engine, val_loader, logger, max_batches=50):
    """在验证集上评估"""
    model_engine.eval()
    total_loss = 0.0
    count = 0
    
    # 创建验证进度条
    pbar = tqdm(
        enumerate(val_loader),
        total=min(max_batches, len(val_loader)),
        desc="验证中",
        ncols=80,
        leave=False
    )
    
    for i, (x, y) in pbar:
        if i >= max_batches:
            break
        
        x = x.to(model_engine.device)
        y = y.to(model_engine.device)
        
        _, loss, _ = model_engine(x, targets=y)
        total_loss += loss.item()
        count += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    model_engine.train()
    return total_loss / max(count, 1)


def save_model(model_engine, save_path, config, logger):
    """保存模型"""
    import json
    
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型权重
    model = model_engine.module
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # 保存配置
    model_config = {
        "vocab_size": config.model.vocab_size,
        "n_layer": config.model.n_layer,
        "n_head": config.model.n_head,
        "n_kv_head": config.model.n_kv_head,
        "n_embd": config.model.n_embd,
        "block_size": config.model.block_size,
        "dropout": config.model.dropout,
        "bias": config.model.bias,
        "rope_theta": config.model.rope_theta
    }
    
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # 保存完整配置
    save_config(config, os.path.join(save_path, "training_config.yaml"))


if __name__ == "__main__":
    main()
