#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预训练脚本：支持验证集评估、Checkpoint管理、训练监控"""
import os, argparse, torch, deepspeed, logging, json, sys
from torch.utils.data import DataLoader, Subset

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import load_data_module, get_model_components
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard 未安装，训练监控功能将被禁用")

data_module = load_data_module()
MMapDataset = data_module.MMapDataset

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser()
    # 模型
    p.add_argument("--vocab_size", type=int, default=6400)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    # 训练
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--train_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32, help="批次大小 (RTX 4090: 32)")
    # 数据 & DeepSpeed
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    p.add_argument("--deepspeed", type=str, default="configs/deepspeed_zero2.json")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=8, help="数据加载线程数 (Linux: 4-8)")
    # Checkpoint 管理
    p.add_argument("--save_steps", type=int, default=500, help="保存checkpoint的步数间隔")
    p.add_argument("--keep_recent", type=int, default=2, help="保留最近N个checkpoint")
    p.add_argument("--keep_best", type=int, default=1, help="保留最佳M个checkpoint")
    p.add_argument("--val_batches", type=int, default=20, help="验证时使用的batch数")
    # 训练监控
    p.add_argument("--log_dir", type=str, default="runs", help="TensorBoard 日志目录")
    p.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="日志级别")
    
    args = p.parse_args()
    
    # 参数验证
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError(f"val_ratio 必须在 [0, 1) 范围内，当前值: {args.val_ratio}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，当前值: {args.batch_size}")
    if args.total_steps <= 0:
        raise ValueError(f"total_steps 必须为正整数，当前值: {args.total_steps}")
    
    return args


def save_model(model_engine, folder):
    os.makedirs(folder, exist_ok=True)
    m = model_engine.module
    torch.save(m.state_dict(), os.path.join(folder, "pytorch_model.bin"))
    config = {k: getattr(m, k) for k in ["vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "block_size"]}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model saved to {folder}")


@torch.no_grad()
def evaluate(model_engine, val_loader, max_batches=50):
    """在验证集上计算loss"""
    model_engine.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.cuda(), y.cuda()
        _, loss, _ = model_engine(x, targets=y)
        total_loss += loss.item()
        count += 1
    model_engine.train()
    return total_loss / max(count, 1)


def main():
    args = get_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        deepspeed.init_distributed()
    except Exception as e:
        logger.error(f"DeepSpeed 初始化失败: {e}")
        logger.info("提示: 请使用 deepspeed 命令启动，例如: deepspeed --num_gpus 1 第4步-train_v3.py")
        raise
    
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(args.local_rank)

    components = get_model_components()
    MyGPT = components['MyGPT']
    CheckpointManager = components['CheckpointManager']
    
    model = MyGPT(
        vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head or args.n_head // 3, n_embd=args.n_embd,
        block_size=args.block_size, dropout=args.dropout
    ).cuda()

    # 数据：划分训练集和验证集
    bin_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".bin")]
    if not bin_files:
        raise FileNotFoundError(f"在 {args.data_dir} 目录下未找到 .bin 文件")
    
    full_ds = MMapDataset(bin_files, block_size=args.block_size)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    
    # 固定随机种子确保划分一致
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices, val_indices = indices[:n_train], indices[n_train:]
    
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    
    # 数据加载器配置（Windows 兼容）
    import platform
    is_windows = platform.system() == "Windows"
    
    # Windows 上 num_workers > 0 可能导致问题
    if is_windows:
        num_workers = 0  # Windows 上使用单进程
    else:
        # Linux/Mac 上使用多进程，但不超过合理范围
        max_workers = min(args.num_workers, 8)  # 最多8个worker
        num_workers = max_workers if len(train_ds) > max_workers else 0
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers, 
        persistent_workers=(num_workers > 0 and not is_windows),
        pin_memory=True, 
        pin_memory_device=f"cuda:{args.local_rank}"
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=0,  # 验证集使用单进程更稳定
        pin_memory=True, 
        pin_memory_device=f"cuda:{args.local_rank}"
    )
    
    logger.info(f"数据集: 训练 {n_train} 样本, 验证 {n_val} 样本")

    # DeepSpeed（学习率调度、梯度裁剪等都在配置文件中设置）
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=args.deepspeed)
    
    # Checkpoint 管理器
    ckpt_manager = CheckpointManager(ckpt_dir=os.path.join("ckpt", "pretrain"), keep_recent=args.keep_recent, keep_best=args.keep_best)
    
    # TensorBoard 监控
    writer = None
    if TENSORBOARD_AVAILABLE and args.local_rank <= 0:
        log_dir = os.path.join(args.log_dir, f"pretrain_{args.n_layer}L_{args.n_embd}D")
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard 日志目录: {log_dir}")
        logger.info(f"启动 TensorBoard: tensorboard --logdir={args.log_dir}")
    
    step = 0
    
    # 训练速度监控
    import time
    step_start_time = time.time()
    tokens_processed = 0

    # 训练循环
    model_engine.train()
    max_epochs = args.train_epochs or 1000
    
    try:
        for epoch in range(max_epochs):
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                
                # 前向传播
                _, loss, _ = model_engine(x, targets=y)
                
                # 反向传播
                model_engine.backward(loss)
                
                # 梯度范数监控（在优化器更新前）
                if args.local_rank <= 0 and step % args.log_interval == 0:
                    # 计算梯度范数
                    total_norm = 0.0
                    for p in model_engine.module.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5
                
                # 优化器更新
                model_engine.step()
                step += 1
                
                # 训练速度统计
                tokens_processed += x.numel()

                if step % args.log_interval == 0 and args.local_rank <= 0:
                    # 计算训练速度
                    step_end_time = time.time()
                    elapsed_time = step_end_time - step_start_time
                    tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                    samples_per_sec = (args.log_interval * args.batch_size) / elapsed_time if elapsed_time > 0 else 0
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"step={step}  loss={loss.item():.4f}  lr={current_lr:.2e}  "
                              f"tokens/s={tokens_per_sec:.0f}  samples/s={samples_per_sec:.1f}  "
                              f"grad_norm={grad_norm:.4f}")
                    
                    # TensorBoard 记录
                    if writer:
                        writer.add_scalar('train/loss', loss.item(), step)
                        writer.add_scalar('train/learning_rate', current_lr, step)
                        writer.add_scalar('train/epoch', epoch, step)
                        writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
                        writer.add_scalar('train/samples_per_sec', samples_per_sec, step)
                        writer.add_scalar('train/grad_norm', grad_norm, step)
                    
                    # 重置计时器
                    step_start_time = time.time()
                    tokens_processed = 0
                
                if step % args.save_steps == 0 and args.local_rank <= 0:
                    # 计算验证loss
                    val_loss = evaluate(model_engine, val_loader, max_batches=args.val_batches)
                    logger.info(f"step={step}  val_loss={val_loss:.4f}")
                    
                    # TensorBoard 记录验证指标
                    if writer:
                        writer.add_scalar('val/loss', val_loss, step)
                        # 记录训练/验证 loss 对比
                        writer.add_scalars('loss_comparison', {
                            'train': loss.item(),
                            'val': val_loss
                        }, step)
                    
                    # 保存checkpoint
                    ckpt_dir = os.path.join("ckpt", "pretrain", f"step_{step}")
                    model_engine.save_checkpoint(ckpt_dir, client_state={"step": step, "val_loss": val_loss})
                    save_model(model_engine, ckpt_dir)
                    
                    # 记录并清理旧checkpoint
                    ckpt_manager.record(step=step, val_loss=val_loss, ckpt_path=ckpt_dir)
                    ckpt_manager.cleanup()
                    
                if step >= args.total_steps:
                    break
            if step >= args.total_steps:
                break
    
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
        if args.local_rank <= 0:
            # 保存中断时的 checkpoint
            interrupt_dir = os.path.join("ckpt", "pretrain", f"interrupted_step_{step}")
            logger.info(f"保存中断 checkpoint: {interrupt_dir}")
            model_engine.save_checkpoint(interrupt_dir, client_state={"step": step, "interrupted": True})
            save_model(model_engine, interrupt_dir)
        raise
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        if args.local_rank <= 0:
            # 保存错误时的 checkpoint
            error_dir = os.path.join("ckpt", "pretrain", f"error_step_{step}")
            logger.info(f"保存错误 checkpoint: {error_dir}")
            try:
                model_engine.save_checkpoint(error_dir, client_state={"step": step, "error": str(e)})
                save_model(model_engine, error_dir)
            except:
                pass
        raise

    if args.local_rank <= 0:
        # 最终验证
        final_val_loss = evaluate(model_engine, val_loader, max_batches=args.val_batches)
        logger.info(f"最终验证 loss={final_val_loss:.4f}")
        
        # TensorBoard 记录最终指标
        if writer:
            writer.add_scalar('val/final_loss', final_val_loss, step)
            writer.add_text('training/summary', f'Final loss: {final_val_loss:.4f}, Total steps: {step}')
            writer.close()
        
        model_engine.save_checkpoint(os.path.join("ckpt", "pretrain", "final"), client_state={"step": step, "val_loss": final_val_loss})
        save_model(model_engine, os.path.join("ckpt", "pretrain", "final"))
        
        # 输出checkpoint摘要
        summary = ckpt_manager.get_summary()
        logger.info(f"Checkpoint摘要: 共保存 {summary['total_saved']} 个, 最佳 step={summary['best_step']} val_loss={summary['best_val_loss']:.4f}")
    
    # 清理资源
    if hasattr(full_ds, 'close'):
        try:
            full_ds.close()
        except Exception as e:
            logger.warning(f"关闭数据集时出错: {e}")
    
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
