#!/usr/bin/env python3
"""PPO 训练脚本（修复 value_head 加载错误 + Tokenizer 兼容 + 自定义保存路径 + 完整 TensorBoard 可视化）"""
import os, argparse, logging, torch, deepspeed, sys, json, time
from transformers import PreTrainedTokenizerFast
import numpy as np
from typing import Any
# ========== TensorBoard 核心模块 ==========
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import get_model_components
from src.models.reward_model import RewardModel
from src.training.ppo_trainer import PPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S")


@torch.no_grad()
def evaluate_policy(policy, reward_model, tokenizer, eval_prompts, max_new_tokens=32, pad_token_id=None):
    """评估策略性能（返回详细奖励统计）"""
    policy.eval()
    total_reward = 0.0
    reward_list = []
    num_samples = 0
    
    for prompt in eval_prompts:
        try:
            # 生成响应
            response = policy.generate(
                prompt.unsqueeze(0).cuda(),
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                pad_token_id=pad_token_id or tokenizer.pad_token_id
            )
            
            # 计算奖励
            mask = (response != (pad_token_id or tokenizer.pad_token_id)).long()
            reward = reward_model(response, mask)
            reward_val = reward.item()
            total_reward += reward_val
            reward_list.append(reward_val)
            num_samples += 1
        except Exception as e:
            logging.warning(f"评估样本失败: {e}")
            continue
    
    policy.train()
    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
    # 返回平均奖励 + 奖励标准差（新增统计）
    reward_std = np.std(reward_list) if reward_list else 0.0
    return avg_reward, reward_std


class PromptDataset(torch.utils.data.Dataset):
    """PPO训练的Prompt数据集 - 从JSONL文件加载对话上下文"""
    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 64, sample_ratio: float = 1.0, seed: int = 42):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        if not (0 < sample_ratio <= 1.0):
            raise ValueError(f"sample_ratio 必须在 (0, 1] 范围内，当前值: {sample_ratio}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []

        # 加载JSONL数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 构建对话上下文作为prompt
                    context = obj.get("context", [])
                    if context:
                        # 拼接对话历史
                        prompt_text = "\n".join(f"{turn.get('role', '')}: {turn.get('text', '')}" for turn in context)
                        prompt_text += "\nassistant:"  # 添加assistant前缀，让模型生成回复
                        self.prompts.append(prompt_text)
                except Exception as e:
                    logging.warning(f"跳过无效行: {e}")
                    continue

        if not self.prompts:
            raise ValueError(f"没有成功加载任何prompt数据")

        # 采样
        subset_size = max(1, int(len(self.prompts) * sample_ratio))
        indices = np.random.default_rng(seed).choice(len(self.prompts), size=subset_size, replace=False)
        indices.sort()
        self.prompts = [self.prompts[i] for i in indices]

        logging.info(f"成功加载 {len(self.prompts)} 个prompt")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_text = self.prompts[idx]
        # 兼容 TokenizersBackend 的调用方式
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # 返回 (input_ids, attention_mask)
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="minimind_dataset/hh_rlhf_cn/helpful_base_cn_train.jsonl")
    p.add_argument("--sample_ratio", type=float, default=0.5)
    p.add_argument("--deepspeed", default="configs/deepspeed_zero2_ppo.json")
    p.add_argument("--vocab_size", type=int, default=6400)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=32)
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--eps_v", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16, help="批次大小 (RTX 4090: 16)")
    p.add_argument("--mini_batch", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--rm_ckpt", default="ckpt/rm/pytorch_model.bin")
    p.add_argument("--sft_ckpt", default="ckpt/pretrain/final/pytorch_model.bin")
    
    # 强化分词器路径参数
    p.add_argument("--tokenizer_path", 
                   type=str, 
                   default="my_tokenizer",
                   help="分词器路径（支持本地目录/ HuggingFace远程仓库，建议使用绝对路径）")
    
    # PPO 训练参数
    p.add_argument("--ppo_epochs", type=int, default=4, help="PPO 训练轮数")
    p.add_argument("--value_loss_coef", type=float, default=0.1, help="Value loss 系数")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="熵正则化系数")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    # 评估和保存参数
    p.add_argument("--eval_steps", type=int, default=50, help="评估间隔")
    p.add_argument("--save_best", action="store_true", help="是否保存最佳模型")
    # Deepspeed 参数
    p.add_argument("--local_rank", type=int, default=-1, 
                   help="Local rank passed from DeepSpeed launcher (无需手动设置)")
    # 自定义保存路径
    p.add_argument("--output_dir", type=str, default="ckpt/ppo",
                   help="模型保存根目录（默认: ckpt/ppo）")
    # TensorBoard 相关参数
    p.add_argument("--log_dir", type=str, default="runs/ppo_train",
                   help="TensorBoard 日志保存目录（默认: runs/ppo_train）")
    p.add_argument("--log_interval", type=int, default=10,
                   help="TensorBoard 日志记录间隔（默认每10步记录一次）")
    
    return p.parse_args()


def filter_value_head(state_dict):
    """移除 state_dict 中的 value_head 相关参数，适配 ref_policy"""
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("value_head.")}
    if len(filtered_dict) < len(state_dict):
        logging.info(f"已过滤 state_dict 中的 value_head 相关参数，原参数数: {len(state_dict)}, 过滤后: {len(filtered_dict)}")
    return filtered_dict


def main():
    args = get_args()
    deepspeed.init_distributed()
    local_rank = deepspeed.comm.get_local_rank()
    torch.cuda.set_device(local_rank)

    # ========== 初始化 TensorBoard（仅主进程） ==========
    writer = None
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        logging.info(f"✅ TensorBoard 日志保存到: {args.log_dir}")
        # 记录超参数（新增）
        writer.add_hparams({
            'batch_size': args.batch_size,
            'ppo_epochs': args.ppo_epochs,
            'entropy_coef': args.entropy_coef,
            'kl_beta': args.kl_beta,
            'clip_eps': args.clip_eps,
            'gamma': args.gamma,
            'lam': args.lam
        }, {})

    MyGPT = get_model_components()['MyGPT']
    
    # 加载分词器
    try:
        tokenizer_path = args.tokenizer_path
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(project_root, tokenizer_path)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True
        )
        logging.info(f"✅ 成功加载分词器: {tokenizer_path}")
        
        # 设置 pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.warning(f"分词器无pad_token，已使用eos_token替代: {tokenizer.eos_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logging.warning(f"分词器无pad_token和eos_token，已新增[PAD] token")
    except Exception as e:
        logging.error(f"❌ 加载分词器失败: {e}")
        sys.exit(1)

    # 初始化模型
    base_kwargs = dict(vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                       n_kv_head=args.n_kv_head or args.n_head // 3, n_embd=args.n_embd,
                       block_size=args.block_size, dropout=args.dropout)

    # 处理checkpoint路径
    sft_ckpt = args.sft_ckpt if os.path.isabs(args.sft_ckpt) else os.path.join(project_root, args.sft_ckpt)
    rm_ckpt = args.rm_ckpt if os.path.isabs(args.rm_ckpt) else os.path.join(project_root, args.rm_ckpt)

    # Policy + Value 模型
    policy = MyGPT(**base_kwargs).cuda()
    policy.value_head = torch.nn.Linear(args.n_embd, 1, bias=False).cuda()
    torch.nn.init.normal_(policy.value_head.weight, mean=0.0, std=0.02)
    sft_state_dict = torch.load(sft_ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(sft_state_dict, strict=False)

    # Reference Policy
    ref_policy = MyGPT(**base_kwargs).cuda()
    ref_policy.load_state_dict(filter_value_head(sft_state_dict), strict=False)
    for p in ref_policy.parameters():
        p.requires_grad = False
    ref_policy.eval()

    # Reward Model
    rm = RewardModel(MyGPT(**base_kwargs)).cuda()
    rm.load_state_dict(torch.load(rm_ckpt, map_location="cpu", weights_only=True), strict=False)
    for p in rm.parameters():
        p.requires_grad = False
    rm.eval()

    # 初始化 DeepSpeed
    deepspeed_config = args.deepspeed if os.path.isabs(args.deepspeed) else os.path.join(project_root, args.deepspeed)
    policy_engine, optimizer, _, _ = deepspeed.initialize(
        model=policy,
        model_parameters=policy.parameters(),
        config=deepspeed_config
    )
    trainer = PPOTrainer(
        policy=policy_engine, 
        ref_policy=ref_policy, 
        reward_model=rm, 
        value_model=policy_engine,
        tokenizer=tokenizer, 
        clip_eps=args.clip_eps, 
        beta=args.kl_beta, 
        eps_v=args.eps_v,
        gradient_accumulation_steps=args.grad_acc, 
        pad_token_id=tokenizer.pad_token_id
    )

    # 重写参考策略更新方法
    def safe_update_reference():
        policy_state_dict = policy_engine.module.state_dict()
        filtered_dict = filter_value_head(policy_state_dict)
        ref_policy.load_state_dict(filtered_dict, strict=False)
    trainer.update_reference = safe_update_reference

    # 加载数据集
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(project_root, args.data_path)
    prompt_ds = PromptDataset(data_path, tokenizer, max_length=args.prompt_len, sample_ratio=args.sample_ratio)
    prompt_loader = torch.utils.data.DataLoader(
        prompt_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )

    # 模型保存目录
    ckpt_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(project_root, args.output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    if local_rank == 0:
        logging.info(f"✅ 模型保存目录: {ckpt_dir}")
    
    # 准备评估数据（强制启用，确保奖励指标能记录）
    eval_prompts = []
    eval_data_path = "minimind_dataset/hh_rlhf_cn/helpful_base_cn_test.jsonl"
    eval_data_path = eval_data_path if os.path.isabs(eval_data_path) else os.path.join(project_root, eval_data_path)
    if os.path.exists(eval_data_path):
        try:
            eval_ds = PromptDataset(eval_data_path, tokenizer, max_length=args.prompt_len, sample_ratio=0.01)
            eval_size = min(10, len(eval_ds))
            for i in range(eval_size):
                prompt_ids, _ = eval_ds[i]
                eval_prompts.append(prompt_ids)
            logging.info(f"✅ 加载评估集样本数: {len(eval_prompts)}")
        except Exception as e:
            logging.warning(f"加载评估集失败: {e}，使用训练集前10个样本评估")
            eval_size = min(10, len(prompt_ds))
            for i in range(eval_size):
                prompt_ids, _ = prompt_ds[i]
                eval_prompts.append(prompt_ids)
    else:
        logging.warning(f"评估集不存在，使用训练集前10个样本评估")
        eval_size = min(10, len(prompt_ds))
        for i in range(eval_size):
            prompt_ids, _ = prompt_ds[i]
            eval_prompts.append(prompt_ids)
    
    # 训练状态初始化
    best_reward = -float('inf')
    iter_count = 0
    total_steps = len(prompt_loader) * args.epochs
    start_time = time.time()  # 记录训练开始时间

    # 主训练循环
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        for batch_idx, (prompt_ids, attention_mask) in enumerate(prompt_loader):
            # 收集 rollouts
            rollouts = trainer.collect(
                prompts=prompt_ids.cuda(), 
                max_new_tokens=args.gen_len
            )
            
            # PPO 更新
            step_stats = {}
            for ppo_ep in range(args.ppo_epochs):
                stats = trainer.ppo_epoch(
                    rollouts=rollouts, 
                    mini_batch=args.mini_batch,
                    gamma=args.gamma,
                    lam=args.lam,
                    value_loss_coef=args.value_loss_coef,
                    entropy_coef=args.entropy_coef
                )
                step_stats.update(stats)
            
            # 计算批次奖励
            batch_reward = rollouts["rewards"].mean().item()
            epoch_losses.append(batch_reward)
            
            # 每10步记录日志和TensorBoard
            if iter_count % args.log_interval == 0 and local_rank == 0:
                # 基础指标
                pg_loss = step_stats.get('pg_loss', 0.0)
                vf_loss = step_stats.get('vf_loss', 0.0)
                entropy = step_stats.get('entropy', 0.0)
                kl = step_stats.get('kl', 0.0)
                clipfrac = step_stats.get('clipfrac', 0.0)
                
                # 计算训练进度和速度
                elapsed_time = time.time() - start_time
                steps_per_sec = iter_count / elapsed_time if elapsed_time > 0 else 0
                progress = (iter_count / total_steps) * 100
                
                # 终端日志
                log_msg = (
                    f"Epoch {epoch+1}/{args.epochs} | Step {iter_count}/{total_steps} ({progress:.1f}%) | "
                    f"PG Loss: {pg_loss:.4f} | VF Loss: {vf_loss:.4f} | "
                    f"Entropy: {entropy:.4f} | KL: {kl:.4f} | "
                    f"Clipfrac: {clipfrac:.4f} | Avg Reward: {batch_reward:.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )
                logging.info(log_msg)

                # ========== 核心：完整的 TensorBoard 指标记录 ==========
                if writer:
                    # 1. 损失指标
                    writer.add_scalar('Loss/Policy_Gradient', pg_loss, iter_count)
                    writer.add_scalar('Loss/Value_Function', vf_loss, iter_count)
                    writer.add_scalar('Loss/Total_Loss', pg_loss + vf_loss, iter_count)
                    
                    # 2. 策略指标
                    writer.add_scalar('Policy/Entropy', entropy, iter_count)
                    writer.add_scalar('Policy/KL_Divergence', kl, iter_count)
                    writer.add_scalar('Policy/Clip_Fraction', clipfrac, iter_count)
                    
                    # 3. 奖励指标（核心补充）
                    writer.add_scalar('Reward/Batch_Average', batch_reward, iter_count)
                    writer.add_scalar('Reward/Rollout_Min', rollouts["rewards"].min().item(), iter_count)
                    writer.add_scalar('Reward/Rollout_Max', rollouts["rewards"].max().item(), iter_count)
                    writer.add_scalar('Reward/Rollout_Std', rollouts["rewards"].std().item(), iter_count)
                    
                    # 4. 训练效率指标（新增）
                    writer.add_scalar('Training/Steps_Per_Second', steps_per_sec, iter_count)
                    writer.add_scalar('Training/Progress_Percent', progress, iter_count)
                    writer.add_scalar('Training/Epoch', epoch + 1, iter_count)
                    
                    # 5. 学习率（关键补充：兼容DeepSpeed优化器）
                    if hasattr(optimizer, 'param_groups'):
                        lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('Training/Learning_Rate', lr, iter_count)
                    elif hasattr(policy_engine, 'optimizer'):
                        lr = policy_engine.optimizer.param_groups[0]['lr']
                        writer.add_scalar('Training/Learning_Rate', lr, iter_count)
                
            # 更新参考策略
            if iter_count % 5 == 0:
                trainer.update_reference()
            
            # 评估并保存最佳模型（每eval_steps步强制执行）
            if iter_count % args.eval_steps == 0 and local_rank == 0:
                eval_avg_reward, eval_reward_std = evaluate_policy(
                    policy=policy_engine.module,
                    reward_model=rm,
                    tokenizer=tokenizer,
                    eval_prompts=eval_prompts,
                    max_new_tokens=args.gen_len,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 记录评估奖励（核心补充）
                logging.info(f"📊 评估奖励 | 平均: {eval_avg_reward:.4f} | 标准差: {eval_reward_std:.4f} | 最佳: {best_reward:.4f}")
                if writer:
                    writer.add_scalar('Reward/Evaluation_Average', eval_avg_reward, iter_count)
                    writer.add_scalar('Reward/Evaluation_Std', eval_reward_std, iter_count)
                    writer.add_scalar('Reward/Best_Evaluation', best_reward, iter_count)
                
                # 保存最佳模型
                if eval_avg_reward > best_reward:
                    best_reward = eval_avg_reward
                    best_path = os.path.join(ckpt_dir, f"best_model_epoch{epoch}_step{iter_count}.pt")
                    torch.save(policy_engine.module.state_dict(), best_path)
                    logging.info(f"🏆 保存最佳模型到: {best_path} (奖励: {best_reward:.4f})")
            
            # 每100步保存检查点
            if iter_count % 100 == 0 and local_rank == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step{iter_count}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': iter_count,
                    'model_state_dict': policy_engine.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                    'best_reward': best_reward
                }, ckpt_path)
                logging.info(f"💾 保存检查点到: {ckpt_path}")
            
            iter_count += 1
        
        # Epoch结束统计
        epoch_time = time.time() - epoch_start
        avg_epoch_reward = np.mean(epoch_losses) if epoch_losses else 0.0
        if local_rank == 0:
            logging.info(f"📅 Epoch {epoch+1} 完成 | 耗时: {epoch_time:.2f}s | 平均奖励: {avg_epoch_reward:.4f}")
            if writer:
                writer.add_scalar('Training/Epoch_Time', epoch_time, epoch)
                writer.add_scalar('Reward/Epoch_Average', avg_epoch_reward, epoch)

    # 训练结束
    if local_rank == 0:
        # 保存最终模型
        final_path = os.path.join(ckpt_dir, "final_model.pt")
        torch.save(policy_engine.module.state_dict(), final_path)
        logging.info(f"🎉 训练完成 | 最终模型保存到: {final_path} | 最佳评估奖励: {best_reward:.4f}")
        
        # 关闭TensorBoard Writer
        if writer:
            writer.add_scalar('Reward/Final_Best', best_reward, iter_count)
            writer.close()
            logging.info(f"📝 TensorBoard 日志已保存至: {args.log_dir}")


if __name__ == "__main__":
    main()