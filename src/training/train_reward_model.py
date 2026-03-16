#!/usr/bin/env python3
"""Reward Model 训练脚本（参数化版本）"""
import os, argparse, logging, torch, deepspeed, sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import get_model_components
from src.models.reward_model import RewardModel
from src.data.rm_dataset import PairwiseDataset

logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S", level=logging.INFO)


@torch.no_grad()
def evaluate_reward_model(model_engine, val_loader, local_rank, margin=0.0):
    """评估奖励模型性能"""
    model_engine.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_reward_chosen = 0.0
    total_reward_rejected = 0.0
    total_reward_diff = 0.0
    num_batches = 0
    
    for c_ids, c_mask, r_ids, r_mask in val_loader:
        c_ids, c_mask, r_ids, r_mask = c_ids.cuda(), c_mask.cuda(), r_ids.cuda(), r_mask.cuda()
        r_c = model_engine(c_ids, c_mask)
        r_r = model_engine(r_ids, r_mask)
        
        loss = -torch.nn.functional.logsigmoid(r_c - r_r - margin).mean()
        acc = (r_c > r_r).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        total_reward_chosen += r_c.mean().item()
        total_reward_rejected += r_r.mean().item()
        total_reward_diff += (r_c - r_r).mean().item()
        num_batches += 1
    
    model_engine.train()
    
    if num_batches == 0:
        return {}
    
    return {
        'val_loss': total_loss / num_batches,
        'val_acc': total_acc / num_batches,
        'val_reward_chosen': total_reward_chosen / num_batches,
        'val_reward_rejected': total_reward_rejected / num_batches,
        'val_reward_diff': total_reward_diff / num_batches
    }


def get_args():
    p = argparse.ArgumentParser()
    # 模型结构参数（原有）
    p.add_argument("--vocab_size", type=int, default=6400)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    
    # 训练超参数（原有）
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32, help="批次大小 (RTX 4090: 32)")
    p.add_argument("--data_files", nargs="+", 
                   default=["minimind_dataset/hh_rlhf_cn/helpful_base_cn_train.jsonl"],
                   help="RLHF 训练数据文件")
    p.add_argument("--val_data_files", nargs="+",
                   default=["minimind_dataset/hh_rlhf_cn/helpful_base_cn_test.jsonl"],
                   help="RLHF 验证数据文件")
    p.add_argument("--tokenizer_path", type=str, default="my_tokenizer",
                   help="分词器保存路径（相对/绝对路径）")
    p.add_argument("--deepspeed", default="configs/deepspeed_zero2_rm.json",
                   help="Deepspeed配置文件路径")
    p.add_argument("--margin", type=float, default=0.0)
    
    # 评估和保存参数（原有）
    p.add_argument("--eval_steps", type=int, default=100, help="评估间隔")
    p.add_argument("--save_steps", type=int, default=200, help="保存间隔")
    p.add_argument("--log_interval", type=int, default=10, help="日志输出间隔")
    p.add_argument("--save_best", action="store_true", help="是否保存最佳模型")

    # 分布式参数（原有）
    p.add_argument("--local_rank", type=int, default=-1, 
                   help="Local rank passed from DeepSpeed launcher (无需手动设置)")
    
    # ========== 新增：路径配置参数（核心改造） ==========
    p.add_argument("--output_dir", type=str, default="ckpt/rm",
                   help="模型保存根目录（默认: ckpt/rm）")
    p.add_argument("--project_root", type=str, default=project_root,
                   help="项目根目录（自动识别，无需手动设置）")
    p.add_argument("--save_final", action="store_true", default=True,
                   help="是否保存最终模型（默认: True）")
    p.add_argument("--ckpt_prefix", type=str, default="step_",
                   help="Checkpoint文件前缀（默认: step_）")
    
    return p.parse_args()


def main():
    args = get_args()
    deepspeed.init_distributed()
    local_rank = deepspeed.comm.get_local_rank()
    torch.cuda.set_device(local_rank)

    components = get_model_components()
    MyGPT = components['MyGPT']

    # 加载分词器（支持绝对/相对路径）
    tokenizer_path = args.tokenizer_path
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.join(args.project_root, tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建基础模型和奖励模型（原有逻辑）
    base = MyGPT(vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                 n_kv_head=args.n_kv_head or args.n_head // 3, n_embd=args.n_embd,
                 block_size=args.block_size, dropout=args.dropout).cuda()
    rm = RewardModel(base, dropout=args.dropout).cuda()  # 传递 dropout 参数
    train_ds = PairwiseDataset(files=args.data_files, tokenizer=tokenizer, block_size=args.block_size)
    
    # 创建验证集（原有逻辑）
    val_ds = None
    val_loader = None
    if args.save_best and args.val_data_files:
        try:
            val_ds = PairwiseDataset(files=args.val_data_files, tokenizer=tokenizer, block_size=args.block_size)
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, 
                drop_last=False, pin_memory=True
            )
            if local_rank == 0:
                logging.info(f"验证集大小: {len(val_ds)}")
        except Exception as e:
            if local_rank == 0:
                logging.warning(f"无法加载验证集: {e}")
            val_ds = None
            val_loader = None

    # 处理deepspeed配置路径（兼容相对/绝对路径）
    if not os.path.isabs(args.deepspeed):
        args.deepspeed = os.path.join(args.project_root, args.deepspeed)
    
    # 初始化Deepspeed引擎（原有逻辑）
    model_engine, _, _, _ = deepspeed.initialize(model=rm, model_parameters=rm.parameters(), config=args.deepspeed)
    model_engine.train()
    
    # 数据加载器（原有逻辑）
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         pin_memory=True)

    # ========== 改造：使用参数化的输出目录 ==========
    # 处理输出目录（自动创建父目录）
    ckpt_rm_dir = args.output_dir
    if not os.path.isabs(ckpt_rm_dir):
        ckpt_rm_dir = os.path.join(args.project_root, ckpt_rm_dir)
    os.makedirs(ckpt_rm_dir, exist_ok=True)
    
    # 日志输出保存路径
    if local_rank == 0:
        logging.info(f"✅ 模型保存目录: {ckpt_rm_dir}")
        logging.info(f"✅ 分词器路径: {tokenizer_path}")
        logging.info(f"✅ Deepspeed配置: {args.deepspeed}")
    
    best_val_acc = 0.0
    step = 0
    
    # 训练循环（原有逻辑，仅路径改为参数化）
    for epoch in range(args.epochs):
        for c_ids, c_mask, r_ids, r_mask in loader:
            c_ids, c_mask, r_ids, r_mask = c_ids.cuda(), c_mask.cuda(), r_ids.cuda(), r_mask.cuda()
            r_c, r_r = model_engine(c_ids, c_mask), model_engine(r_ids, r_mask)
            loss = -torch.nn.functional.logsigmoid(r_c - r_r - args.margin).mean()
            acc = (r_c > r_r).float().mean()
            
            model_engine.backward(loss)
            model_engine.step()
            step += 1
            
            # 详细日志输出（原有）
            if step % args.log_interval == 0 and local_rank == 0:
                logging.info(
                    f"Epoch={epoch+1} Step={step} | "
                    f"Loss={loss.item():.4f} | "
                    f"Acc={acc.item():.4f} | "
                    f"R_chosen={r_c.mean().item():.4f} | "
                    f"R_rejected={r_r.mean().item():.4f} | "
                    f"R_diff={(r_c - r_r).mean().item():.4f}"
                )
            
            # 验证集评估（原有）
            if args.save_best and val_loader and step % args.eval_steps == 0 and local_rank == 0:
                eval_stats = evaluate_reward_model(model_engine, val_loader, local_rank, args.margin)
                if eval_stats:
                    logging.info(
                        f"验证集评估 | "
                        f"Val_Loss={eval_stats['val_loss']:.4f} | "
                        f"Val_Acc={eval_stats['val_acc']:.4f} | "
                        f"Val_R_chosen={eval_stats['val_reward_chosen']:.4f} | "
                        f"Val_R_rejected={eval_stats['val_reward_rejected']:.4f} | "
                        f"Val_R_diff={eval_stats['val_reward_diff']:.4f}"
                    )
                    
                    # 保存最佳模型（路径改为参数化）
                    if eval_stats['val_acc'] > best_val_acc:
                        best_val_acc = eval_stats['val_acc']
                        best_model_path = os.path.join(ckpt_rm_dir, "best_model.bin")
                        torch.save(
                            model_engine.module.state_dict(),
                            best_model_path
                        )
                        logging.info(f"保存最佳模型到: {best_model_path} (验证准确率: {best_val_acc:.4f})")
            
            # 定期保存（路径改为参数化）
            if step % args.save_steps == 0 and local_rank == 0:
                step_model_path = os.path.join(ckpt_rm_dir, f"{args.ckpt_prefix}{step}.bin")
                torch.save(
                    model_engine.module.state_dict(),
                    step_model_path
                )
                logging.info(f"保存Checkpoint到: {step_model_path}")

    # 保存最终模型（路径改为参数化，支持开关）
    if local_rank == 0 and args.save_final:
        # 保存Deepspeed checkpoint
        model_engine.save_checkpoint(ckpt_rm_dir, client_state={"epoch": args.epochs})
        
        # 保存pytorch格式模型
        final_model_path = os.path.join(ckpt_rm_dir, "pytorch_model.bin")
        torch.save(model_engine.module.state_dict(), final_model_path)
        
        # 保存分词器
        tokenizer.save_pretrained(ckpt_rm_dir)
        
        logging.info(f"✅ 最终模型保存到 -> {ckpt_rm_dir}")
        logging.info(f"   - PyTorch模型: {final_model_path}")
        logging.info(f"   - 分词器配置: {os.path.join(ckpt_rm_dir, 'tokenizer.json')}")
        
        if args.save_best and best_val_acc > 0:
            logging.info(f"🏆 最佳验证准确率: {best_val_acc:.4f} (模型: best_model.bin)")


if __name__ == "__main__":
    main()