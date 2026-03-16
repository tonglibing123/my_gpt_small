#!/usr/bin/env python3
"""
多轮对话推理脚本 - 支持 KV Cache 复用
演示如何通过保存和复用 KV Cache 来加速多轮对话
"""
import os, json, torch, argparse, time, sys
from transformers import PreTrainedTokenizerFast

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import get_model_components


class ChatBot:
    """
    多轮对话机器人
    
    核心功能：
    1. 保存历史对话的 KV Cache
    2. 新轮次只计算新 token，复用历史 KV Cache
    3. 自动管理上下文窗口（超过 block_size 时截断）
    """
    
    def __init__(self, model, tokenizer, max_history_len=None, device='cuda'):
        """
        Args:
            model: GPT 模型
            tokenizer: 分词器
            max_history_len: 最大历史长度（默认为 block_size）
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_history_len = max_history_len or model.block_size
        
        # 对话状态
        self.kv_cache = None  # 保存的 KV Cache
        self.history_ids = []  # 历史 token ids
        self.history_text = []  # 历史对话文本（用于显示）
        
        # 统计信息
        self.total_tokens_computed = 0  # 总计算的 token 数
        self.total_tokens_generated = 0  # 总生成的 token 数
        
    def chat(self, user_input, max_new_tokens=50, temperature=0.8, top_k=40, top_p=None):
        """
        进行一轮对话
        
        Args:
            user_input: 用户输入
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            
        Returns:
            assistant_response: 助手回复
            stats: 统计信息字典
        """
        # 编码用户输入
        user_ids = self.tokenizer.encode(user_input, add_special_tokens=False)
        
        # 检查是否需要截断历史
        if len(self.history_ids) + len(user_ids) > self.max_history_len:
            self._truncate_history()
        
        # 记录本轮计算的 token 数
        tokens_to_compute = len(user_ids)
        
        # 准备输入：只需要新的 token
        input_ids = torch.tensor([user_ids], dtype=torch.long).to(self.device)
        
        # 生成回复（使用保存的 KV Cache）
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self._generate_with_cache(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        generation_time = time.time() - start_time
        
        # 提取助手回复（去掉用户输入部分）
        assistant_ids = generated_ids[0, len(user_ids):].cpu().tolist()
        assistant_response = self.tokenizer.decode(assistant_ids, skip_special_tokens=True)
        
        # 更新历史
        self.history_ids.extend(user_ids + assistant_ids)
        self.history_text.append({"role": "user", "content": user_input})
        self.history_text.append({"role": "assistant", "content": assistant_response})
        
        # 更新统计
        self.total_tokens_computed += tokens_to_compute
        self.total_tokens_generated += len(assistant_ids)
        
        # 计算统计信息
        stats = {
            "tokens_computed": tokens_to_compute,
            "tokens_generated": len(assistant_ids),
            "generation_time": generation_time,
            "tokens_per_second": len(assistant_ids) / generation_time if generation_time > 0 else 0,
            "history_length": len(self.history_ids),
            "kv_cache_size_mb": self._estimate_cache_size(),
        }
        
        return assistant_response, stats
    
    def _generate_with_cache(self, input_ids, max_new_tokens, temperature, top_k, top_p):
        """
        使用 KV Cache 生成文本
        
        关键：
        1. 第一次调用：input_ids 是用户输入，kv_cache=None
        2. 后续调用：input_ids 只包含新生成的 token，kv_cache 包含历史
        """
        self.model.eval()
        generated = input_ids
        kv_cache = self.kv_cache  # 使用保存的历史 KV Cache
        
        for _ in range(max_new_tokens):
            # 准备输入：第一次是完整输入，后续只有最后一个 token
            if kv_cache is None:
                # 第一次：计算所有 token
                idx_cond = generated
            else:
                # 后续：只计算最后一个 token
                idx_cond = generated[:, -1:]
            
            # 前向传播（会更新 kv_cache）
            logits, _, kv_cache = self.model(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            
            # 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接生成的 token
            generated = torch.cat([generated, idx_next], dim=1)
            
            # EOS 停止
            if self.tokenizer.eos_token_id and (idx_next == self.tokenizer.eos_token_id).all():
                break
        
        # 保存更新后的 KV Cache
        self.kv_cache = kv_cache
        
        return generated
    
    def _truncate_history(self):
        """
        截断历史对话，保留最近的部分
        
        策略：保留最近 50% 的 token
        """
        keep_len = self.max_history_len // 2
        print(f"\n⚠️  历史对话过长，截断到最近 {keep_len} 个 token")
        
        # 截断历史 token
        self.history_ids = self.history_ids[-keep_len:]
        
        # 重置 KV Cache（需要重新计算）
        self.kv_cache = None
        
        # 截断历史文本（保留最近几轮）
        if len(self.history_text) > 4:
            self.history_text = self.history_text[-4:]
    
    def _estimate_cache_size(self):
        """估算 KV Cache 占用的内存（MB）"""
        if self.kv_cache is None:
            return 0.0
        
        total_bytes = 0
        for k, v in self.kv_cache:
            total_bytes += k.element_size() * k.nelement()
            total_bytes += v.element_size() * v.nelement()
        
        return total_bytes / (1024 * 1024)
    
    def reset(self):
        """重置对话状态"""
        self.kv_cache = None
        self.history_ids = []
        self.history_text = []
        self.total_tokens_computed = 0
        self.total_tokens_generated = 0
        print("\n✅ 对话已重置")
    
    def show_history(self):
        """显示对话历史"""
        if not self.history_text:
            print("\n📝 暂无对话历史")
            return
        
        print("\n" + "="*60)
        print("📝 对话历史")
        print("="*60)
        for i, msg in enumerate(self.history_text, 1):
            role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
            print(f"\n{role} (第{i//2 + 1}轮):")
            print(msg["content"])
        print("="*60)
    
    def show_stats(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 对话统计")
        print("="*60)
        print(f"总轮次: {len(self.history_text) // 2}")
        print(f"历史长度: {len(self.history_ids)} tokens")
        print(f"总计算量: {self.total_tokens_computed} tokens")
        print(f"总生成量: {self.total_tokens_generated} tokens")
        print(f"KV Cache: {self._estimate_cache_size():.2f} MB")
        
        # 计算节省的计算量
        if len(self.history_text) > 2:
            # 如果没有 KV Cache，需要重新计算所有历史
            without_cache = sum(len(self.tokenizer.encode(msg["content"], add_special_tokens=False)) 
                              for msg in self.history_text if msg["role"] == "user")
            saved = without_cache - self.total_tokens_computed
            saved_percent = (saved / without_cache * 100) if without_cache > 0 else 0
            print(f"节省计算: {saved} tokens ({saved_percent:.1f}%)")
        print("="*60)


def get_args():
    p = argparse.ArgumentParser(description="多轮对话推理脚本（支持 KV Cache 复用）")
    p.add_argument("--tokenizer_path", type=str, default="my_tokenizer", help="Tokenizer 路径")
    p.add_argument("--checkpoint", type=str, default="ckpt/pretrain/final", help="模型 checkpoint 路径")
    p.add_argument("--max_new_tokens", type=int, default=50, help="每轮最大生成 token 数")
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    p.add_argument("--top_k", type=int, default=40, help="Top-K 采样")
    p.add_argument("--top_p", type=float, default=None, help="Top-P 采样（可选）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    p.add_argument("--max_history_len", type=int, default=None, help="最大历史长度（默认为 block_size）")
    return p.parse_args()


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "="*60)
    print("🤖 多轮对话系统（支持 KV Cache 复用）")
    print("="*60)
    print("\n💡 功能说明：")
    print("  - 自动保存历史对话的 KV Cache")
    print("  - 新轮次只计算新 token，显著加速推理")
    print("  - 自动管理上下文窗口，避免溢出")
    print("\n📝 命令：")
    print("  - 直接输入文本进行对话")
    print("  - /history - 查看对话历史")
    print("  - /stats - 查看统计信息")
    print("  - /reset - 重置对话")
    print("  - /quit 或 /exit - 退出")
    print("="*60 + "\n")


def main():
    args = get_args()
    
    # 加载模型
    MyGPT = get_model_components()['MyGPT']
    
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 路径不存在: {args.tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # 加载配置
    config_path = os.path.join(args.checkpoint, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    print(f"📦 加载模型配置: {config_path}")
    
    # 创建模型
    model = MyGPT(
        vocab_size=len(tokenizer),
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_kv_head=cfg['n_kv_head'],
        n_embd=cfg['n_embd'],
        block_size=cfg['block_size']
    ).to(args.device)
    
    # 加载权重
    model_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ 模型加载成功: {model_path}")
    print(f"📊 模型参数量: {model.get_num_params() / 1e6:.2f}M")
    print(f"🔧 上下文长度: {cfg['block_size']} tokens")
    
    # 创建 ChatBot
    chatbot = ChatBot(
        model=model,
        tokenizer=tokenizer,
        max_history_len=args.max_history_len,
        device=args.device
    )
    
    # 打印欢迎信息
    print_welcome()
    
    # 交互式对话循环
    try:
        while True:
            # 获取用户输入
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ['/quit', '/exit']:
                print("\n👋 再见！")
                break
            elif user_input.lower() == '/reset':
                chatbot.reset()
                continue
            elif user_input.lower() == '/history':
                chatbot.show_history()
                continue
            elif user_input.lower() == '/stats':
                chatbot.show_stats()
                continue
            elif user_input.startswith('/'):
                print(f"❌ 未知命令: {user_input}")
                continue
            
            # 进行对话
            try:
                response, stats = chatbot.chat(
                    user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                # 显示回复
                print(f"\n🤖 助手: {response}")
                
                # 显示统计信息（简化版）
                print(f"\n💡 本轮统计: 计算 {stats['tokens_computed']} tokens, "
                      f"生成 {stats['tokens_generated']} tokens, "
                      f"耗时 {stats['generation_time']:.2f}s "
                      f"({stats['tokens_per_second']:.1f} tokens/s)")
                print(f"📊 历史长度: {stats['history_length']} tokens, "
                      f"KV Cache: {stats['kv_cache_size_mb']:.2f} MB\n")
                
            except Exception as e:
                print(f"\n❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n👋 对话被中断，再见！")
    
    # 显示最终统计
    if len(chatbot.history_text) > 0:
        print("\n" + "="*60)
        print("📊 最终统计")
        print("="*60)
        chatbot.show_stats()


if __name__ == "__main__":
    main()
