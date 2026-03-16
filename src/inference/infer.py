#!/usr/bin/env python3
"""推理脚本（简化版，不使用 DeepSpeed）"""
import os, json, torch, argparse, sys
from transformers import PreTrainedTokenizerFast

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import get_model_components


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", type=str, default="my_tokenizer")
    p.add_argument("--checkpoint", type=str, default="ckpt/pretrain/final")
    p.add_argument("--prompt", type=str, default="人工智能在未来十年")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = p.parse_args()
    
    # 参数验证
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens 必须为正整数，当前值: {args.max_new_tokens}")
    if args.temperature <= 0:
        raise ValueError(f"temperature 必须为正数，当前值: {args.temperature}")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，当前值: {args.top_k}")
    if args.top_p is not None and not (0 < args.top_p <= 1.0):
        raise ValueError(f"top_p 必须在 (0, 1] 范围内，当前值: {args.top_p}")
    
    return args


def main():
    args = get_args()
    MyGPT = get_model_components()['MyGPT']

    # 加载 tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 路径不存在: {args.tokenizer_path}")
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    vocab_size = len(tok)

    # 加载配置
    config_path = os.path.join(args.checkpoint, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"从配置文件加载模型参数: {config_path}")
    else:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 创建模型
    model = MyGPT(
        vocab_size=vocab_size,
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
    print(f"模型加载成功: {model_path}")
    print(f"模型参数量: {model.get_num_params() / 1e6:.2f}M")

    # 编码输入
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("输入 prompt 编码后为空，请检查 prompt 内容")
    
    # 检查 token 是否在词表范围内
    if max(ids) >= vocab_size:
        raise ValueError(f"Token ID {max(ids)} 超出词表大小 {vocab_size}")
    
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(args.device)
    
    # 检查输入长度
    if x.size(1) > cfg['block_size']:
        print(f"警告: 输入长度 {x.size(1)} 超过 block_size {cfg['block_size']}，将截断到最后 {cfg['block_size']} 个 token")
        x = x[:, -cfg['block_size']:]
    
    if x.size(1) == 0:
        raise ValueError("截断后输入为空")

    # 生成
    print(f"\n{'='*60}")
    print(f"输入: {args.prompt}")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'top_k': args.top_k,
            'temperature': args.temperature,
            'eos_token_id': tok.eos_token_id
        }
        if args.top_p:
            gen_kwargs['top_p'] = args.top_p
        
        generated = model.generate(x, **gen_kwargs)

    output = tok.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
    print(f"输出: {output}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
