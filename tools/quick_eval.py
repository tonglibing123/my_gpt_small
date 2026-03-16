#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速评估脚本 - 教学简化版

只评估最核心的指标，适合快速验证训练效果
- 困惑度 (Perplexity)
- 生成质量
- 简单直观的输出

使用方法:
    python tools/quick_eval.py
    python tools/quick_eval.py --checkpoint ckpt/pretrain/step_1000
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """加载模型和分词器"""
    print_header("加载模型")
    
    # 加载分词器
    from transformers import PreTrainedTokenizerFast
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"分词器不存在: {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"✓ 分词器: {len(tokenizer)} tokens")
    
    # 加载模型配置
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    # 创建模型
    from tools.utils import get_model_components
    MyGPT = get_model_components()['MyGPT']
    
    model = MyGPT(
        vocab_size=cfg['vocab_size'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_kv_head=cfg['n_kv_head'],
        n_embd=cfg['n_embd'],
        block_size=cfg['block_size']
    ).to(device)
    
    # 加载权重
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ 模型: {model.get_num_params() / 1e6:.2f}M 参数")
    print(f"✓ 配置: {cfg['n_layer']}层, {cfg['n_embd']}维")
    
    return model, tokenizer, cfg


@torch.no_grad()
def quick_perplexity(model, tokenizer, device, test_texts=None):
    """快速计算困惑度（使用少量样本）"""
    print_header("计算困惑度")
    
    if test_texts is None:
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个重要分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的生成能力",
            "强化学习可以让模型学习人类偏好",
            "计算机视觉在图像识别中应用广泛",
            "神经网络是深度学习的基础",
            "Transformer架构改变了NLP领域",
            "预训练模型可以迁移到下游任务",
            "注意力机制是Transformer的核心"
        ]
    
    total_loss = 0.0
    total_tokens = 0
    
    print(f"使用 {len(test_texts)} 个测试样本...")
    
    for text in test_texts:
        # 编码
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        
        x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(device)
        
        # 前向传播
        _, loss, _ = model(x, targets=y)
        
        total_loss += loss.item() * len(ids)
        total_tokens += len(ids)
    
    # 计算困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n结果:")
    print(f"  平均 Loss: {avg_loss:.4f}")
    print(f"  困惑度 (Perplexity): {perplexity:.2f}")
    
    # 解释困惑度
    print(f"\n解释:")
    if perplexity < 20:
        print(f"  ✅ 优秀 - 模型训练得很好")
    elif perplexity < 50:
        print(f"  ✓ 良好 - 模型表现不错")
    elif perplexity < 100:
        print(f"  ⚠ 一般 - 模型还需要更多训练")
    else:
        print(f"  ❌ 较差 - 模型可能训练不足或有问题")
    
    return perplexity, avg_loss


@torch.no_grad()
def quick_generation(model, tokenizer, device, prompts=None):
    """快速生成测试"""
    print_header("生成质量测试")
    
    if prompts is None:
        prompts = [
            "人工智能",
            "深度学习是",
            "自然语言处理",
            "大语言模型",
            "机器学习的应用"
        ]
    
    print(f"生成 {len(prompts)} 个样本...\n")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        # 编码
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 生成
        generated = model.generate(
            x,
            max_new_tokens=30,
            temperature=0.8,
            top_k=40,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
        )
        
        # 解码
        output = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        
        print(f"样本 {i}:")
        print(f"  输入: {prompt}")
        print(f"  输出: {output}")
        print()
        
        results.append({
            "prompt": prompt,
            "output": output
        })
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速评估工具 - 教学简化版")
    parser.add_argument("--checkpoint", type=str, default="ckpt/pretrain/final",
                       help="模型checkpoint路径")
    parser.add_argument("--tokenizer", type=str, default="my_tokenizer",
                       help="分词器路径")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="设备")
    parser.add_argument("--output", type=str, default=None,
                       help="保存结果到JSON文件（可选）")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  快速评估工具 - 教学简化版")
    print("="*60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Device: {args.device}")
    
    try:
        # 1. 加载模型和分词器
        model, tokenizer, cfg = load_model_and_tokenizer(
            args.checkpoint, args.tokenizer, args.device
        )
        
        # 2. 计算困惑度
        perplexity, loss = quick_perplexity(model, tokenizer, args.device)
        
        # 3. 生成测试
        generation_results = quick_generation(model, tokenizer, args.device)
        
        # 4. 总结
        print_header("评估总结")
        print(f"\n模型: {args.checkpoint}")
        print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
        print(f"困惑度: {perplexity:.2f}")
        print(f"平均Loss: {loss:.4f}")
        print(f"生成样本数: {len(generation_results)}")
        
        # 5. 保存结果（可选）
        if args.output:
            results = {
                "checkpoint": args.checkpoint,
                "model_config": cfg,
                "perplexity": perplexity,
                "loss": loss,
                "generation_samples": generation_results
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ 结果已保存到: {args.output}")
        
        print("\n" + "="*60)
        print("  ✅ 评估完成！")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示:")
        print("  1. 确保已训练模型: python src/training/train_model.py")
        print("  2. 确保已训练分词器: python src/training/train_tokenizer.py")
        return 1
    
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
