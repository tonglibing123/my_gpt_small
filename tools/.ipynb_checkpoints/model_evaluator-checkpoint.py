#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本：评估模型性能
- 困惑度 (Perplexity)
- 准确率
- 生成质量
"""
import os
import json
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="ckpt/pretrain/final")
    p.add_argument("--tokenizer_path", type=str, default="my_tokenizer")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=1000, help="最大评估样本数")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="evaluation_results.json")
    return p.parse_args()


def load_model(checkpoint_path, device):
    """加载模型"""
    from tools.utils import get_model_components
    
    # 加载配置
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    print(f"加载模型配置: {cfg}")
    
    # 创建模型
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
    
    print(f"✅ 模型加载成功")
    print(f"   参数量: {model.get_num_params() / 1e6:.2f}M")
    
    return model, cfg


def load_dataset(data_dir, block_size, max_samples=None):
    """加载数据集"""
    from tools.utils import load_data_module
    
    data_module = load_data_module()
    MMapDataset = data_module.MMapDataset
    
    bin_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith(".bin")]
    
    if not bin_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下未找到 .bin 文件")
    
    dataset = MMapDataset(bin_files, block_size=block_size)
    
    # 限制样本数
    if max_samples and max_samples < len(dataset):
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"✅ 数据集加载成功: {len(dataset)} 样本")
    
    return dataset


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device):
    """计算困惑度"""
    print("\n" + "=" * 60)
    print("📊 计算困惑度 (Perplexity)")
    print("=" * 60)
    
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in tqdm(dataloader, desc="评估中"):
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        
        # 统计非 padding token
        valid_tokens = (y != -1).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"\n结果:")
    print(f"  - 平均 Loss: {avg_loss:.4f}")
    print(f"  - 困惑度 (Perplexity): {perplexity:.2f}")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens
    }


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    """计算准确率"""
    print("\n" + "=" * 60)
    print("🎯 计算预测准确率")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    for x, y in tqdm(dataloader, desc="评估中"):
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        
        # 预测下一个 token
        predictions = logits.argmax(dim=-1)
        
        # 只统计非 padding token
        mask = (y != -1)
        correct += (predictions == y)[mask].sum().item()
        total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n结果:")
    print(f"  - 准确率: {accuracy * 100:.2f}%")
    print(f"  - 正确预测: {correct:,} / {total:,}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, device, prompts=None):
    """评估生成质量"""
    print("\n" + "=" * 60)
    print("✍️  评估生成质量")
    print("=" * 60)
    
    if prompts is None:
        prompts = [
            "人工智能",
            "机器学习是",
            "深度学习的应用包括",
            "自然语言处理",
            "计算机视觉",
        ]
    
    results = []
    generated_texts = []
    
    for prompt in prompts:
        print(f"\n输入: {prompt}")
        
        # 编码
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 生成
        generated = model.generate(
            x,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 解码
        output = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        print(f"输出: {output}")
        
        generated_texts.append(output)
        
        results.append({
            "prompt": prompt,
            "output": output,
            "length": len(generated[0]) - len(ids)
        })
    
    # 计算 Diversity Metrics
    diversity_metrics = calculate_diversity_metrics(generated_texts)
    
    print(f"\n生成多样性指标:")
    print(f"  - Distinct-1: {diversity_metrics['distinct_1']:.4f}")
    print(f"  - Distinct-2: {diversity_metrics['distinct_2']:.4f}")
    print(f"  - Distinct-3: {diversity_metrics['distinct_3']:.4f}")
    print(f"  - 平均长度: {diversity_metrics['avg_length']:.1f} tokens")
    print(f"  - 词表使用率: {diversity_metrics['vocab_usage']:.2%}")
    
    return {
        "samples": results,
        "diversity": diversity_metrics
    }


def calculate_diversity_metrics(texts):
    """计算生成文本的多样性指标"""
    if not texts:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "distinct_3": 0.0,
            "avg_length": 0.0,
            "vocab_usage": 0.0
        }
    
    all_tokens = []
    all_unigrams = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        # 简单分词（按空格和标点）
        import re
        tokens = re.findall(r'\w+', text.lower())
        all_tokens.extend(tokens)
        
        # n-grams
        if len(tokens) >= 1:
            all_unigrams.extend(tokens)
        if len(tokens) >= 2:
            all_bigrams.extend([tuple(tokens[i:i+2]) for i in range(len(tokens)-1)])
        if len(tokens) >= 3:
            all_trigrams.extend([tuple(tokens[i:i+3]) for i in range(len(tokens)-2)])
    
    # 计算 Distinct-n
    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    distinct_3 = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0.0
    
    # 平均长度
    avg_length = len(all_tokens) / len(texts) if texts else 0.0
    
    # 词表使用率
    vocab_usage = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "avg_length": avg_length,
        "vocab_usage": vocab_usage,
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens))
    }


def compare_checkpoints(checkpoint_paths, tokenizer_path, data_dir, device):
    """比较多个 checkpoint"""
    print("\n" + "=" * 60)
    print("📊 比较多个 Checkpoint")
    print("=" * 60)
    
    results = []
    
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            print(f"⚠️  跳过不存在的 checkpoint: {ckpt_path}")
            continue
        
        print(f"\n评估: {ckpt_path}")
        
        try:
            # 加载模型
            model, cfg = load_model(ckpt_path, device)
            
            # 加载数据
            dataset = load_dataset(data_dir, cfg['block_size'], max_samples=500)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            # 评估
            ppl_result = evaluate_perplexity(model, dataloader, device)
            
            results.append({
                "checkpoint": ckpt_path,
                "perplexity": ppl_result["perplexity"],
                "loss": ppl_result["loss"]
            })
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
    
    # 打印比较结果
    if results:
        print("\n" + "=" * 60)
        print("📊 比较结果")
        print("=" * 60)
        print(f"\n{'Checkpoint':<40} {'Perplexity':>12} {'Loss':>10}")
        print("-" * 65)
        for r in sorted(results, key=lambda x: x['perplexity']):
            print(f"{r['checkpoint']:<40} {r['perplexity']:>12.2f} {r['loss']:>10.4f}")
    
    return results


def main():
    args = get_args()
    
    print("\n" + "🔍 模型评估工具".center(60, "="))
    print()
    
    # 加载 tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 不存在: {args.tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    print(f"✅ Tokenizer 加载成功: {len(tokenizer)} tokens")
    
    # 加载模型
    model, cfg = load_model(args.checkpoint, args.device)
    
    # 加载数据集
    dataset = load_dataset(args.data_dir, cfg['block_size'], args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 评估
    results = {}
    
    # 1. 困惑度
    ppl_result = evaluate_perplexity(model, dataloader, args.device)
    results.update(ppl_result)
    
    # 2. 准确率
    acc_result = evaluate_accuracy(model, dataloader, args.device)
    results.update(acc_result)
    
    # 3. 生成质量
    gen_results = evaluate_generation(model, tokenizer, args.device)
    results["generation_samples"] = gen_results["samples"]
    results["diversity_metrics"] = gen_results["diversity"]
    
    # 保存结果
    results["checkpoint"] = args.checkpoint
    results["model_config"] = cfg
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)
    print(f"\n结果已保存: {args.output}")
    print(f"\n总结:")
    print(f"  - 困惑度: {results['perplexity']:.2f}")
    print(f"  - 准确率: {results['accuracy'] * 100:.2f}%")
    print(f"  - 平均 Loss: {results['loss']:.4f}")
    print(f"\n生成多样性:")
    print(f"  - Distinct-1: {results['diversity_metrics']['distinct_1']:.4f}")
    print(f"  - Distinct-2: {results['diversity_metrics']['distinct_2']:.4f}")
    print(f"  - Distinct-3: {results['diversity_metrics']['distinct_3']:.4f}")
    print()


if __name__ == "__main__":
    main()
