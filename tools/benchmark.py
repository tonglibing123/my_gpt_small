#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基准测试工具

提供标准化的评估指标，用于客观评估模型性能
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Benchmark:
    """基准测试类"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        初始化基准测试
        
        参数:
            model: 语言模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, texts: List[str]) -> float:
        """
        计算困惑度 (Perplexity)
        
        困惑度越低，模型越好
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="计算困惑度"):
                # 编码
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                
                if input_ids.size(1) < 2:
                    continue
                
                # 前向传播
                logits, loss, _ = self.model(input_ids, targets=input_ids)
                
                if loss is not None:
                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
        
        # 计算平均loss和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def compute_accuracy(self, texts: List[str]) -> float:
        """
        计算下一个token预测准确率
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="计算准确率"):
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                
                if input_ids.size(1) < 2:
                    continue
                
                # 前向传播
                logits, _, _ = self.model(input_ids[:, :-1])
                
                # 预测
                predictions = torch.argmax(logits, dim=-1)
                targets = input_ids[:, 1:]
                
                # 统计
                correct += (predictions == targets).sum().item()
                total += targets.numel()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def generate_samples(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """
        生成样本
        """
        samples = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="生成样本"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # 生成
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=50
                )
                
                # 解码
                text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                samples.append(text)
        
        return samples
    
    def compute_diversity(self, texts: List[str]) -> Dict[str, float]:
        """
        计算生成多样性
        
        指标:
        - distinct-1: 不同unigram的比例
        - distinct-2: 不同bigram的比例
        """
        all_unigrams = []
        all_bigrams = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            
            # Unigrams
            all_unigrams.extend(tokens)
            
            # Bigrams
            for i in range(len(tokens) - 1):
                all_bigrams.append((tokens[i], tokens[i+1]))
        
        # 计算distinct
        distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        
        return {
            'distinct-1': distinct_1,
            'distinct-2': distinct_2
        }
    
    def run_full_benchmark(self, test_texts: List[str], test_prompts: List[str]) -> Dict:
        """
        运行完整基准测试
        
        参数:
            test_texts: 测试文本（用于困惑度和准确率）
            test_prompts: 测试提示（用于生成）
            
        返回:
            评估结果字典
        """
        print("\n" + "="*60)
        print("开始基准测试")
        print("="*60)
        
        results = {}
        
        # 1. 困惑度
        print("\n1. 计算困惑度...")
        perplexity = self.compute_perplexity(test_texts)
        results['perplexity'] = perplexity
        print(f"   困惑度: {perplexity:.2f}")
        
        # 2. 准确率
        print("\n2. 计算准确率...")
        accuracy = self.compute_accuracy(test_texts)
        results['accuracy'] = accuracy
        print(f"   准确率: {accuracy:.2%}")
        
        # 3. 生成样本
        print("\n3. 生成样本...")
        samples = self.generate_samples(test_prompts, max_new_tokens=30)
        results['samples'] = samples
        print(f"   生成了 {len(samples)} 个样本")
        
        # 4. 多样性
        print("\n4. 计算多样性...")
        diversity = self.compute_diversity(samples)
        results['diversity'] = diversity
        print(f"   Distinct-1: {diversity['distinct-1']:.4f}")
        print(f"   Distinct-2: {diversity['distinct-2']:.4f}")
        
        print("\n" + "="*60)
        print("基准测试完成")
        print("="*60)
        
        return results


def load_test_data(data_path: str, max_samples: int = 100) -> Tuple[List[str], List[str]]:
    """
    加载测试数据
    
    返回:
        (test_texts, test_prompts)
    """
    test_texts = []
    test_prompts = []
    
    if not os.path.exists(data_path):
        print(f"警告: 测试数据不存在: {data_path}")
        print("使用默认测试数据")
        
        # 默认测试数据
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个重要分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的生成能力",
            "强化学习可以让模型学习人类偏好"
        ] * 20
        
        test_prompts = [
            "人工智能",
            "深度学习",
            "自然语言处理",
            "大语言模型",
            "强化学习"
        ]
        
        return test_texts, test_prompts
    
    # 从文件加载
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    text = obj['text']
                    test_texts.append(text)
                    
                    # 使用前10个字符作为prompt
                    if len(text) > 10:
                        test_prompts.append(text[:10])
            except:
                continue
    
    return test_texts, test_prompts


def save_results(results: Dict, output_path: str):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, list, dict)):
            serializable_results[key] = value
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = str(value)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    from transformers import PreTrainedTokenizerFast
    from tools.utils import create_model
    
    parser = argparse.ArgumentParser(description="模型基准测试")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--tokenizer", type=str, default="my_tokenizer", help="分词器路径")
    parser.add_argument("--test_data", type=str, default="minimind_dataset/pretrain_hq.jsonl", help="测试数据路径")
    parser.add_argument("--max_samples", type=int, default=100, help="最大测试样本数")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="输出路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_kv_head", type=int, default=2)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("模型基准测试工具")
    print("="*60)
    
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    print(f"✓ 分词器加载完成，词表大小: {len(tokenizer)}")
    
    # 加载模型
    print("\n加载模型...")
    model = create_model(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        block_size=args.block_size
    )
    
    # 加载checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"✓ 模型加载完成，参数量: {model.get_num_params() / 1e6:.2f}M")
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_texts, test_prompts = load_test_data(args.test_data, args.max_samples)
    print(f"✓ 测试文本: {len(test_texts)} 条")
    print(f"✓ 测试提示: {len(test_prompts)} 条")
    
    # 运行基准测试
    benchmark = Benchmark(model, tokenizer, args.device)
    results = benchmark.run_full_benchmark(test_texts, test_prompts)
    
    # 保存结果
    save_results(results, args.output)
    
    print("\n" + "="*60)
    print("基准测试总结")
    print("="*60)
    print(f"困惑度: {results['perplexity']:.2f}")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"Distinct-1: {results['diversity']['distinct-1']:.4f}")
    print(f"Distinct-2: {results['diversity']['distinct-2']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

