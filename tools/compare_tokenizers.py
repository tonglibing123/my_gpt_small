#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器对比实验脚本
对比不同配置的分词器性能
"""
import sys
import os
from pathlib import Path
import argparse
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compare_vocab_sizes(test_texts, vocab_sizes=[1000, 2000, 5000, 10000]):
    """对比不同词表大小"""
    print("\n" + "="*70)
    print("  实验1: 词表大小对比")
    print("="*70)
    
    print(f"\n测试配置:")
    print(f"  - 词表大小: {vocab_sizes}")
    print(f"  - 测试文本数: {len(test_texts)}")
    
    from tools.tokenizer_utils import create_test_tokenizer
    
    results = []
    
    print(f"\n{'词表大小':<12} {'训练时间':<12} {'Token数':<12} {'压缩率':<12} {'词表利用率'}")
    print("-"*70)
    
    for vocab_size in vocab_sizes:
        # 训练分词器
        start_time = time.time()
        tokenizer = create_test_tokenizer(
            vocab_size=vocab_size,
            texts=test_texts * 10,  # 重复以提供足够数据
            min_frequency=2,
            show_progress=False
        )
        train_time = time.time() - start_time
        
        # 测试编码
        total_chars = 0
        total_tokens = 0
        
        for text in test_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            total_chars += len(text)
            total_tokens += len(ids)
        
        compression = total_tokens / total_chars if total_chars > 0 else 0
        
        # 词表利用率（简化计算）
        vocab = tokenizer.get_vocab()
        utilization = len(vocab) / vocab_size * 100
        
        print(f"{vocab_size:<12} {train_time:<12.2f}s {total_tokens:<12} {compression:<12.3f} {utilization:>10.1f}%")
        
        results.append({
            "vocab_size": vocab_size,
            "train_time": train_time,
            "total_tokens": total_tokens,
            "compression": compression,
            "utilization": utilization
        })
    
    # 分析
    print(f"\n分析:")
    best_compression = min(results, key=lambda x: x['compression'])
    fastest_train = min(results, key=lambda x: x['train_time'])
    
    print(f"  - 最佳压缩率: vocab_size={best_compression['vocab_size']} (压缩率={best_compression['compression']:.3f})")
    print(f"  - 最快训练: vocab_size={fastest_train['vocab_size']} (时间={fastest_train['train_time']:.2f}s)")
    print(f"  - 推荐配置: vocab_size=5000-10000 (平衡性能和效率)")
    
    return results


def compare_min_frequencies(test_texts, min_frequencies=[2, 5, 10, 20, 50]):
    """对比不同最小频率"""
    print("\n" + "="*70)
    print("  实验2: 最小频率对比")
    print("="*70)
    
    print(f"\n测试配置:")
    print(f"  - 最小频率: {min_frequencies}")
    print(f"  - 词表大小: 5000 (固定)")
    print(f"  - 测试文本数: {len(test_texts)}")
    
    from tools.tokenizer_utils import create_test_tokenizer
    
    results = []
    
    print(f"\n{'最小频率':<12} {'实际词表':<12} {'Token数':<12} {'压缩率':<12} {'未知词率'}")
    print("-"*70)
    
    for min_freq in min_frequencies:
        # 训练分词器
        tokenizer = create_test_tokenizer(
            vocab_size=5000,
            texts=test_texts * 20,
            min_frequency=min_freq,
            show_progress=False
        )
        
        # 测试编码
        total_chars = 0
        total_tokens = 0
        unk_count = 0
        
        unk_token = tokenizer.unk_token or "<unk>"
        
        for text in test_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            
            total_chars += len(text)
            total_tokens += len(ids)
            
            if unk_token in decoded:
                unk_count += 1
        
        compression = total_tokens / total_chars if total_chars > 0 else 0
        unk_rate = unk_count / len(test_texts) * 100
        
        vocab = tokenizer.get_vocab()
        actual_vocab_size = len(vocab)
        
        print(f"{min_freq:<12} {actual_vocab_size:<12} {total_tokens:<12} {compression:<12.3f} {unk_rate:>10.1f}%")
        
        results.append({
            "min_frequency": min_freq,
            "actual_vocab_size": actual_vocab_size,
            "total_tokens": total_tokens,
            "compression": compression,
            "unk_rate": unk_rate
        })
    
    # 分析
    print(f"\n分析:")
    best_coverage = min(results, key=lambda x: x['unk_rate'])
    
    print(f"  - 最佳覆盖: min_frequency={best_coverage['min_frequency']} (未知词率={best_coverage['unk_rate']:.1f}%)")
    print(f"  - 推荐配置: min_frequency=10-20 (平衡质量和覆盖)")
    
    return results


def compare_languages(tokenizer, test_cases):
    """对比不同语言的编码效率"""
    print("\n" + "="*70)
    print("  实验3: 语言编码效率对比")
    print("="*70)
    
    print(f"\n测试 {len(test_cases)} 个样本:\n")
    
    results = {
        "中文": [],
        "英文": [],
        "混合": []
    }
    
    print(f"{'语言':<10} {'文本':<40} {'字符':<8} {'Token':<8} {'压缩率'}")
    print("-"*70)
    
    for lang, text in test_cases:
        ids = tokenizer.encode(text, add_special_tokens=False)
        char_count = len(text)
        token_count = len(ids)
        compression = token_count / char_count if char_count > 0 else 0
        
        text_display = text[:38] + ".." if len(text) > 38 else text
        print(f"{lang:<10} {text_display:<40} {char_count:<8} {token_count:<8} {compression:.3f}")
        
        results[lang].append(compression)
    
    # 统计
    print("-"*70)
    for lang in ["中文", "英文", "混合"]:
        if results[lang]:
            avg = sum(results[lang]) / len(results[lang])
            print(f"{lang} 平均压缩率: {avg:.3f}")
    
    return results


def compare_tokenizer_types():
    """对比不同类型的分词器"""
    print("\n" + "="*70)
    print("  实验4: 分词器类型对比")
    print("="*70)
    
    print(f"\n对比 BPE vs WordPiece vs Unigram:\n")
    
    print("BPE (Byte Pair Encoding):")
    print("  优点: ✓ 简单高效")
    print("       ✓ 训练快速")
    print("       ✓ 适合教学")
    print("  缺点: ✗ 贪心算法")
    print("       ✗ 不考虑概率")
    
    print("\nWordPiece:")
    print("  优点: ✓ 考虑似然")
    print("       ✓ 效果稍好")
    print("  缺点: ✗ 稍复杂")
    print("       ✗ 训练稍慢")
    
    print("\nUnigram:")
    print("  优点: ✓ 概率模型")
    print("       ✓ 理论最优")
    print("  缺点: ✗ 最复杂")
    print("       ✗ 训练最慢")
    
    print("\n推荐:")
    print("  - 教学项目: BPE")
    print("  - 生产项目: BPE 或 WordPiece")
    print("  - 研究项目: Unigram")


def run_all_experiments(tokenizer_path=None):
    """运行所有对比实验"""
    print("="*70)
    print("  分词器对比实验套件")
    print("="*70)
    
    # 准备测试数据
    test_texts = [
        "人工智能在未来将会改变世界",
        "深度学习是机器学习的一个分支",
        "自然语言处理技术发展迅速",
        "大语言模型具有强大的能力",
        "Artificial intelligence is transforming the world",
        "Deep learning is a subset of machine learning",
        "Natural language processing technology is advancing",
        "Large language models have powerful capabilities",
    ]
    
    # 实验1: 词表大小对比
    compare_vocab_sizes(test_texts)
    
    # 实验2: 最小频率对比
    compare_min_frequencies(test_texts)
    
    # 实验3: 语言对比（如果提供了分词器）
    if tokenizer_path:
        try:
            from tools.tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer(tokenizer_path)
            
            test_cases = [
                ("中文", "人工智能在未来将会改变世界"),
                ("中文", "深度学习是机器学习的一个分支"),
                ("中文", "自然语言处理技术发展迅速"),
                ("英文", "Artificial intelligence is transforming the world"),
                ("英文", "Deep learning is a subset of machine learning"),
                ("英文", "Natural language processing technology"),
                ("混合", "AI人工智能 Machine Learning机器学习"),
                ("混合", "Deep Learning深度学习 NLP自然语言处理"),
            ]
            
            compare_languages(tokenizer, test_cases)
        except Exception as e:
            print(f"\n⚠️  无法加载分词器进行语言对比: {e}")
    
    # 实验4: 类型对比
    compare_tokenizer_types()
    
    # 总结
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    
    print("\n关键发现:")
    print("  1. 词表大小: 5000-10000 是最佳平衡点")
    print("  2. 最小频率: 10-20 适合大多数场景")
    print("  3. 中文压缩率: 通常 0.4-0.6")
    print("  4. 英文压缩率: 通常 0.2-0.4")
    print("  5. BPE算法: 简单高效，适合教学")
    
    print("\n推荐配置:")
    print("  - 小项目: vocab_size=2000, min_frequency=5")
    print("  - 中项目: vocab_size=6400, min_frequency=20 (MiniMind默认)")
    print("  - 大项目: vocab_size=32000, min_frequency=50")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分词器对比实验脚本")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="分词器路径（用于语言对比实验）"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "vocab", "frequency", "language", "types"],
        help="实验类型"
    )
    
    args = parser.parse_args()
    
    # 准备测试数据
    test_texts = [
        "人工智能在未来将会改变世界",
        "深度学习是机器学习的一个分支",
        "自然语言处理技术发展迅速",
        "大语言模型具有强大的能力",
        "Artificial intelligence is transforming the world",
        "Deep learning is a subset of machine learning",
    ]
    
    # 运行实验
    if args.experiment == "all":
        run_all_experiments(args.tokenizer)
    elif args.experiment == "vocab":
        compare_vocab_sizes(test_texts)
    elif args.experiment == "frequency":
        compare_min_frequencies(test_texts)
    elif args.experiment == "language":
        if args.tokenizer:
            from tools.tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer(args.tokenizer)
            test_cases = [
                ("中文", "人工智能在未来将会改变世界"),
                ("英文", "Artificial intelligence is transforming"),
                ("混合", "AI人工智能 Machine Learning"),
            ]
            compare_languages(tokenizer, test_cases)
        else:
            print("错误: 语言对比实验需要提供 --tokenizer 参数")
            return 1
    elif args.experiment == "types":
        compare_tokenizer_types()
    
    print("\n" + "="*70)
    print("  实验完成")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
