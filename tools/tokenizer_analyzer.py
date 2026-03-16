#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器可视化分析工具
提供词表分布、编码效率、语言分析等可视化功能
"""
import sys
import os
from pathlib import Path
from collections import Counter
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_vocab_distribution(tokenizer, top_n=20):
    """分析词表分布"""
    print("\n" + "="*70)
    print("  词表分布分析")
    print("="*70)
    
    vocab = tokenizer.get_vocab()
    
    # 按ID排序
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    print(f"\n词表大小: {len(vocab)}")
    print(f"\n前{top_n}个token (按ID):")
    print(f"{'ID':<8} {'Token':<30} {'长度'}")
    print("-"*70)
    
    for token, token_id in sorted_vocab[:top_n]:
        token_display = repr(token)[:28]
        print(f"{token_id:<8} {token_display:<30} {len(token)}")
    
    # 统计token长度分布
    length_dist = Counter(len(token) for token in vocab.keys())
    
    print(f"\nToken长度分布:")
    print(f"{'长度':<10} {'数量':<10} {'占比':<10} {'可视化'}")
    print("-"*70)
    
    max_count = max(length_dist.values())
    for length in sorted(length_dist.keys())[:15]:
        count = length_dist[length]
        percentage = count / len(vocab) * 100
        bar_length = int(count / max_count * 40)
        bar = "█" * bar_length
        print(f"{length:<10} {count:<10} {percentage:>6.2f}%   {bar}")
    
    return {
        "vocab_size": len(vocab),
        "length_distribution": dict(length_dist),
        "avg_token_length": sum(len(t) * c for t, c in length_dist.items()) / len(vocab)
    }


def analyze_encoding_efficiency(tokenizer, test_texts=None):
    """分析编码效率"""
    print("\n" + "="*70)
    print("  编码效率分析")
    print("="*70)
    
    if test_texts is None:
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的能力",
            "Artificial intelligence is transforming the world",
            "Deep learning is a subset of machine learning",
            "Natural language processing technology is advancing rapidly",
            "Large language models have powerful capabilities",
        ]
    
    print(f"\n测试 {len(test_texts)} 个样本:\n")
    
    results = []
    total_chars = 0
    total_tokens = 0
    
    print(f"{'文本':<50} {'字符':<8} {'Token':<8} {'压缩率'}")
    print("-"*70)
    
    for text in test_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        char_count = len(text)
        token_count = len(ids)
        compression = token_count / char_count if char_count > 0 else 0
        
        total_chars += char_count
        total_tokens += token_count
        
        text_display = text[:48] + ".." if len(text) > 48 else text
        print(f"{text_display:<50} {char_count:<8} {token_count:<8} {compression:.3f}")
        
        results.append({
            "text": text,
            "chars": char_count,
            "tokens": token_count,
            "compression": compression
        })
    
    avg_compression = total_tokens / total_chars if total_chars > 0 else 0
    
    print("-"*70)
    print(f"{'总计/平均':<50} {total_chars:<8} {total_tokens:<8} {avg_compression:.3f}")
    
    # 分析中英文差异
    chinese_results = [r for r in results if any('\u4e00' <= c <= '\u9fff' for c in r['text'])]
    english_results = [r for r in results if not any('\u4e00' <= c <= '\u9fff' for c in r['text'])]
    
    if chinese_results:
        avg_cn = sum(r['compression'] for r in chinese_results) / len(chinese_results)
        print(f"\n中文平均压缩率: {avg_cn:.3f}")
    
    if english_results:
        avg_en = sum(r['compression'] for r in english_results) / len(english_results)
        print(f"英文平均压缩率: {avg_en:.3f}")
    
    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_compression": avg_compression,
        "results": results
    }


def analyze_special_tokens(tokenizer):
    """分析特殊token"""
    print("\n" + "="*70)
    print("  特殊Token分析")
    print("="*70)
    
    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
    }
    
    print(f"\n{'Token类型':<20} {'Token':<15} {'ID':<10} {'状态'}")
    print("-"*70)
    
    for name, token in special_tokens.items():
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token)
            status = "✓ 已配置"
        else:
            token_id = "N/A"
            status = "✗ 未配置"
        
        print(f"{name:<20} {str(token):<15} {str(token_id):<10} {status}")
    
    return special_tokens


def analyze_language_coverage(tokenizer, test_texts=None):
    """分析语言覆盖率"""
    print("\n" + "="*70)
    print("  语言覆盖率分析")
    print("="*70)
    
    if test_texts is None:
        test_texts = [
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言处理",
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Natural Language Processing",
            "你好世界",
            "Hello World",
        ]
    
    print(f"\n测试 {len(test_texts)} 个样本:\n")
    
    unk_token = tokenizer.unk_token or "<unk>"
    unk_count = 0
    total_count = len(test_texts)
    
    print(f"{'文本':<40} {'Token数':<10} {'未知词'}")
    print("-"*70)
    
    for text in test_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        has_unk = unk_token in tokenizer.decode(ids, skip_special_tokens=False)
        if has_unk:
            unk_count += 1
        
        text_display = text[:38] + ".." if len(text) > 38 else text
        unk_status = "✗ 有" if has_unk else "✓ 无"
        
        print(f"{text_display:<40} {len(ids):<10} {unk_status}")
    
    coverage_rate = (total_count - unk_count) / total_count * 100 if total_count > 0 else 0
    
    print("-"*70)
    print(f"覆盖率: {coverage_rate:.1f}% ({total_count - unk_count}/{total_count})")
    print(f"未知词率: {unk_count / total_count * 100:.1f}% ({unk_count}/{total_count})")
    
    return {
        "coverage_rate": coverage_rate,
        "unknown_rate": unk_count / total_count * 100 if total_count > 0 else 0,
        "total_samples": total_count,
        "unknown_samples": unk_count
    }


def analyze_token_frequency(tokenizer, sample_texts=None):
    """分析token使用频率"""
    print("\n" + "="*70)
    print("  Token使用频率分析")
    print("="*70)
    
    if sample_texts is None:
        sample_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个分支",
            "自然语言处理技术发展迅速",
        ] * 10
    
    # 统计token使用频率
    token_counter = Counter()
    
    for text in sample_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counter.update(ids)
    
    total_tokens = sum(token_counter.values())
    vocab_size = len(tokenizer.get_vocab())
    used_tokens = len(token_counter)
    
    print(f"\n样本统计:")
    print(f"  总Token数: {total_tokens:,}")
    print(f"  使用的不同Token数: {used_tokens:,}")
    print(f"  词表大小: {vocab_size:,}")
    print(f"  词表利用率: {used_tokens / vocab_size * 100:.1f}%")
    
    # 显示最常用的token
    print(f"\n最常用的20个Token:")
    print(f"{'排名':<6} {'Token ID':<10} {'Token':<30} {'频率':<10} {'占比'}")
    print("-"*70)
    
    for rank, (token_id, count) in enumerate(token_counter.most_common(20), 1):
        token_text = tokenizer.decode([token_id])
        token_display = repr(token_text)[:28]
        percentage = count / total_tokens * 100
        print(f"{rank:<6} {token_id:<10} {token_display:<30} {count:<10} {percentage:>5.2f}%")
    
    return {
        "total_tokens": total_tokens,
        "used_tokens": used_tokens,
        "vocab_size": vocab_size,
        "utilization_rate": used_tokens / vocab_size * 100 if vocab_size > 0 else 0
    }


def generate_report(tokenizer, output_file=None):
    """生成完整分析报告"""
    print("\n" + "="*70)
    print("  分词器完整分析报告")
    print("="*70)
    
    report = {
        "vocab_distribution": analyze_vocab_distribution(tokenizer),
        "encoding_efficiency": analyze_encoding_efficiency(tokenizer),
        "special_tokens": analyze_special_tokens(tokenizer),
        "language_coverage": analyze_language_coverage(tokenizer),
        "token_frequency": analyze_token_frequency(tokenizer),
    }
    
    # 总结
    print("\n" + "="*70)
    print("  分析总结")
    print("="*70)
    
    print(f"\n词表信息:")
    print(f"  - 词表大小: {report['vocab_distribution']['vocab_size']:,}")
    print(f"  - 平均Token长度: {report['vocab_distribution']['avg_token_length']:.2f}")
    
    print(f"\n编码效率:")
    print(f"  - 平均压缩率: {report['encoding_efficiency']['avg_compression']:.3f}")
    
    print(f"\n语言覆盖:")
    print(f"  - 覆盖率: {report['language_coverage']['coverage_rate']:.1f}%")
    print(f"  - 未知词率: {report['language_coverage']['unknown_rate']:.1f}%")
    
    print(f"\n词表利用:")
    print(f"  - 利用率: {report['token_frequency']['utilization_rate']:.1f}%")
    
    # 评分
    score = 0
    max_score = 5
    
    # 评分标准
    if report['vocab_distribution']['vocab_size'] >= 5000:
        score += 1
    if 0.2 <= report['encoding_efficiency']['avg_compression'] <= 0.7:
        score += 1
    if report['language_coverage']['coverage_rate'] >= 95:
        score += 1
    if report['language_coverage']['unknown_rate'] <= 5:
        score += 1
    if report['token_frequency']['utilization_rate'] >= 50:
        score += 1
    
    print(f"\n总体评分: {score}/{max_score} = {score/max_score*100:.0f}%")
    
    if score >= 4:
        print("评价: ✅ 优秀")
    elif score >= 3:
        print("评价: ✓ 良好")
    else:
        print("评价: ⚠️  需要改进")
    
    # 保存报告
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存到: {output_file}")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分词器可视化分析工具")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="分词器路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "vocab", "efficiency", "special", "coverage", "frequency"],
        help="分析模式"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径（JSON格式）"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  分词器可视化分析工具")
    print("="*70)
    
    # 加载分词器
    try:
        from tools.tokenizer_utils import load_tokenizer
        print(f"\n加载分词器: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
        print("✓ 分词器加载成功")
    except Exception as e:
        print(f"✗ 分词器加载失败: {e}")
        return 1
    
    # 执行分析
    if args.mode == "all":
        generate_report(tokenizer, args.output)
    elif args.mode == "vocab":
        analyze_vocab_distribution(tokenizer)
    elif args.mode == "efficiency":
        analyze_encoding_efficiency(tokenizer)
    elif args.mode == "special":
        analyze_special_tokens(tokenizer)
    elif args.mode == "coverage":
        analyze_language_coverage(tokenizer)
    elif args.mode == "frequency":
        analyze_token_frequency(tokenizer)
    
    print("\n" + "="*70)
    print("  分析完成")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
