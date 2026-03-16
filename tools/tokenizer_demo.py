#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器交互式演示工具
提供实时编码解码、可视化token分割等功能
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.tokenizer_utils import load_tokenizer, create_test_tokenizer, print_tokenizer_info
import argparse


def interactive_encode_decode(tokenizer):
    """交互式编码解码演示"""
    print("\n" + "="*60)
    print("交互式编码解码演示")
    print("="*60)
    print("输入文本进行编码解码测试（输入 'quit' 退出）")
    print("-"*60)
    
    while True:
        try:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("退出演示")
                break
            
            if not text:
                continue
            
            # 编码
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids_with_special = tokenizer.encode(text, add_special_tokens=True)
            
            # 解码
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_with_special = tokenizer.decode(ids_with_special, skip_special_tokens=False)
            
            # 显示结果
            print(f"\n原文: {text}")
            print(f"Token数量: {len(ids)}")
            print(f"Token IDs: {ids}")
            
            # 显示每个token
            print(f"\nToken分割:")
            for i, token_id in enumerate(ids):
                token_text = tokenizer.decode([token_id])
                print(f"  [{i}] ID={token_id:4d} → '{token_text}'")
            
            print(f"\n解码结果: {decoded}")
            print(f"匹配: {'✓ 完全匹配' if text == decoded else '✗ 不匹配'}")
            
            if len(ids_with_special) > len(ids):
                print(f"\n带特殊token: {decoded_with_special}")
                print(f"特殊token数量: {len(ids_with_special) - len(ids)}")
            
        except KeyboardInterrupt:
            print("\n\n退出演示")
            break
        except Exception as e:
            print(f"错误: {e}")


def batch_test(tokenizer, test_texts=None):
    """批量测试演示"""
    print("\n" + "="*60)
    print("批量测试演示")
    print("="*60)
    
    if test_texts is None:
        test_texts = [
            "人工智能",
            "深度学习模型",
            "自然语言处理技术",
            "Hello World",
            "Artificial Intelligence",
            "你好世界",
            "机器学习算法",
            "神经网络训练",
        ]
    
    print(f"测试 {len(test_texts)} 个样本:\n")
    
    total_chars = 0
    total_tokens = 0
    
    for i, text in enumerate(test_texts, 1):
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        match = "✓" if text == decoded else "✗"
        
        total_chars += len(text)
        total_tokens += len(ids)
        
        print(f"{i}. '{text}'")
        print(f"   字符数: {len(text)}, Token数: {len(ids)}, 压缩率: {len(ids)/len(text):.2f}, 匹配: {match}")
    
    print(f"\n统计:")
    print(f"  总字符数: {total_chars}")
    print(f"  总Token数: {total_tokens}")
    print(f"  平均压缩率: {total_tokens/total_chars:.2f}")


def visualize_tokenization(tokenizer, text):
    """可视化token分割"""
    print("\n" + "="*60)
    print("Token分割可视化")
    print("="*60)
    
    ids = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\n原文: {text}")
    print(f"Token数量: {len(ids)}\n")
    
    # 显示分割
    print("分割结果:")
    tokens = []
    for token_id in ids:
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)
    
    # 用 | 分隔显示
    print("  " + " | ".join(tokens))
    
    # 显示详细信息
    print("\n详细信息:")
    for i, (token_id, token_text) in enumerate(zip(ids, tokens)):
        print(f"  {i+1:2d}. ID={token_id:5d}  '{token_text}'")


def compare_vocab_sizes(text, vocab_sizes=[500, 1000, 2000, 6400]):
    """对比不同词表大小的效果"""
    print("\n" + "="*60)
    print("词表大小对比")
    print("="*60)
    
    print(f"\n测试文本: {text}")
    print(f"文本长度: {len(text)} 字符\n")
    
    print(f"{'词表大小':<10} {'Token数':<10} {'压缩率':<10} {'训练时间'}")
    print("-"*60)
    
    for vocab_size in vocab_sizes:
        try:
            # 创建测试分词器
            test_texts = [text] * 50  # 重复以提供足够训练数据
            tokenizer = create_test_tokenizer(
                vocab_size=vocab_size,
                texts=test_texts,
                show_progress=False
            )
            
            # 编码
            ids = tokenizer.encode(text, add_special_tokens=False)
            compression = len(ids) / len(text)
            
            print(f"{vocab_size:<10} {len(ids):<10} {compression:<10.2f} {'快' if vocab_size < 2000 else '中' if vocab_size < 5000 else '慢'}")
            
        except Exception as e:
            print(f"{vocab_size:<10} 错误: {e}")


def demo_special_tokens(tokenizer):
    """演示特殊token的使用"""
    print("\n" + "="*60)
    print("特殊Token演示")
    print("="*60)
    
    text = "测试文本"
    
    # 不带特殊token
    ids_no_special = tokenizer.encode(text, add_special_tokens=False)
    decoded_no_special = tokenizer.decode(ids_no_special, skip_special_tokens=True)
    
    # 带特殊token
    ids_with_special = tokenizer.encode(text, add_special_tokens=True)
    decoded_with_special = tokenizer.decode(ids_with_special, skip_special_tokens=False)
    
    print(f"\n原文: {text}")
    print(f"\n不带特殊token:")
    print(f"  IDs: {ids_no_special}")
    print(f"  解码: '{decoded_no_special}'")
    
    print(f"\n带特殊token:")
    print(f"  IDs: {ids_with_special}")
    print(f"  解码: '{decoded_with_special}'")
    
    print(f"\n特殊token信息:")
    print(f"  <pad> (padding): ID={tokenizer.pad_token_id}")
    print(f"  <unk> (unknown): ID={tokenizer.unk_token_id}")
    print(f"  <s> (begin): ID={tokenizer.bos_token_id}")
    print(f"  </s> (end): ID={tokenizer.eos_token_id}")


def main():
    parser = argparse.ArgumentParser(description="分词器交互式演示工具")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="分词器路径（默认创建测试分词器）")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "visualize", "compare", "special"],
                       help="演示模式")
    parser.add_argument("--text", type=str, default="人工智能在未来将会改变世界",
                       help="测试文本（用于visualize和compare模式）")
    
    args = parser.parse_args()
    
    print("="*60)
    print("分词器交互式演示工具")
    print("="*60)
    
    # 加载或创建分词器
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"\n加载分词器: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        print("\n创建测试分词器...")
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的能力",
            "Artificial intelligence is transforming the world",
            "Deep learning is a subset of machine learning",
        ] * 100
        tokenizer = create_test_tokenizer(vocab_size=1000, texts=test_texts, show_progress=True)
        print("✓ 测试分词器创建完成")
    
    # 显示分词器信息
    print("\n分词器信息:")
    print_tokenizer_info(tokenizer, verbose=False)
    
    # 根据模式运行演示
    if args.mode == "interactive":
        interactive_encode_decode(tokenizer)
    elif args.mode == "batch":
        batch_test(tokenizer)
    elif args.mode == "visualize":
        visualize_tokenization(tokenizer, args.text)
    elif args.mode == "compare":
        compare_vocab_sizes(args.text)
    elif args.mode == "special":
        demo_special_tokens(tokenizer)
    
    print("\n" + "="*60)
    print("演示结束")
    print("="*60)


if __name__ == "__main__":
    main()
