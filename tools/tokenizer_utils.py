#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器工具模块
提供统一的分词器创建、加载、分析接口，消除代码重复
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
import os


def create_test_tokenizer(vocab_size=1000, texts=None, min_frequency=2, show_progress=False):
    """
    创建测试用的BPE分词器（统一接口）
    
    参数:
        vocab_size: 词表大小，默认1000
        texts: 训练文本列表，默认使用简单测试文本
        min_frequency: 最小词频，默认2
        show_progress: 是否显示训练进度，默认False
    
    返回:
        PreTrainedTokenizerFast: 训练好的分词器
    
    示例:
        >>> tokenizer = create_test_tokenizer(vocab_size=500)
        >>> ids = tokenizer.encode("测试文本")
        >>> text = tokenizer.decode(ids)
    """
    # 创建BPE分词器
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    
    # 准备训练数据
    if texts is None:
        texts = [
            "hello world",
            "test data",
            "sample text",
            "人工智能",
            "机器学习",
            "深度学习",
        ]
    
    # 训练分词器
    tokenizer.train_from_iterator(texts, trainer)
    
    # 包装为HuggingFace格式
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    return wrapped_tokenizer


def create_simple_tokenizer(vocab_size=1000, texts=None):
    """
    创建简单的BPE分词器（使用Whitespace预处理）
    适用于简单的测试场景
    
    参数:
        vocab_size: 词表大小
        texts: 训练文本列表
    
    返回:
        PreTrainedTokenizerFast: 训练好的分词器
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
    )
    
    if texts is None:
        texts = ["hello world", "test data"]
    
    tokenizer.train_from_iterator(texts, trainer)
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    return wrapped_tokenizer


def load_tokenizer(tokenizer_path):
    """
    加载已训练的分词器（统一接口）
    
    参数:
        tokenizer_path: 分词器目录路径
    
    返回:
        PreTrainedTokenizerFast: 加载的分词器
    
    示例:
        >>> tokenizer = load_tokenizer("my_tokenizer")
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"分词器路径不存在: {tokenizer_path}")
    
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def analyze_tokenizer(tokenizer, test_texts=None):
    """
    分析分词器性能
    
    参数:
        tokenizer: 分词器对象
        test_texts: 测试文本列表
    
    返回:
        dict: 分析结果
    """
    if test_texts is None:
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个分支",
            "Artificial intelligence is transforming the world",
        ]
    
    results = {
        "vocab_size": len(tokenizer.get_vocab()),
        "special_tokens": {
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
        },
        "test_results": []
    }
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        results["test_results"].append({
            "text": text,
            "token_count": len(ids),
            "decoded": decoded,
            "match": text == decoded
        })
    
    return results


def print_tokenizer_info(tokenizer, verbose=True):
    """
    打印分词器信息
    
    参数:
        tokenizer: 分词器对象
        verbose: 是否显示详细信息
    """
    vocab_size = len(tokenizer.get_vocab())
    
    print(f"词表大小: {vocab_size}")
    print(f"特殊 token ID:")
    print(f"  <pad>: {tokenizer.pad_token_id}")
    print(f"  <unk>: {tokenizer.unk_token_id}")
    print(f"  <s>: {tokenizer.bos_token_id}")
    print(f"  </s>: {tokenizer.eos_token_id}")
    
    if verbose:
        # 测试编码解码
        test_text = "人工智能测试"
        ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(ids)
        print(f"\n测试编码解码:")
        print(f"  原文: {test_text}")
        print(f"  Token数: {len(ids)}")
        print(f"  解码: {decoded}")
        print(f"  匹配: {'✓' if test_text == decoded else '✗'}")


def compare_tokenizers(tokenizers_dict, test_texts=None):
    """
    对比多个分词器的性能
    
    参数:
        tokenizers_dict: 分词器字典 {名称: 分词器对象}
        test_texts: 测试文本列表
    
    返回:
        dict: 对比结果
    """
    if test_texts is None:
        test_texts = [
            "人工智能",
            "深度学习模型",
            "Hello World",
        ]
    
    results = {}
    
    for name, tokenizer in tokenizers_dict.items():
        results[name] = {
            "vocab_size": len(tokenizer.get_vocab()),
            "encodings": []
        }
        
        for text in test_texts:
            ids = tokenizer.encode(text)
            results[name]["encodings"].append({
                "text": text,
                "token_count": len(ids),
                "tokens": ids[:10]  # 只显示前10个
            })
    
    return results
