#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE 分词器训练脚本
使用 pretrain_hq.jsonl 数据训练分词器

支持命令行参数自定义配置
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
import os
import json
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "0"


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练BPE分词器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python src/training/train_tokenizer.py
  
  # 自定义词表大小
  python src/training/train_tokenizer.py --vocab_size 10000
  
  # 自定义所有参数
  python src/training/train_tokenizer.py \\
    --input_file data/my_data.jsonl \\
    --output_dir my_tokenizer \\
    --vocab_size 8000 \\
    --min_frequency 10
        """
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="minimind_dataset/pretrain_hq.jsonl",
        help="输入数据文件路径 (JSONL格式，每行包含'text'字段)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my_tokenizer",
        help="分词器输出目录"
    )
    
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=6400,
        help="词表大小 (推荐: 1000-50000，默认6400)"
    )
    
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=20,
        help="最小词频 (低于此频率的token会被过滤，默认20)"
    )
    
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<pad>", "<unk>", "<s>", "</s>"],
        help="特殊token列表 (默认: <pad> <unk> <s> </s>)"
    )
    
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="显示训练进度条"
    )
    
    parser.add_argument(
        "--test_text",
        type=str,
        default="人工智能在未来将会改变世界",
        help="用于验证的测试文本"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = get_args()
    
    print("=" * 60)
    print("BPE 分词器训练")
    print("=" * 60)
    print(f"配置:")
    print(f"  输入文件: {args.input_file}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  词表大小: {args.vocab_size}")
    print(f"  最小频率: {args.min_frequency}")
    print(f"  特殊token: {', '.join(args.special_tokens)}")
    print("=" * 60)
    
    # 1. 检查数据文件
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(
            f"数据文件不存在: {args.input_file}\n"
            f"请确保文件路径正确，或从以下位置下载:\n"
            f"  - ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind-dataset/files\n"
            f"  - HuggingFace: https://huggingface.co/datasets/jingyaogong\n"
            f"并放置到正确的目录"
        )
    
    print(f"\n✓ 数据文件: {args.input_file}")
    
    # 2. 统计数据
    print("正在统计数据...")
    total_lines = 0
    valid_lines = 0
    total_chars = 0
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj and obj["text"]:
                    valid_lines += 1
                    total_chars += len(obj["text"])
            except json.JSONDecodeError:
                continue
    
    print(f"✓ 总行数: {total_lines:,}")
    print(f"✓ 有效行数: {valid_lines:,}")
    print(f"✓ 总字符数: {total_chars:,}")
    print(f"✓ 平均每行字符数: {total_chars // valid_lines if valid_lines > 0 else 0}")
    
    if valid_lines == 0:
        raise ValueError("没有找到有效的文本数据")
    
    # 3. 定义文本迭代器
    def iter_text():
        """从数据文件读取文本"""
        count = 0
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj and obj["text"]:
                        count += 1
                        yield obj["text"]
                except json.JSONDecodeError as e:
                    if line_num <= 10:  # 只显示前10个错误
                        print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
        print(f"\n✓ 成功读取 {count:,} 条文本")
    
    # 4. 构造并训练 BPE
    print("\n正在初始化 BPE 分词器...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        min_frequency=args.min_frequency,
        show_progress=args.show_progress,
    )
    
    print(f"✓ 词表大小: {args.vocab_size}")
    print(f"✓ 最小频率: {args.min_frequency}")
    print(f"✓ 特殊 token: {', '.join(args.special_tokens)}")
    
    print("\n开始训练分词器...")
    if not args.show_progress:
        print("(这可能需要几分钟，请耐心等待...)")
    
    tokenizer.train_from_iterator(iter_text(), trainer)
    print("✓ 分词器训练完成")
    
    # 5. 导出 HuggingFace 标准目录
    print(f"\n正在保存到 {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
    )
    wrapped_tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ 分词器已保存到: {args.output_dir}/")
    
    # 6. 验证分词器
    print("\n验证分词器...")
    test_text = args.test_text
    tokens = wrapped_tokenizer.encode(test_text, add_special_tokens=False)
    decoded = wrapped_tokenizer.decode(tokens, skip_special_tokens=True)
    
    print(f"测试文本: {test_text}")
    print(f"Token 数量: {len(tokens)}")
    print(f"Token IDs: {tokens[:20]}..." if len(tokens) > 20 else f"Token IDs: {tokens}")
    print(f"解码结果: {decoded}")
    print(f"一致性检查: {'✓ 通过' if test_text == decoded else '✗ 失败'}")
    
    # 7. 统计信息
    vocab = tokenizer.get_vocab()
    print(f"\n✓ 实际词表大小: {len(vocab)}")
    print(f"✓ 特殊 token ID:")
    for token in args.special_tokens:
        token_id = wrapped_tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: {token_id}")
    
    # 8. 性能分析
    print(f"\n性能分析:")
    compression_ratio = len(tokens) / len(test_text)
    print(f"  压缩率: {compression_ratio:.2f} (token数/字符数)")
    print(f"  词表利用率: {len(vocab) / args.vocab_size * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("分词器训练完成！")
    print("=" * 60)
    print(f"数据来源: {args.input_file}")
    print(f"保存目录: {args.output_dir}/")
    print(f"词表大小: {len(vocab)}")
    print(f"有效文本: {valid_lines:,} 条")
    print("\n使用方法:")
    print(f"  from transformers import PreTrainedTokenizerFast")
    print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{args.output_dir}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
