#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理测试
测试数据预处理、数据集加载功能
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pretokenize_small():
    """测试小规模数据预处理"""
    print("\n" + "="*60)
    print("测试1: 小规模数据预处理")
    print("="*60)
    
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        data_file = os.path.join(temp_dir, "test_data.jsonl")
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(tokenizer_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备测试数据
        test_data = [
            {"text": "人工智能在未来将会改变世界"},
            {"text": "深度学习是机器学习的一个重要分支"},
            {"text": "自然语言处理技术发展迅速"},
            {"text": "大语言模型具有强大的生成能力"},
        ] * 10
        
        with open(data_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ 创建测试数据: {len(test_data)} 条")
        
        # 训练分词器
        def iter_texts():
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if "text" in obj:
                        yield obj["text"]
        
        from tools.tokenizer_utils import create_test_tokenizer
        
        # 收集所有文本用于训练
        all_texts = list(iter_texts())
        tokenizer_wrapped = create_test_tokenizer(
            vocab_size=500,
            texts=all_texts,
            min_frequency=1,
            show_progress=False
        )
        tokenizer_wrapped.save_pretrained(tokenizer_dir)
        print("✓ 分词器训练完成")
        
        # 加载tokenizer对象用于后续处理
        from tokenizers import Tokenizer as TokenizerObj
        tokenizer = TokenizerObj.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        
        # 预处理数据
        eos_id = tokenizer.token_to_id("</s>")
        block_size = 32
        output_file = os.path.join(output_dir, "train.bin")
        
        all_tokens = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if "text" in obj:
                    ids = tokenizer.encode(obj["text"]).ids + [eos_id]
                    all_tokens.extend(ids)
        
        # 切分成 block
        blocks = []
        for i in range(0, len(all_tokens) - block_size, block_size):
            block = all_tokens[i:i+block_size]
            if len(block) == block_size:
                blocks.append(block)
        
        # 保存为 bin 文件
        with open(output_file, 'wb') as f:
            for block in blocks:
                f.write(np.array(block, dtype=np.uint16).tobytes())
        
        print(f"✓ 预处理完成: {len(blocks)} 个 block")
        print(f"✓ 输出文件: {output_file}")
        
        # 验证文件
        file_size = os.path.getsize(output_file)
        expected_size = len(blocks) * block_size * 2  # uint16 = 2 bytes
        assert file_size == expected_size, f"文件大小不匹配: {file_size} != {expected_size}"
        print(f"✓ 文件大小验证通过: {file_size} bytes")
        
        # 清理
        shutil.rmtree(temp_dir)
        
        print("✅ 小规模数据预处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 小规模数据预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*60)
    print("测试2: 数据集加载")
    print("="*60)
    
    try:
        from src.data.dataset import MMapDataset
        import torch
        
        # 创建临时 bin 文件
        temp_dir = tempfile.mkdtemp()
        bin_file = os.path.join(temp_dir, "test.bin")
        
        # 生成测试数据
        block_size = 32
        n_blocks = 10
        vocab_size = 500
        
        data = np.random.randint(0, vocab_size, size=n_blocks * (block_size + 1), dtype=np.uint16)
        with open(bin_file, 'wb') as f:
            f.write(data.tobytes())
        
        print(f"✓ 创建测试数据: {n_blocks} 个 block")
        
        # 加载数据集
        dataset = MMapDataset([bin_file], block_size=block_size)
        print(f"✓ 数据集加载成功: {len(dataset)} 个样本")
        
        # 验证数据集长度
        assert len(dataset) == n_blocks, f"数据集长度不匹配: {len(dataset)} != {n_blocks}"
        print(f"✓ 数据集长度验证通过")
        
        # 测试数据读取
        for i in range(min(3, len(dataset))):
            x, y = dataset[i]
            assert x.shape == (block_size,), f"输入形状错误: {x.shape}"
            assert y.shape == (block_size,), f"目标形状错误: {y.shape}"
            # y 应该是 x 的右移版本（下一个 token）
            # 验证 y[0] 应该等于 x[1]（如果有的话）
            if block_size > 1:
                assert y[0] == x[1], "目标不是输入的右移"
            print(f"✓ 样本 {i}: x.shape={x.shape}, y.shape={y.shape}")
        
        # 清理
        dataset.close()
        shutil.rmtree(temp_dir)
        
        print("✅ 数据集加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_integrity():
    """测试数据完整性"""
    print("\n" + "="*60)
    print("测试3: 数据完整性")
    print("="*60)
    
    try:
        from src.data.dataset import MMapDataset
        
        # 创建临时 bin 文件
        temp_dir = tempfile.mkdtemp()
        bin_file = os.path.join(temp_dir, "test.bin")
        
        # 生成测试数据（确保在词表范围内）
        block_size = 32
        n_blocks = 5
        vocab_size = 100
        
        data = np.random.randint(0, vocab_size, size=n_blocks * (block_size + 1), dtype=np.uint16)
        with open(bin_file, 'wb') as f:
            f.write(data.tobytes())
        
        print(f"✓ 创建测试数据: vocab_size={vocab_size}")
        
        # 加载数据集
        dataset = MMapDataset([bin_file], block_size=block_size)
        
        # 检查所有 token 是否在词表范围内
        max_id = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            max_id = max(max_id, x.max().item(), y.max().item())
        
        assert max_id < vocab_size, f"发现超出词表的 token: {max_id} >= {vocab_size}"
        print(f"✓ 最大 token ID: {max_id} < {vocab_size}")
        
        # 检查数据不为空
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.numel() > 0, f"样本 {i} 的输入为空"
            assert y.numel() > 0, f"样本 {i} 的目标为空"
        
        print(f"✓ 所有样本非空")
        
        # 清理
        dataset.close()
        shutil.rmtree(temp_dir)
        
        print("✅ 数据完整性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据完整性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_bin_files():
    """测试多个 bin 文件加载"""
    print("\n" + "="*60)
    print("测试4: 多个 bin 文件加载")
    print("="*60)
    
    try:
        from src.data.dataset import MMapDataset
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建多个 bin 文件
        block_size = 32
        n_files = 3
        n_blocks_per_file = 5
        vocab_size = 100
        
        bin_files = []
        for i in range(n_files):
            bin_file = os.path.join(temp_dir, f"test_{i}.bin")
            data = np.random.randint(0, vocab_size, size=n_blocks_per_file * (block_size + 1), dtype=np.uint16)
            with open(bin_file, 'wb') as f:
                f.write(data.tobytes())
            bin_files.append(bin_file)
        
        print(f"✓ 创建 {n_files} 个 bin 文件")
        
        # 加载数据集
        dataset = MMapDataset(bin_files, block_size=block_size)
        
        # 验证总长度
        expected_len = n_files * n_blocks_per_file
        assert len(dataset) == expected_len, f"数据集长度不匹配: {len(dataset)} != {expected_len}"
        print(f"✓ 数据集总长度: {len(dataset)}")
        
        # 测试随机访问
        for i in [0, len(dataset) // 2, len(dataset) - 1]:
            x, y = dataset[i]
            assert x.shape == (block_size,), f"样本 {i} 形状错误"
            print(f"✓ 样本 {i} 访问成功")
        
        # 清理
        dataset.close()
        shutil.rmtree(temp_dir)
        
        print("✅ 多个 bin 文件加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 多个 bin 文件加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("数据预处理测试套件")
    print("="*60)
    
    tests = [
        ("小规模数据预处理", test_pretokenize_small),
        ("数据集加载", test_dataset_loading),
        ("数据完整性", test_data_integrity),
        ("多个 bin 文件加载", test_multiple_bin_files),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"成功率: {passed / len(tests) * 100:.1f}%")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
