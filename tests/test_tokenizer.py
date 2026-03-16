#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器测试
测试分词器训练、编码、解码功能
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入工具函数
from tools.tokenizer_utils import create_test_tokenizer

def test_tokenizer_training():
    """测试分词器训练"""
    print("\n" + "="*60)
    print("测试1: 分词器训练")
    print("="*60)
    
    try:
        # 准备测试数据
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的一个分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的能力",
            "Artificial intelligence is transforming the world",
            "Deep learning is a subset of machine learning",
        ] * 100  # 重复以增加训练数据
        
        print(f"✓ 准备了 {len(test_texts)} 条测试文本")
        
        # 使用工具函数创建分词器
        tokenizer = create_test_tokenizer(
            vocab_size=1000,
            texts=test_texts,
            min_frequency=2,
            show_progress=False
        )
        print("✓ 分词器训练完成")
        
        # 验证词表大小
        vocab = tokenizer.get_vocab()
        assert len(vocab) <= 1000, f"词表大小超出预期: {len(vocab)}"
        print(f"✓ 词表大小: {len(vocab)}")
        
        # 验证特殊 token
        for token in ["<pad>", "<unk>", "<s>", "</s>"]:
            assert token in vocab, f"缺少特殊 token: {token}"
        print("✓ 特殊 token 验证通过")
        
        # 测试保存和加载
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "test_tokenizer")
        tokenizer.save_pretrained(output_dir)
        print(f"✓ 分词器已保存到: {output_dir}")
        
        # 清理
        shutil.rmtree(temp_dir)
        
        print("✅ 分词器训练测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 分词器训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_encode_decode():
    """测试编码和解码"""
    print("\n" + "="*60)
    print("测试2: 编码和解码")
    print("="*60)
    
    try:
        # 训练一个简单的分词器
        test_texts = [
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言处理",
        ] * 50
        
        tokenizer = create_test_tokenizer(
            vocab_size=500,
            texts=test_texts,
            min_frequency=1,
            show_progress=False
        )
        print("✓ 测试分词器训练完成")
        
        # 测试中文编码
        test_cases = [
            "人工智能在未来",
            "深度学习模型",
            "自然语言处理技术",
            "Hello World",
            "你好世界",
        ]
        
        for text in test_cases:
            # 编码
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            # 解码
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # 验证
            assert decoded == text, f"解码不一致: '{text}' -> '{decoded}'"
            print(f"✓ '{text}' -> {len(token_ids)} tokens -> '{decoded}'")
        
        print("✅ 编码解码测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 编码解码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_special_tokens():
    """测试特殊 token 处理"""
    print("\n" + "="*60)
    print("测试3: 特殊 token 处理")
    print("="*60)
    
    try:
        # 训练分词器
        test_texts = ["测试文本"] * 100
        
        tokenizer = create_test_tokenizer(
            vocab_size=500,
            texts=test_texts,
            min_frequency=1,
            show_progress=False
        )
        
        # 测试特殊 token ID
        pad_id = tokenizer.pad_token_id
        unk_id = tokenizer.unk_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        
        assert pad_id is not None, "pad_token_id 为 None"
        assert unk_id is not None, "unk_token_id 为 None"
        assert bos_id is not None, "bos_token_id 为 None"
        assert eos_id is not None, "eos_token_id 为 None"
        
        print(f"✓ pad_token_id: {pad_id}")
        print(f"✓ unk_token_id: {unk_id}")
        print(f"✓ bos_token_id: {bos_id}")
        print(f"✓ eos_token_id: {eos_id}")
        
        # 测试 skip_special_tokens
        text = "测试文本"
        ids = tokenizer.encode(text, add_special_tokens=True)
        decoded_with = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_without = tokenizer.decode(ids, skip_special_tokens=True)
        
        print(f"✓ 带特殊 token: '{decoded_with}'")
        print(f"✓ 不带特殊 token: '{decoded_without}'")
        
        print("✅ 特殊 token 测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 特殊 token 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_vocab_size():
    """测试词表大小限制"""
    print("\n" + "="*60)
    print("测试4: 词表大小限制")
    print("="*60)
    
    try:
        test_texts = ["测试文本 " + str(i) for i in range(1000)]
        
        for vocab_size in [100, 500, 1000]:
            tokenizer = create_test_tokenizer(
                vocab_size=vocab_size,
                texts=test_texts,
                min_frequency=1,
                show_progress=False
            )
            
            actual_size = len(tokenizer.get_vocab())
            assert actual_size <= vocab_size, f"词表大小超出限制: {actual_size} > {vocab_size}"
            print(f"✓ 目标词表: {vocab_size}, 实际词表: {actual_size}")
        
        print("✅ 词表大小限制测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 词表大小限制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("分词器测试套件")
    print("="*60)
    
    tests = [
        ("分词器训练", test_tokenizer_training),
        ("编码和解码", test_tokenizer_encode_decode),
        ("特殊 token 处理", test_tokenizer_special_tokens),
        ("词表大小限制", test_tokenizer_vocab_size),
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
