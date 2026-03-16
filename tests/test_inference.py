#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理测试
测试模型推理、文本生成功能
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil
import torch
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_inference():
    """测试模型推理"""
    print("\n" + "="*60)
    print("测试1: 模型推理")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 准备输入
        prompt = torch.randint(0, vocab_size, (1, 10))
        
        # 推理
        with torch.no_grad():
            logits, loss, caches = model(prompt)
        
        # 验证输出
        assert logits.shape == (1, 10, vocab_size), f"输出形状错误: {logits.shape}"
        print(f"✓ 输入形状: {prompt.shape}")
        print(f"✓ 输出形状: {logits.shape}")
        
        # 获取预测
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        assert 0 <= next_token.item() < vocab_size, f"预测 token 超出范围: {next_token.item()}"
        print(f"✓ 预测下一个 token: {next_token.item()}")
        
        print("✅ 模型推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_generation():
    """测试文本生成"""
    print("\n" + "="*60)
    print("测试2: 文本生成")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        max_new_tokens = 20
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 准备输入
        prompt = torch.randint(0, vocab_size, (1, 5))
        
        # 生成文本
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=None,
        )
        
        # 验证生成结果
        assert generated.shape[0] == 1, f"batch size 错误: {generated.shape[0]}"
        assert generated.shape[1] == 5 + max_new_tokens, f"生成长度错误: {generated.shape[1]}"
        
        print(f"✓ 输入长度: {prompt.shape[1]}")
        print(f"✓ 生成长度: {generated.shape[1]}")
        print(f"✓ 新生成 token 数: {generated.shape[1] - prompt.shape[1]}")
        
        # 验证所有 token 在词表范围内
        assert torch.all(generated >= 0) and torch.all(generated < vocab_size), "生成的 token 超出词表范围"
        print(f"✓ 所有 token 在词表范围内")
        
        print("✅ 文本生成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文本生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_with_temperature():
    """测试不同温度的生成"""
    print("\n" + "="*60)
    print("测试3: 不同温度的生成")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 准备输入
        prompt = torch.randint(0, vocab_size, (1, 5))
        
        # 测试不同温度
        temperatures = [0.1, 0.5, 1.0, 1.5]
        
        for temp in temperatures:
            generated = model.generate(
                prompt,
                max_new_tokens=10,
                temperature=temp,
                top_k=None,
            )
            
            assert generated.shape == (1, 15), f"温度 {temp} 生成形状错误: {generated.shape}"
            print(f"✓ 温度 {temp}: 生成成功")
        
        print("✅ 不同温度生成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 不同温度生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_with_top_k():
    """测试 top-k 采样"""
    print("\n" + "="*60)
    print("测试4: Top-k 采样")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 准备输入
        prompt = torch.randint(0, vocab_size, (1, 5))
        
        # 测试不同 top_k 值
        top_k_values = [10, 50, 100]
        
        for top_k in top_k_values:
            generated = model.generate(
                prompt,
                max_new_tokens=10,
                temperature=1.0,
                top_k=top_k,
            )
            
            assert generated.shape == (1, 15), f"top_k {top_k} 生成形状错误: {generated.shape}"
            print(f"✓ top_k {top_k}: 生成成功")
        
        print("✅ Top-k 采样测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Top-k 采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "="*60)
    print("测试5: 批量推理")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        batch_size = 4
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 准备批量输入
        prompts = torch.randint(0, vocab_size, (batch_size, 10))
        
        # 批量推理
        with torch.no_grad():
            logits, loss, caches = model(prompts)
        
        # 验证输出
        assert logits.shape == (batch_size, 10, vocab_size), f"批量输出形状错误: {logits.shape}"
        print(f"✓ 批量大小: {batch_size}")
        print(f"✓ 输入形状: {prompts.shape}")
        print(f"✓ 输出形状: {logits.shape}")
        
        print("✅ 批量推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 批量推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_with_tokenizer():
    """测试使用分词器的推理"""
    print("\n" + "="*60)
    print("测试6: 使用分词器的推理")
    print("="*60)
    
    try:
        from tools.utils import create_model
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        
        # 训练一个简单的分词器
        test_texts = ["人工智能", "机器学习", "深度学习"] * 50
        
        from tools.tokenizer_utils import create_test_tokenizer
        
        tokenizer = create_test_tokenizer(
            vocab_size=500,
            texts=test_texts,
            min_frequency=1,
            show_progress=False
        )
        
        print("✓ 分词器训练完成")
        
        # 创建模型
        vocab_size = len(tokenizer.get_vocab())
        block_size = 32
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.eval()
        
        # 编码文本
        text = "人工智能"
        input_ids = tokenizer.encode(text, return_tensors="pt")
        
        print(f"✓ 输入文本: '{text}'")
        print(f"✓ 编码后: {input_ids.shape}")
        
        # 推理
        with torch.no_grad():
            logits, loss, caches = model(input_ids)
        
        # 获取预测
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        
        # 解码
        next_token = tokenizer.decode([next_token_id.item()])
        print(f"✓ 预测下一个 token: '{next_token}'")
        
        print("✅ 使用分词器的推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 使用分词器的推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("推理测试套件")
    print("="*60)
    
    tests = [
        ("模型推理", test_model_inference),
        ("文本生成", test_text_generation),
        ("不同温度的生成", test_generation_with_temperature),
        ("Top-k 采样", test_generation_with_top_k),
        ("批量推理", test_batch_inference),
        ("使用分词器的推理", test_inference_with_tokenizer),
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
