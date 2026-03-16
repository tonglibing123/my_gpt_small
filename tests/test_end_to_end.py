#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端集成测试
测试完整的训练和推理流程（从数据到模型到推理）
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil
import torch
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_full_pipeline_small():
    """测试完整流程（小规模）：数据准备→分词器→预处理→训练→保存→加载→推理"""
    print("\n" + "="*60)
    print("测试: 完整端到端流程（小规模）")
    print("="*60)
    
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        from tools.utils import create_model
        from torch.utils.data import TensorDataset, DataLoader
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"✓ 临时目录: {temp_dir}")
        
        # ========== 步骤1: 准备数据 ==========
        print("\n步骤1: 准备数据")
        test_texts = [
            "人工智能在未来将会改变世界",
            "深度学习是机器学习的重要分支",
            "自然语言处理技术发展迅速",
            "大语言模型具有强大的能力",
        ] * 20
        
        print(f"✓ 准备了 {len(test_texts)} 条文本")
        
        # ========== 步骤2: 训练分词器 ==========
        print("\n步骤2: 训练分词器")
        from tools.tokenizer_utils import create_test_tokenizer
        
        tokenizer = create_test_tokenizer(
            vocab_size=500,
            texts=test_texts,
            min_frequency=1,
            show_progress=False
        )
        
        vocab_size = len(tokenizer.get_vocab())
        print(f"✓ 分词器训练完成，词表大小: {vocab_size}")
        
        # ========== 步骤3: 预处理数据 ==========
        print("\n步骤3: 预处理数据")
        all_tokens = []
        for text in test_texts:
            ids = tokenizer.encode(text)
            all_tokens.extend(ids)
        
        block_size = 32
        blocks = []
        for i in range(0, len(all_tokens) - block_size, block_size):
            block = all_tokens[i:i+block_size+1]
            if len(block) == block_size + 1:
                blocks.append(block)
        
        print(f"✓ 预处理完成: {len(blocks)} 个 block")
        
        # ========== 步骤4: 创建数据集 ==========
        print("\n步骤4: 创建数据集")
        x_data = torch.tensor([b[:-1] for b in blocks], dtype=torch.long)
        y_data = torch.tensor([b[1:] for b in blocks], dtype=torch.long)
        
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # ========== 步骤5: 创建模型 ==========
        print("\n步骤5: 创建模型")
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        
        print(f"✓ 模型参数量: {model.get_num_params() / 1e6:.2f}M")
        
        # ========== 步骤6: 训练模型 ==========
        print("\n步骤6: 训练模型")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        n_epochs = 5
        for epoch in range(n_epochs):
            total_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
                logits, loss, caches = model(x, targets=y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 2 == 0:
                print(f"✓ Epoch {epoch}: loss={avg_loss:.4f}")
        
        # ========== 步骤7: 保存模型 ==========
        print("\n步骤7: 保存模型")
        ckpt_dir = os.path.join(temp_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        
        config = {
            "vocab_size": vocab_size,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 1,
            "n_embd": 64,
            "block_size": block_size,
        }
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        print(f"✓ 模型已保存到: {ckpt_dir}")
        
        # ========== 步骤8: 加载模型 ==========
        print("\n步骤8: 加载模型")
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            loaded_config = json.load(f)
        
        model_loaded = create_model(**loaded_config)
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"))
        model_loaded.load_state_dict(state_dict)
        model_loaded.eval()
        
        print(f"✓ 模型加载成功")
        
        # ========== 步骤9: 推理测试 ==========
        print("\n步骤9: 推理测试")
        test_text = "人工智能"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")
        
        with torch.no_grad():
            generated = model_loaded.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                top_k=50,
            )
        
        output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
        print(f"✓ 输入: '{test_text}'")
        print(f"✓ 输出: '{output_text}'")
        
        # 清理
        shutil.rmtree(temp_dir)
        
        print("\n✅ 完整流程测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行端到端集成测试"""
    print("\n" + "="*60)
    print("端到端集成测试套件")
    print("="*60)
    print("\n说明: 此测试验证完整的训练和推理流程")
    print("      从数据准备到模型训练再到推理生成\n")
    
    tests = [
        ("完整端到端流程", test_full_pipeline_small),
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
    print("\n提示: 其他集成测试已分散到各专项测试文件中")
    print("      - test_tokenizer.py: 分词器测试")
    print("      - test_training.py: 训练流程测试")
    print("      - test_inference.py: 推理测试")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
