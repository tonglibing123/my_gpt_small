#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练流程测试
测试模型训练、checkpoint 保存加载
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("测试1: 模型创建")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建小模型
        model = create_model(
            vocab_size=500,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=32,
            dropout=0.1
        )
        
        print(f"✓ 模型创建成功")
        print(f"✓ 参数量: {model.get_num_params() / 1e6:.2f}M")
        
        # 验证模型属性
        assert model.vocab_size == 500
        assert model.n_layer == 2
        assert model.n_head == 2
        assert model.n_embd == 64
        assert model.block_size == 32
        
        print("✓ 模型属性验证通过")
        
        print("✅ 模型创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试2: 前向传播")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        batch_size = 2
        
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
        x = torch.randint(0, vocab_size, (batch_size, block_size))
        
        # 前向传播
        with torch.no_grad():
            logits, loss, caches = model(x, targets=x)
        
        # 验证输出形状
        assert logits.shape == (batch_size, block_size, vocab_size), f"logits 形状错误: {logits.shape}"
        assert loss is not None, "loss 为 None"
        
        print(f"✓ 输入形状: {x.shape}")
        print(f"✓ 输出形状: {logits.shape}")
        print(f"✓ Loss: {loss.item():.4f}")
        
        print("✅ 前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试单步训练"""
    print("\n" + "="*60)
    print("测试3: 单步训练")
    print("="*60)
    
    try:
        from tools.utils import create_model
        
        # 创建模型
        vocab_size = 500
        block_size = 32
        batch_size = 2
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
            block_size=block_size,
        )
        model.train()
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 准备数据
        x = torch.randint(0, vocab_size, (batch_size, block_size))
        y = torch.randint(0, vocab_size, (batch_size, block_size))
        
        # 训练前的 loss
        with torch.no_grad():
            logits_before, loss_before, caches = model(x, targets=y)
        
        print(f"✓ 训练前 loss: {loss_before.item():.4f}")
        
        # 训练一步
        optimizer.zero_grad()
        logits, loss, caches = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        print(f"✓ 训练后 loss: {loss.item():.4f}")
        
        # 验证 loss 有变化
        assert loss.item() != loss_before.item(), "loss 没有变化"
        
        print("✅ 单步训练测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 单步训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试 checkpoint 保存和加载"""
    print("\n" + "="*60)
    print("测试4: Checkpoint 保存和加载")
    print("="*60)
    
    try:
        from tools.utils import create_model
        import json
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        ckpt_dir = os.path.join(temp_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 创建模型
        model_config = {
            'vocab_size': 500,
            'n_layer': 2,
            'n_head': 2,
            'n_kv_head': 1,
            'n_embd': 64,
            'block_size': 32,
        }
        
        model = create_model(**model_config)
        
        # 保存 checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        
        # 保存配置
        config = {k: getattr(model, k) for k in ["vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "block_size"]}
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        print(f"✓ Checkpoint 已保存到: {ckpt_dir}")
        
        # 加载 checkpoint
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            loaded_config = json.load(f)
        
        model_loaded = create_model(**loaded_config)
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"))
        model_loaded.load_state_dict(state_dict)
        
        print(f"✓ Checkpoint 已加载")
        
        # 验证权重一致
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_loaded.named_parameters()):
            assert name1 == name2, f"参数名不匹配: {name1} != {name2}"
            assert torch.allclose(param1, param2), f"参数值不匹配: {name1}"
        
        print(f"✓ 权重验证通过")
        
        # 清理
        shutil.rmtree(temp_dir)
        
        print("✅ Checkpoint 保存加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint 保存加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_convergence():
    """测试训练收敛（小数据集）"""
    print("\n" + "="*60)
    print("测试5: 训练收敛")
    print("="*60)
    
    try:
        from tools.utils import create_model
        from torch.utils.data import TensorDataset, DataLoader
        
        # 创建模型
        vocab_size = 100
        block_size = 16
        batch_size = 4
        
        model = create_model(
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=32,
            block_size=block_size,
        )
        model.train()
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 创建简单的训练数据（重复模式）
        n_samples = 20
        x_data = torch.randint(0, vocab_size, (n_samples, block_size))
        y_data = x_data.clone()  # 简单的复制任务
        
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练多个 epoch
        n_epochs = 10
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
                logits, loss, caches = model(x, targets=y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"✓ Epoch {epoch}: loss={avg_loss:.4f}")
        
        # 验证 loss 下降
        assert losses[-1] < losses[0], f"Loss 没有下降: {losses[0]:.4f} -> {losses[-1]:.4f}"
        print(f"✓ Loss 从 {losses[0]:.4f} 下降到 {losses[-1]:.4f}")
        
        print("✅ 训练收敛测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练收敛测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("训练流程测试套件")
    print("="*60)
    
    tests = [
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("单步训练", test_training_step),
        ("Checkpoint 保存加载", test_checkpoint_save_load),
        ("训练收敛", test_training_convergence),
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
