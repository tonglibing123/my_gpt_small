#!/usr/bin/env python3
"""
PPO 测试脚本
测试 PPOTrainer 和 PPO 训练流程
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入工具函数
from tools.tokenizer_utils import create_simple_tokenizer


def test_ppo_trainer_creation():
    """测试 PPO 训练器创建"""
    print("\n" + "="*60)
    print("测试 1: PPO 训练器创建")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from src.training.ppo_trainer import PPOTrainer
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(vocab_size=1000, texts=["hello world"])
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_kwargs = dict(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        # Policy
        policy = MyGPT(**base_kwargs)
        policy.value_head = torch.nn.Linear(64, 1, bias=False)
        
        # Reference
        ref_policy = MyGPT(**base_kwargs)
        for p in ref_policy.parameters():
            p.requires_grad = False
        ref_policy.eval()
        
        # Reward Model
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        for p in rm.parameters():
            p.requires_grad = False
        rm.eval()
        
        # 创建 PPO 训练器
        trainer = PPOTrainer(
            policy=policy,
            ref_policy=ref_policy,
            reward_model=rm,
            value_model=policy,
            tokenizer=tokenizer,
            clip_eps=0.2,
            beta=0.01,
            eps_v=0.2
        )
        
        print(f"✅ PPO 训练器创建成功")
        print(f"   - Clip epsilon: {trainer.clip_eps}")
        print(f"   - KL beta: {trainer.beta}")
        print(f"   - Value epsilon: {trainer.eps_v}")
        print(f"   - Pad token ID: {trainer.pad_token_id}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_collect():
    """测试 PPO rollout 收集"""
    print("\n" + "="*60)
    print("测试 2: PPO Rollout 收集")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from src.training.ppo_trainer import PPOTrainer
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(vocab_size=1000, texts=["hello world test"])
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_kwargs = dict(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        policy = MyGPT(**base_kwargs)
        policy.value_head = torch.nn.Linear(64, 1, bias=False)
        policy.eval()
        
        ref_policy = MyGPT(**base_kwargs)
        ref_policy.eval()
        
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        rm.eval()
        
        trainer = PPOTrainer(
            policy=policy,
            ref_policy=ref_policy,
            reward_model=rm,
            value_model=policy,
            tokenizer=tokenizer
        )
        
        # 测试 rollout 收集
        batch_size = 2
        prompt_len = 16
        prompts = torch.randint(1, 1000, (batch_size, prompt_len))
        
        rollouts = trainer.collect(prompts, max_new_tokens=16)
        
        print(f"✅ Rollout 收集成功")
        print(f"   - Sequences 形状: {rollouts['sequences'].shape}")
        print(f"   - Logprobs 形状: {rollouts['logprobs'].shape}")
        print(f"   - Values 形状: {rollouts['values'].shape}")
        print(f"   - Rewards 形状: {rollouts['rewards'].shape}")
        
        # 验证输出
        assert 'sequences' in rollouts
        assert 'logprobs' in rollouts
        assert 'values' in rollouts
        assert 'rewards' in rollouts
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_epoch():
    """测试 PPO 更新"""
    print("\n" + "="*60)
    print("测试 3: PPO 更新")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from src.training.ppo_trainer import PPOTrainer
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(vocab_size=1000, texts=["hello world"])
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_kwargs = dict(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        policy = MyGPT(**base_kwargs)
        policy.value_head = torch.nn.Linear(64, 1, bias=False)
        policy.train()
        
        ref_policy = MyGPT(**base_kwargs)
        ref_policy.eval()
        
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        rm.eval()
        
        trainer = PPOTrainer(
            policy=policy,
            ref_policy=ref_policy,
            reward_model=rm,
            value_model=policy,
            tokenizer=tokenizer
        )
        
        # 创建模拟 rollouts
        batch_size = 4
        seq_len = 32
        
        rollouts = {
            'sequences': torch.randint(1, 1000, (batch_size, seq_len)),
            'logprobs': torch.randn(batch_size, seq_len - 1),
            'values': torch.randn(batch_size, seq_len - 1),
            'rewards': torch.randn(batch_size, seq_len - 1)
        }
        
        # 执行 PPO 更新
        trainer.ppo_epoch(rollouts, mini_batch=2)
        
        print(f"✅ PPO 更新成功")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 序列长度: {seq_len}")
        print(f"   - Mini batch: 2")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_update_reference():
    """测试 Reference 更新"""
    print("\n" + "="*60)
    print("测试 4: Reference 更新")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from src.training.ppo_trainer import PPOTrainer
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(vocab_size=1000, texts=["hello"])
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_kwargs = dict(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        policy = MyGPT(**base_kwargs)
        policy.value_head = torch.nn.Linear(64, 1, bias=False)
        
        ref_policy = MyGPT(**base_kwargs)
        
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        
        trainer = PPOTrainer(
            policy=policy,
            ref_policy=ref_policy,
            reward_model=rm,
            value_model=policy,
            tokenizer=tokenizer
        )
        
        # 修改 policy 权重
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        
        # 更新 reference
        trainer.update_reference()
        
        # 验证权重已复制
        policy_params = list(policy.parameters())
        ref_params = list(ref_policy.parameters())
        
        all_equal = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(policy_params, ref_params)
        )
        
        print(f"✅ Reference 更新成功")
        print(f"   - 权重已复制: {all_equal}")
        print(f"   - Reference 模式: eval")
        
        assert all_equal, "Reference 权重未正确复制"
        assert not ref_policy.training, "Reference 应该在 eval 模式"
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_value_head_creation():
    """测试 Value Head 创建"""
    print("\n" + "="*60)
    print("测试 5: Value Head 创建")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        # 添加 value_head
        model.value_head = torch.nn.Linear(64, 1, bias=False)
        torch.nn.init.normal_(model.value_head.weight, mean=0.0, std=0.02)
        
        # 测试前向传播
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, 1000, (batch_size, seq_len))
        
        logits, values, _ = model(idx, return_value=True)
        
        print(f"✅ Value Head 创建成功")
        print(f"   - Logits 形状: {logits.shape}")
        print(f"   - Values 形状: {values.shape}")
        print(f"   - Value Head 参数: {sum(p.numel() for p in model.value_head.parameters())}")
        
        # 验证形状
        assert logits.shape == (batch_size, seq_len, 1000)
        assert values.shape == (batch_size, seq_len)
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_with_short_sequences():
    """测试 PPO 处理短序列"""
    print("\n" + "="*60)
    print("测试 6: PPO 处理短序列")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from src.training.ppo_trainer import PPOTrainer
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(vocab_size=1000, texts=["test"])
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_kwargs = dict(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        policy = MyGPT(**base_kwargs)
        policy.value_head = torch.nn.Linear(64, 1, bias=False)
        policy.eval()
        
        ref_policy = MyGPT(**base_kwargs)
        ref_policy.eval()
        
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        rm.eval()
        
        trainer = PPOTrainer(
            policy=policy,
            ref_policy=ref_policy,
            reward_model=rm,
            value_model=policy,
            tokenizer=tokenizer
        )
        
        # 测试非常短的 prompt
        batch_size = 2
        prompt_len = 2  # 非常短
        prompts = torch.randint(1, 1000, (batch_size, prompt_len))
        
        # 应该能处理短序列而不崩溃
        rollouts = trainer.collect(prompts, max_new_tokens=4)
        
        print(f"✅ 短序列处理成功")
        print(f"   - Prompt 长度: {prompt_len}")
        print(f"   - 生成长度: 4")
        print(f"   - Sequences 形状: {rollouts['sequences'].shape}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("  PPO 测试套件")
    print("="*70)
    
    tests = [
        ("PPO 训练器创建", test_ppo_trainer_creation),
        ("PPO Rollout 收集", test_ppo_collect),
        ("PPO 更新", test_ppo_epoch),
        ("Reference 更新", test_ppo_update_reference),
        ("Value Head 创建", test_value_head_creation),
        ("PPO 短序列处理", test_ppo_with_short_sequences),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        result = test_func()
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:  # None = skipped
            skipped += 1
    
    print("\n" + "="*70)
    print(f"  测试结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("="*70 + "\n")
    
    if failed == 0:
        print("🎉 所有测试通过！")
        if skipped > 0:
            print(f"   ({skipped} 个测试因缺少依赖被跳过)")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
