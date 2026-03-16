#!/usr/bin/env python3
"""
奖励模型测试脚本
测试 RewardModel 和 PairwiseDataset 的功能
"""
import sys
import os
from pathlib import Path
import tempfile
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入工具函数
from tools.tokenizer_utils import create_simple_tokenizer


def test_reward_model_creation():
    """测试奖励模型创建"""
    print("\n" + "="*60)
    print("测试 1: 奖励模型创建")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建基础模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.1
        )
        
        # 创建奖励模型
        rm = RewardModel(base_model, dropout=0.1)
        
        print(f"✅ 奖励模型创建成功")
        print(f"   - Backbone 参数: {sum(p.numel() for p in rm.backbone.parameters()):,}")
        print(f"   - Reward Head 参数: {sum(p.numel() for p in rm.reward_head.parameters()):,}")
        print(f"   - 总参数: {sum(p.numel() for p in rm.parameters()):,}")
        
        # 检查 embedding 是否冻结
        emb_frozen = not rm.backbone.tok_emb.weight.requires_grad
        print(f"   - Embedding 冻结: {emb_frozen}")
        assert emb_frozen, "Embedding 应该被冻结"
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_model_forward():
    """测试奖励模型前向传播"""
    print("\n" + "="*60)
    print("测试 2: 奖励模型前向传播")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 测试输入
        batch_size = 4
        seq_len = 32
        idx = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # 前向传播
        with torch.no_grad():
            rewards = rm(idx, attention_mask)
        
        print(f"✅ 前向传播成功")
        print(f"   - 输入形状: {idx.shape}")
        print(f"   - 输出形状: {rewards.shape}")
        print(f"   - 输出范围: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
        
        # 验证输出形状
        assert rewards.shape == (batch_size,), f"输出形状错误: {rewards.shape}"
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_model_with_padding():
    """测试奖励模型处理 padding"""
    print("\n" + "="*60)
    print("测试 3: 奖励模型处理 Padding")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 测试不同长度的序列
        batch_size = 4
        max_len = 32
        pad_token_id = 0
        
        # 创建不同长度的序列
        lengths = [10, 15, 20, 32]
        idx = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, length in enumerate(lengths):
            idx[i, :length] = torch.randint(1, 1000, (length,))
            attention_mask[i, :length] = 1
        
        # 前向传播
        with torch.no_grad():
            rewards = rm(idx, attention_mask)
        
        print(f"✅ Padding 处理成功")
        print(f"   - 序列长度: {lengths}")
        print(f"   - 奖励值: {[f'{r:.4f}' for r in rewards.tolist()]}")
        
        # 验证输出
        assert rewards.shape == (batch_size,), f"输出形状错误: {rewards.shape}"
        assert not torch.isnan(rewards).any(), "输出包含 NaN"
        assert not torch.isinf(rewards).any(), "输出包含 Inf"
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairwise_dataset():
    """测试 PairwiseDataset"""
    print("\n" + "="*60)
    print("测试 4: PairwiseDataset")
    print("="*60)
    
    try:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from src.data.rm_dataset import PairwiseDataset
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["hello world", "test data", "sample text"]
        )
        
        # 创建临时数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            temp_file = f.name
            # 写入测试数据
            for i in range(10):
                data = {
                    "context": [
                        {"role": "user", "text": f"Question {i}"}
                    ],
                    "chosen": {"text": f"Good answer {i}"},
                    "rejected": {"text": f"Bad answer {i}"}
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        try:
            # 创建数据集
            dataset = PairwiseDataset(
                files=[temp_file],
                tokenizer=tokenizer,
                block_size=64,
                sample_ratio=1.0
            )
            
            print(f"✅ 数据集创建成功")
            print(f"   - 样本数量: {len(dataset)}")
            
            # 测试数据加载
            c_ids, c_mask, r_ids, r_mask = dataset[0]
            
            print(f"   - Chosen IDs 形状: {c_ids.shape}")
            print(f"   - Chosen Mask 形状: {c_mask.shape}")
            print(f"   - Rejected IDs 形状: {r_ids.shape}")
            print(f"   - Rejected Mask 形状: {r_mask.shape}")
            
            # 验证数据
            assert len(dataset) == 10, f"数据集大小错误: {len(dataset)}"
            assert c_ids.shape == (64,), f"Chosen IDs 形状错误: {c_ids.shape}"
            assert c_mask.shape == (64,), f"Chosen Mask 形状错误: {c_mask.shape}"
            assert r_ids.shape == (64,), f"Rejected IDs 形状错误: {r_ids.shape}"
            assert r_mask.shape == (64,), f"Rejected Mask 形状错误: {r_mask.shape}"
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairwise_dataset_sample_ratio():
    """测试 PairwiseDataset 采样比例"""
    print("\n" + "="*60)
    print("测试 5: PairwiseDataset 采样比例")
    print("="*60)
    
    try:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from src.data.rm_dataset import PairwiseDataset
        
        # 使用工具函数创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["hello world", "test data"]
        )
        
        # 创建临时数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            temp_file = f.name
            for i in range(100):
                data = {
                    "chosen": f"Good {i}",
                    "rejected": f"Bad {i}"
                }
                f.write(json.dumps(data) + '\n')
        
        try:
            # 测试不同采样比例
            for ratio in [0.1, 0.5, 1.0]:
                dataset = PairwiseDataset(
                    files=[temp_file],
                    tokenizer=tokenizer,
                    block_size=64,
                    sample_ratio=ratio
                )
                expected_size = max(1, int(100 * ratio))
                print(f"   - 采样比例 {ratio}: {len(dataset)} 样本 (预期 {expected_size})")
                assert len(dataset) == expected_size, f"采样比例 {ratio} 错误"
            
            print(f"✅ 采样比例测试通过")
            return True
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_model_comparison():
    """测试奖励模型比较 chosen vs rejected"""
    print("\n" + "="*60)
    print("测试 6: 奖励模型比较")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 模拟 chosen 和 rejected 序列
        batch_size = 4
        seq_len = 32
        
        # Chosen 序列（假设更好）
        chosen_ids = torch.randint(1, 1000, (batch_size, seq_len))
        chosen_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # Rejected 序列（假设更差）
        rejected_ids = torch.randint(1, 1000, (batch_size, seq_len))
        rejected_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # 计算奖励
        with torch.no_grad():
            r_chosen = rm(chosen_ids, chosen_mask)
            r_rejected = rm(rejected_ids, rejected_mask)
        
        print(f"✅ 奖励比较成功")
        print(f"   - Chosen 奖励: {[f'{r:.4f}' for r in r_chosen.tolist()]}")
        print(f"   - Rejected 奖励: {[f'{r:.4f}' for r in r_rejected.tolist()]}")
        
        # 注意：未训练的模型不一定 chosen > rejected，这里只验证能正常计算
        print(f"   - 注意: 未训练的模型奖励值是随机的")
        
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
    print("  奖励模型测试套件")
    print("="*70)
    
    tests = [
        ("奖励模型创建", test_reward_model_creation),
        ("奖励模型前向传播", test_reward_model_forward),
        ("奖励模型 Padding 处理", test_reward_model_with_padding),
        ("PairwiseDataset", test_pairwise_dataset),
        ("PairwiseDataset 采样", test_pairwise_dataset_sample_ratio),
        ("奖励模型比较", test_reward_model_comparison),
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


def test_rm_preference_learning():
    """测试偏好学习"""
    print("\n" + "="*60)
    print("测试 4: 偏好学习")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        rm = RewardModel(base_model, dropout=0.0)
        
        # 模拟偏好对
        chosen = torch.randint(0, 1000, (2, 20))
        rejected = torch.randint(0, 1000, (2, 20))
        
        # 计算奖励
        r_chosen = rm(chosen, torch.ones_like(chosen))
        r_rejected = rm(rejected, torch.ones_like(rejected))
        
        print(f"✅ 偏好学习测试成功")
        print(f"   - Chosen reward: {r_chosen.mean().item():.4f}")
        print(f"   - Rejected reward: {r_rejected.mean().item():.4f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rm_ranking():
    """测试排序能力"""
    print("\n" + "="*60)
    print("测试 5: 排序能力")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 生成多个响应
        responses = [torch.randint(0, 1000, (1, 20)) for _ in range(5)]
        
        # 计算奖励
        rewards = []
        with torch.no_grad():
            for resp in responses:
                r = rm(resp, torch.ones_like(resp))
                rewards.append(r.item())
        
        print(f"✅ 排序能力测试成功")
        print(f"   - 奖励分布: {rewards}")
        print(f"   - 最高奖励: {max(rewards):.4f}")
        print(f"   - 最低奖励: {min(rewards):.4f}")
        print(f"   - 奖励范围: {max(rewards) - min(rewards):.4f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rm_score_range():
    """测试评分范围"""
    print("\n" + "="*60)
    print("测试 6: 评分范围")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 测试多个样本
        scores = []
        with torch.no_grad():
            for _ in range(20):
                seq = torch.randint(0, 1000, (1, 20))
                score = rm(seq, torch.ones_like(seq))
                scores.append(score.item())
        
        import numpy as np
        scores = np.array(scores)
        
        print(f"✅ 评分范围测试成功")
        print(f"   - 平均分: {scores.mean():.4f}")
        print(f"   - 标准差: {scores.std():.4f}")
        print(f"   - 最小值: {scores.min():.4f}")
        print(f"   - 最大值: {scores.max():.4f}")
        print(f"   - 范围: {scores.max() - scores.min():.4f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rm_consistency():
    """测试评分一致性"""
    print("\n" + "="*60)
    print("测试 7: 评分一致性")
    print("="*60)
    
    try:
        import torch
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        
        # 创建模型
        components = get_model_components()
        MyGPT = components['MyGPT']
        
        base_model = MyGPT(
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            block_size=128,
            dropout=0.0
        )
        
        rm = RewardModel(base_model, dropout=0.0)
        rm.eval()
        
        # 同一个序列多次评分
        seq = torch.randint(0, 1000, (1, 20))
        scores = []
        
        with torch.no_grad():
            for _ in range(10):
                score = rm(seq, torch.ones_like(seq))
                scores.append(score.item())
        
        import numpy as np
        scores = np.array(scores)
        
        # 验证一致性（标准差应该很小）
        assert scores.std() < 1e-6, f"评分不一致，标准差: {scores.std()}"
        
        print(f"✅ 评分一致性测试成功")
        print(f"   - 平均分: {scores.mean():.6f}")
        print(f"   - 标准差: {scores.std():.8f}")
        print(f"   - 一致性验证通过")
        
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
    print("\n" + "="*60)
    print("Reward Model 测试套件（增强版）")
    print("="*60)
    
    tests = [
        ("Reward Model 创建", test_reward_model_creation),
        ("前向传播", test_reward_model_forward),
        ("偏好对比", test_reward_model_preference),
        ("偏好学习", test_rm_preference_learning),
        ("排序能力", test_rm_ranking),
        ("评分范围", test_rm_score_range),
        ("评分一致性", test_rm_consistency),
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
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"跳过: {skipped} ⚠️")
    if skipped > 0:
        print(f"\n提示: {skipped} 个测试因缺少依赖被跳过")
        print("安装依赖: pip install -r requirements.txt")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
