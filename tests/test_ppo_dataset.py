#!/usr/bin/env python3
"""
PPO数据集测试
测试 PromptDataset 和 RLHF 数据加载
"""
import sys
import os
from pathlib import Path
import tempfile
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_prompt_dataset_creation():
    """测试 PromptDataset 创建"""
    print("\n" + "="*60)
    print("测试 1: PromptDataset 创建")
    print("="*60)
    
    try:
        from src.training.train_ppo import PromptDataset
        from tools.tokenizer_utils import create_simple_tokenizer
        
        # 创建临时JSONL文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        
        # 写入测试数据
        test_data = [
            {
                "context": [
                    {"role": "human", "text": "你好"},
                    {"role": "assistant", "text": "你好！有什么可以帮助你的吗？"},
                    {"role": "human", "text": "请介绍一下自己"}
                ],
                "chosen": {"role": "assistant", "text": "我是一个AI助手"},
                "rejected": {"role": "assistant", "text": "我不知道"}
            },
            {
                "context": [
                    {"role": "human", "text": "今天天气怎么样？"}
                ],
                "chosen": {"role": "assistant", "text": "今天天气很好"},
                "rejected": {"role": "assistant", "text": "不清楚"}
            },
            {
                "context": [
                    {"role": "human", "text": "什么是人工智能？"},
                    {"role": "assistant", "text": "人工智能是计算机科学的一个分支"},
                    {"role": "human", "text": "它有什么应用？"}
                ],
                "chosen": {"role": "assistant", "text": "人工智能有很多应用"},
                "rejected": {"role": "assistant", "text": "不太了解"}
            }
        ]
        
        for item in test_data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        
        # 创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["你好 世界 人工智能"],
            show_progress=False
        )
        
        # 创建数据集
        dataset = PromptDataset(
            file_path=temp_file.name,
            tokenizer=tokenizer,
            max_length=64,
            sample_ratio=1.0
        )
        
        print(f"✅ PromptDataset 创建成功")
        print(f"   - 数据集大小: {len(dataset)}")
        print(f"   - 预期大小: {len(test_data)}")
        
        assert len(dataset) == len(test_data), f"数据集大小不匹配: {len(dataset)} != {len(test_data)}"
        
        # 清理
        os.unlink(temp_file.name)
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_dataset_getitem():
    """测试 PromptDataset __getitem__"""
    print("\n" + "="*60)
    print("测试 2: PromptDataset __getitem__")
    print("="*60)
    
    try:
        from src.training.train_ppo import PromptDataset
        from tools.tokenizer_utils import create_simple_tokenizer
        import torch
        
        # 创建临时JSONL文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        
        test_data = [
            {
                "context": [
                    {"role": "human", "text": "你好"},
                    {"role": "assistant", "text": "你好！"},
                    {"role": "human", "text": "请介绍一下"}
                ],
                "chosen": {"role": "assistant", "text": "好的"},
                "rejected": {"role": "assistant", "text": "不"}
            }
        ]
        
        for item in test_data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        
        # 创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["你好 世界"],
            show_progress=False
        )
        
        # 创建数据集
        dataset = PromptDataset(
            file_path=temp_file.name,
            tokenizer=tokenizer,
            max_length=64,
            sample_ratio=1.0
        )
        
        # 获取一个样本
        prompt_ids, attention_mask = dataset[0]
        
        print(f"✅ __getitem__ 测试成功")
        print(f"   - prompt_ids 类型: {type(prompt_ids)}")
        print(f"   - prompt_ids 形状: {prompt_ids.shape}")
        print(f"   - attention_mask 形状: {attention_mask.shape}")
        
        # 验证
        assert isinstance(prompt_ids, torch.Tensor), "prompt_ids 应该是 Tensor"
        assert isinstance(attention_mask, torch.Tensor), "attention_mask 应该是 Tensor"
        assert prompt_ids.shape[0] == 64, f"prompt_ids 长度应该是 64，实际: {prompt_ids.shape[0]}"
        assert attention_mask.shape[0] == 64, f"attention_mask 长度应该是 64，实际: {attention_mask.shape[0]}"
        
        # 清理
        os.unlink(temp_file.name)
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_dataset_sampling():
    """测试 PromptDataset 采样"""
    print("\n" + "="*60)
    print("测试 3: PromptDataset 采样")
    print("="*60)
    
    try:
        from src.training.train_ppo import PromptDataset
        from tools.tokenizer_utils import create_simple_tokenizer
        
        # 创建临时JSONL文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        
        # 写入100条数据
        for i in range(100):
            item = {
                "context": [
                    {"role": "human", "text": f"问题{i}"}
                ],
                "chosen": {"role": "assistant", "text": f"回答{i}"},
                "rejected": {"role": "assistant", "text": "不知道"}
            }
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        
        # 创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["测试"],
            show_progress=False
        )
        
        # 测试不同采样比例
        for ratio in [0.1, 0.5, 1.0]:
            dataset = PromptDataset(
                file_path=temp_file.name,
                tokenizer=tokenizer,
                max_length=64,
                sample_ratio=ratio
            )
            
            expected_size = max(1, int(100 * ratio))
            actual_size = len(dataset)
            
            print(f"✅ 采样比例 {ratio}: 预期={expected_size}, 实际={actual_size}")
            
            assert actual_size == expected_size, f"采样大小不匹配: {actual_size} != {expected_size}"
        
        # 清理
        os.unlink(temp_file.name)
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_dataset_with_real_data():
    """测试 PromptDataset 使用真实数据"""
    print("\n" + "="*60)
    print("测试 4: PromptDataset 使用真实数据")
    print("="*60)
    
    try:
        from src.training.train_ppo import PromptDataset
        from tools.tokenizer_utils import create_simple_tokenizer
        
        # 检查真实数据文件是否存在
        data_file = "minimind_dataset/hh_rlhf_cn/helpful_base_cn_train.jsonl"
        if not os.path.exists(data_file):
            print(f"⚠️  跳过测试: 数据文件不存在 {data_file}")
            return None
        
        # 创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["你好 世界 人工智能"],
            show_progress=False
        )
        
        # 创建数据集（使用小采样比例）
        dataset = PromptDataset(
            file_path=data_file,
            tokenizer=tokenizer,
            max_length=64,
            sample_ratio=0.001  # 只采样0.1%
        )
        
        print(f"✅ 真实数据加载成功")
        print(f"   - 数据集大小: {len(dataset)}")
        print(f"   - 数据文件: {data_file}")
        
        # 获取一个样本
        prompt_ids, attention_mask = dataset[0]
        print(f"   - 样本形状: {prompt_ids.shape}")
        
        assert len(dataset) > 0, "数据集不能为空"
        
        return True
        
    except ImportError as e:
        print(f"⚠️  跳过测试: 缺少依赖 {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_policy_function():
    """测试 evaluate_policy 函数"""
    print("\n" + "="*60)
    print("测试 5: evaluate_policy 函数")
    print("="*60)
    
    try:
        import torch
        from src.training.train_ppo import evaluate_policy
        from tools.utils import get_model_components
        from src.models.reward_model import RewardModel
        from tools.tokenizer_utils import create_simple_tokenizer
        
        # 创建分词器
        tokenizer = create_simple_tokenizer(
            vocab_size=1000,
            texts=["hello world"],
            show_progress=False
        )
        
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
        policy.eval()
        
        # Reward Model
        rm = RewardModel(MyGPT(**base_kwargs), dropout=0.0)
        rm.eval()
        
        # 准备评估prompts
        eval_prompts = [
            torch.randint(1, 1000, (16,)),
            torch.randint(1, 1000, (16,)),
        ]
        
        # 评估
        avg_reward = evaluate_policy(
            policy=policy,
            reward_model=rm,
            tokenizer=tokenizer,
            eval_prompts=eval_prompts,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id
        )
        
        print(f"✅ evaluate_policy 测试成功")
        print(f"   - 平均奖励: {avg_reward:.4f}")
        print(f"   - 评估样本数: {len(eval_prompts)}")
        
        assert isinstance(avg_reward, float), "avg_reward 应该是 float"
        
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
    print("  PPO 数据集测试套件")
    print("="*70)
    
    tests = [
        ("PromptDataset 创建", test_prompt_dataset_creation),
        ("PromptDataset __getitem__", test_prompt_dataset_getitem),
        ("PromptDataset 采样", test_prompt_dataset_sampling),
        ("PromptDataset 真实数据", test_prompt_dataset_with_real_data),
        ("evaluate_policy 函数", test_evaluate_policy_function),
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
