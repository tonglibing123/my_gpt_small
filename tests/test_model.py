# test_model.py - 模型测试代码
import torch
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.utils import create_model


def test_model(name, **model_kwargs):
    """通用模型测试函数"""
    print(f"{'=' * 50}\n测试 {name}\n{'=' * 50}")
    model = create_model(**model_kwargs)
    print(f"模型参数量: {model.get_num_params() / 1e6:.2f}M")
    
    x = torch.randint(0, model_kwargs.get('vocab_size', 6400), (2, 64))
    logits, loss, _ = model(x, targets=x)
    print(f"输出形状: {logits.shape}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(x[:1, :10], max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"生成序列长度: {generated.shape[1]}")
    print(f"{name} 测试通过！\n")
    return model


def test_value_head():
    """测试 Value Head（用于 RLHF）"""
    print(f"{'=' * 50}\n测试 Value Head\n{'=' * 50}")
    model = create_model(vocab_size=6400, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=256)
    model.value_head = torch.nn.Linear(model.n_embd, 1, bias=False)
    
    x = torch.randint(0, 6400, (2, 64))
    logits, values, _ = model(x, return_value=True)
    print(f"Logits: {logits.shape}, Values: {values.shape}")
    print("Value Head 测试通过！\n")


def test_top_p_sampling():
    """测试 Top-P 采样"""
    print(f"{'=' * 50}\n测试 Top-P 采样\n{'=' * 50}")
    model = create_model(vocab_size=6400, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=256)
    model.eval()
    with torch.no_grad():
        generated = model.generate(torch.randint(0, 6400, (1, 10)), max_new_tokens=20, top_k=None, top_p=0.9)
    print(f"Top-P 生成长度: {generated.shape[1]}")
    print("Top-P 采样测试通过！\n")


def run_all_tests():
    print("\n" + "=" * 60 + "\n开始运行所有模型测试\n" + "=" * 60 + "\n")
    try:
        # 教学用标准配置（与训练脚本一致）
        test_model("Standard GPT", vocab_size=6400, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=256)
        test_value_head()
        test_top_p_sampling()
        print("=" * 60 + "\n所有测试完成！\n" + "=" * 60)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
