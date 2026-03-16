#!/usr/bin/env python3
"""
导入测试脚本 - 验证所有模块导入是否正常
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_import(module_path, description):
    """测试单个模块导入"""
    try:
        parts = module_path.split('.')
        if len(parts) == 1:
            __import__(module_path)
        else:
            # 对于函数导入，只测试模块本身
            if parts[-1] in ['get_model_components', 'load_data_module', 'create_model']:
                module_name = '.'.join(parts[:-1])
                module = __import__(module_name, fromlist=[parts[-1]])
                # 检查函数是否存在
                if hasattr(module, parts[-1]):
                    print(f"✅ {description:40s} - {module_path}")
                    return True
                else:
                    print(f"❌ {description:40s} - {module_path}")
                    print(f"   错误: 模块中不存在 {parts[-1]}")
                    return False
            else:
                __import__(module_path)
        print(f"✅ {description:40s} - {module_path}")
        return True
    except ModuleNotFoundError as e:
        # 检查是否是因为缺少依赖（torch, transformers等）
        missing_dep = str(e).split("'")[1] if "'" in str(e) else None
        if missing_dep in ['torch', 'transformers', 'deepspeed', 'tokenizers', 'numpy', 'tqdm']:
            print(f"⚠️  {description:40s} - {module_path}")
            print(f"   跳过: 缺少依赖 {missing_dep}")
            return None  # 返回 None 表示跳过
        else:
            print(f"❌ {description:40s} - {module_path}")
            print(f"   错误: {e}")
            return False
    except Exception as e:
        print(f"❌ {description:40s} - {module_path}")
        print(f"   错误: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  导入测试 - 验证所有模块路径正确")
    print("="*70 + "\n")
    
    tests = [
        # 工具模块
        ("tools.utils", "工具函数模块"),
        ("tools.utils.get_model_components", "获取模型组件"),
        ("tools.utils.load_data_module", "加载数据模块"),
        ("tools.utils.create_model", "创建模型函数"),
        
        # 模型模块
        ("src.models.model", "模型定义模块"),
        ("src.models.components", "通用组件模块"),
        
        # 数据模块
        ("src.data.dataset", "数据集模块"),
        
        # 训练模块
        ("src.training.train_tokenizer", "Tokenizer训练模块"),
        
        # 推理模块
        ("src.inference.infer", "推理模块"),
        ("src.inference.chat", "多轮对话模块"),
        
        # RLHF 模块
        ("src.models.reward_model", "Reward Model模块"),
        ("src.training.ppo_trainer", "PPO训练器模块"),
        ("src.data.rm_dataset", "RM数据集模块"),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for module_path, description in tests:
        result = test_import(module_path, description)
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:  # None = skipped
            skipped += 1
    
    print("\n" + "="*70)
    print(f"  测试结果: {passed} 通过, {failed} 失败, {skipped} 跳过（缺少依赖）")
    print("="*70 + "\n")
    
    if failed == 0:
        print("🎉 所有导入路径正确！")
        if skipped > 0:
            print(f"   ({skipped} 个测试因缺少依赖被跳过，这是正常的)")
        print("\n💡 提示: 安装依赖后可运行完整测试")
        print("   pip install -r requirements.txt\n")
        return 0
    else:
        print("⚠️  部分导入测试失败，请检查上述错误。\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
