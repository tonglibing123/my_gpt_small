#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一验证脚本
整合所有验证功能，用于检查项目完整性

使用方法:
    python tools/verify_all.py
    python tools/verify_all.py --check paths
    python tools/verify_all.py --check tokenizer
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def verify_paths():
    """验证路径配置"""
    print_header("验证路径配置")
    
    checks = []
    
    # 检查关键目录
    dirs = ["src", "tests", "tools", "configs", "docs", "ckpt", "minimind_dataset"]
    for d in dirs:
        exists = os.path.exists(d)
        checks.append(("目录", d, exists))
        print(f"{'✓' if exists else '✗'} 目录: {d}")
    
    # 检查关键文件
    files = [
        "src/training/train_model.py",
        "src/training/train_tokenizer.py",
        "src/models/model.py",
        "configs/config.yaml",
        "configs/deepspeed_zero2.json"
    ]
    for f in files:
        exists = os.path.exists(f)
        checks.append(("文件", f, exists))
        print(f"{'✓' if exists else '✗'} 文件: {f}")
    
    # 检查数据文件
    data_files = [
        "minimind_dataset/pretrain_hq.jsonl",
        "minimind_dataset/hh_rlhf_cn/helpful_base_cn_train.jsonl"
    ]
    for f in data_files:
        exists = os.path.exists(f)
        checks.append(("数据", f, exists))
        print(f"{'✓' if exists else '✗'} 数据: {f}")
    
    passed = sum(1 for _, _, exists in checks if exists)
    total = len(checks)
    
    print(f"\n路径验证: {passed}/{total} 通过")
    return passed == total


def verify_tokenizer():
    """验证分词器相关代码"""
    print_header("验证分词器")
    
    checks = []
    
    # 检查分词器工具
    tokenizer_files = [
        "tools/tokenizer_utils.py",
        "tools/tokenizer_demo.py",
        "tools/tokenizer_analyzer.py",
        "tools/compare_tokenizers.py"
    ]
    
    for f in tokenizer_files:
        exists = os.path.exists(f)
        checks.append((f, exists))
        print(f"{'✓' if exists else '✗'} {f}")
    
    # 检查分词器文档
    docs = [
        "docs/BPE分词器原理详解.md",
        "docs/分词器参数调优指南.md"
    ]
    
    for d in docs:
        exists = os.path.exists(d)
        checks.append((d, exists))
        print(f"{'✓' if exists else '✗'} {d}")
    
    passed = sum(1 for _, exists in checks if exists)
    total = len(checks)
    
    print(f"\n分词器验证: {passed}/{total} 通过")
    return passed == total


def verify_pretrain():
    """验证预训练相关代码"""
    print_header("验证预训练")
    
    checks = []
    
    # 检查训练脚本
    train_files = [
        "src/training/train_model.py",
        "src/training/train_model_advanced.py",
        "src/training/train_tokenizer.py"
    ]
    
    for f in train_files:
        exists = os.path.exists(f)
        checks.append((f, exists))
        print(f"{'✓' if exists else '✗'} {f}")
    
    # 检查数据处理
    data_files = [
        "src/data/dataset.py",
        "src/data/pretokenize.py"
    ]
    
    for f in data_files:
        exists = os.path.exists(f)
        checks.append((f, exists))
        print(f"{'✓' if exists else '✗'} {f}")
    
    # 检查评估工具
    eval_files = [
        "tools/quick_eval.py",
        "tools/model_evaluator.py",
        "tools/benchmark.py"
    ]
    
    for f in eval_files:
        exists = os.path.exists(f)
        checks.append((f, exists))
        print(f"{'✓' if exists else '✗'} {f}")
    
    passed = sum(1 for _, exists in checks if exists)
    total = len(checks)
    
    print(f"\n预训练验证: {passed}/{total} 通过")
    return passed == total


def verify_tests():
    """验证测试代码"""
    print_header("验证测试")
    
    checks = []
    
    # 检查测试文件
    test_files = [
        "tests/test_model.py",
        "tests/test_training.py",
        "tests/test_tokenizer.py",
        "tests/test_inference.py",
        "tests/test_end_to_end.py"
    ]
    
    for f in test_files:
        exists = os.path.exists(f)
        checks.append((f, exists))
        print(f"{'✓' if exists else '✗'} {f}")
    
    passed = sum(1 for _, exists in checks if exists)
    total = len(checks)
    
    print(f"\n测试验证: {passed}/{total} 通过")
    return passed == total


def verify_imports():
    """验证关键模块可以导入"""
    print_header("验证模块导入")
    
    checks = []
    
    # 测试导入
    modules = [
        ("tools.utils", "load_data_module"),
        ("tools.utils", "get_model_components"),
        ("tools.tokenizer_utils", "load_tokenizer"),
        ("tools.config_loader", "load_config")
    ]
    
    for module_name, func_name in modules:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            checks.append((f"{module_name}.{func_name}", True))
            print(f"✓ {module_name}.{func_name}")
        except Exception as e:
            checks.append((f"{module_name}.{func_name}", False))
            print(f"✗ {module_name}.{func_name}: {e}")
    
    passed = sum(1 for _, exists in checks if exists)
    total = len(checks)
    
    print(f"\n导入验证: {passed}/{total} 通过")
    return passed == total


def verify_all():
    """运行所有验证"""
    print_header("MiniMind 项目验证")
    
    results = {
        "路径配置": verify_paths(),
        "分词器": verify_tokenizer(),
        "预训练": verify_pretrain(),
        "测试": verify_tests(),
        "模块导入": verify_imports()
    }
    
    print_header("验证总结")
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✅ 所有验证通过！")
    else:
        print("  ⚠️ 部分验证失败，请检查上述输出")
    print("="*60)
    
    return 0 if all_passed else 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MiniMind 项目验证工具")
    parser.add_argument("--check", type=str, 
                       choices=["all", "paths", "tokenizer", "pretrain", "tests", "imports"],
                       default="all",
                       help="选择验证项目")
    
    args = parser.parse_args()
    
    if args.check == "all":
        return verify_all()
    elif args.check == "paths":
        return 0 if verify_paths() else 1
    elif args.check == "tokenizer":
        return 0 if verify_tokenizer() else 1
    elif args.check == "pretrain":
        return 0 if verify_pretrain() else 1
    elif args.check == "tests":
        return 0 if verify_tests() else 1
    elif args.check == "imports":
        return 0 if verify_imports() else 1


if __name__ == "__main__":
    sys.exit(main())
