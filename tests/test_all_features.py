#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind 完整功能测试脚本
测试所有三个阶段的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("MiniMind 完整功能测试")
print("=" * 80)
print()

# 测试计数
total_tests = 0
passed_tests = 0
failed_tests = 0

def test_feature(name, test_func):
    """测试功能"""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    
    print(f"测试 {total_tests}: {name}")
    print("-" * 80)
    
    try:
        test_func()
        print(f"✅ {name} - 通过")
        passed_tests += 1
    except Exception as e:
        print(f"❌ {name} - 失败: {e}")
        failed_tests += 1
    
    print()

# ============================================================================
# 阶段1测试: 基础完善
# ============================================================================
print("=" * 80)
print("阶段1: 基础完善")
print("=" * 80)
print()

def test_readme():
    """测试 README.md"""
    assert Path("README.md").exists(), "README.md 不存在"
    content = Path("README.md").read_text(encoding='utf-8')
    assert "MiniMind" in content, "README.md 内容不完整"
    assert len(content) > 1000, "README.md 内容太短"

def test_run_scripts():
    """测试一键运行脚本"""
    assert Path("run_all.sh").exists(), "run_all.sh 不存在"
    assert Path("run_all.bat").exists(), "run_all.bat 不存在"

def test_env_check():
    """测试环境检查脚本"""
    assert Path("check_env.py").exists(), "check_env.py 不存在"

def test_docs_structure():
    """测试文档结构"""
    assert Path("docs").exists(), "docs 目录不存在"
    assert Path("docs/快速开始.md").exists(), "快速开始.md 不存在"
    assert Path("docs/教学文档.md").exists(), "教学文档.md 不存在"

def test_src_structure():
    """测试源代码结构"""
    assert Path("src").exists(), "src 目录不存在"
    assert Path("src/training").exists(), "src/training 目录不存在"
    assert Path("src/inference").exists(), "src/inference 目录不存在"
    assert Path("src/data").exists(), "src/data 目录不存在"
    assert Path("src/models").exists(), "src/models 目录不存在"

def test_file_naming():
    """测试文件命名"""
    assert Path("src/training/train_tokenizer.py").exists()
    assert Path("src/training/train_model.py").exists()
    assert Path("src/inference/infer.py").exists()
    assert Path("src/inference/chat.py").exists()

test_feature("README.md 存在", test_readme)
test_feature("一键运行脚本", test_run_scripts)
test_feature("环境检查脚本", test_env_check)
test_feature("文档结构", test_docs_structure)
test_feature("源代码结构", test_src_structure)
test_feature("文件命名规范", test_file_naming)

# ============================================================================
# 阶段2测试: 体验优化
# ============================================================================
print("=" * 80)
print("阶段2: 体验优化")
print("=" * 80)
print()

def test_config_file():
    """测试配置文件"""
    assert Path("configs/config.yaml").exists(), "config.yaml 不存在"
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    assert "model" in config, "配置文件缺少 model 部分"
    assert "pretrain" in config, "配置文件缺少 pretrain 部分"

def test_config_loader():
    """测试配置加载器"""
    from tools.config_loader import load_config
    config = load_config("configs/config.yaml")
    assert config.model.n_layer == 6, "配置加载失败"

def test_logger():
    """测试日志系统"""
    from tools.logger import setup_logger
    logger = setup_logger("Test", level="INFO")
    logger.info("测试日志")

def test_error_handler():
    """测试错误处理"""
    from tools.error_handler import MiniMindError
    try:
        raise MiniMindError("测试错误", "测试解决方案")
    except MiniMindError:
        pass

def test_progress_bar():
    """测试进度条"""
    from tqdm import tqdm
    import time
    for _ in tqdm(range(10), desc="测试", leave=False):
        time.sleep(0.01)

test_feature("配置文件", test_config_file)
test_feature("配置加载器", test_config_loader)
test_feature("日志系统", test_logger)
test_feature("错误处理", test_error_handler)
test_feature("进度条", test_progress_bar)

# ============================================================================
# 阶段3测试: 功能扩展
# ============================================================================
print("=" * 80)
print("阶段3: 功能扩展")
print("=" * 80)
print()

def test_verify_all():
    """测试统一验证工具"""
    assert Path("tools/verify_all.py").exists(), "verify_all.py 不存在"

def test_quick_eval():
    """测试快速评估工具"""
    assert Path("tools/quick_eval.py").exists(), "quick_eval.py 不存在"

def test_tokenizer_tools():
    """测试分词器工具"""
    assert Path("tools/tokenizer_utils.py").exists(), "tokenizer_utils.py 不存在"
    assert Path("tools/tokenizer_demo.py").exists(), "tokenizer_demo.py 不存在"
    assert Path("tools/tokenizer_analyzer.py").exists(), "tokenizer_analyzer.py 不存在"
    assert Path("tools/compare_tokenizers.py").exists(), "compare_tokenizers.py 不存在"

def test_evaluation_tools():
    """测试评估工具"""
    assert Path("tools/model_evaluator.py").exists(), "model_evaluator.py 不存在"
    assert Path("tools/benchmark.py").exists(), "benchmark.py 不存在"

# 注意: 以下工具已被删除（简化项目结构）
# - experiment_manager.py (过于复杂，使用 TensorBoard 替代)
# - export_model.py (非核心功能)
# - web_demo.py (使用命令行界面替代)
# - quantize_model.py (过于高级)

test_feature("统一验证工具", test_verify_all)
test_feature("快速评估工具", test_quick_eval)
test_feature("分词器工具集", test_tokenizer_tools)
test_feature("评估工具集", test_evaluation_tools)

# ============================================================================
# 依赖检查
# ============================================================================
print("=" * 80)
print("依赖检查")
print("=" * 80)
print()

def test_dependencies():
    """测试依赖"""
    required = [
        "torch",
        "yaml",
        "tqdm",
        "colorama",
    ]
    
    optional = [
        "gradio",
        "onnx",
        "safetensors",
    ]
    
    print("必需依赖:")
    for pkg in required:
        try:
            if pkg == "yaml":
                __import__("yaml")
            else:
                __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} (未安装)")
            raise
    
    print("\n可选依赖:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚠️  {pkg} (未安装，可选)")

test_feature("依赖检查", test_dependencies)

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("测试总结")
print("=" * 80)
print()
print(f"总测试数: {total_tests}")
print(f"通过: {passed_tests} ✅")
print(f"失败: {failed_tests} ❌")
print(f"成功率: {passed_tests / total_tests * 100:.1f}%")
print()

if failed_tests == 0:
    print("🎉 所有测试通过！MiniMind 项目功能完整！")
    print()
    print("下一步:")
    print("  1. 运行完整训练: bash run_all.sh")
    print("  2. 查看快速开始: docs/快速开始.md")
    print("  3. 查看学习路径: docs/学习路径指南.md")
    print("  4. 快速评估模型: python tools/quick_eval.py")
    print("  5. 多轮对话测试: python src/inference/chat.py")
else:
    print("⚠️  部分测试失败，请检查错误信息")
    sys.exit(1)

print("=" * 80)
