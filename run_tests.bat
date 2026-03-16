@echo off
REM ============================================
REM MiniMind 测试运行脚本 (Windows)
REM ============================================
REM 用途: 运行所有测试套件
REM 环境: Windows + Python 3.10+

echo ============================================
echo MiniMind 测试套件
echo ============================================
echo.

REM 检查 Python 版本
echo 检查 Python 版本...
python --version
echo.

REM 检查环境
echo 检查环境依赖...
python tools/check_env.py
if errorlevel 1 (
    echo ❌ 环境检查失败，请先安装依赖:
    echo    pip install -r requirements.txt
    exit /b 1
)
echo.

REM 运行导入测试
echo ============================================
echo 运行导入测试
echo ============================================
python tests/test_imports.py
if errorlevel 1 (
    echo ❌ 导入测试失败
    exit /b 1
)
echo.

REM 运行分词器测试
echo ============================================
echo 运行分词器测试
echo ============================================
python tests/test_tokenizer.py
if errorlevel 1 (
    echo ❌ 分词器测试失败
    exit /b 1
)
echo.

REM 运行数据预处理测试
echo ============================================
echo 运行数据预处理测试
echo ============================================
python tests/test_preprocessing.py
if errorlevel 1 (
    echo ❌ 数据预处理测试失败
    exit /b 1
)
echo.

REM 运行训练流程测试
echo ============================================
echo 运行训练流程测试
echo ============================================
python tests/test_training.py
if errorlevel 1 (
    echo ❌ 训练流程测试失败
    exit /b 1
)
echo.

REM 运行推理测试
echo ============================================
echo 运行推理测试
echo ============================================
python tests/test_inference.py
if errorlevel 1 (
    echo ❌ 推理测试失败
    exit /b 1
)
echo.

REM 运行端到端测试
echo ============================================
echo 运行端到端测试
echo ============================================
python tests/test_end_to_end.py
if errorlevel 1 (
    echo ❌ 端到端测试失败
    exit /b 1
)
echo.

REM 运行模型功能测试
echo ============================================
echo 运行模型功能测试
echo ============================================
python tests/test_model.py
if errorlevel 1 (
    echo ❌ 模型功能测试失败
    exit /b 1
)
echo.

REM 运行奖励模型测试
echo ============================================
echo 运行奖励模型测试
echo ============================================
python tests/test_reward_model.py
if errorlevel 1 (
    echo ❌ 奖励模型测试失败
    exit /b 1
)
echo.

REM 运行PPO测试
echo ============================================
echo 运行PPO测试
echo ============================================
python tests/test_ppo.py
if errorlevel 1 (
    echo ❌ PPO测试失败
    exit /b 1
)
echo.

REM 运行PPO数据集测试
echo ============================================
echo 运行PPO数据集测试
echo ============================================
python tests/test_ppo_dataset.py
if errorlevel 1 (
    echo ❌ PPO数据集测试失败
    exit /b 1
)
echo.

REM 运行完整功能测试
echo ============================================
echo 运行完整功能测试
echo ============================================
python tests/test_all_features.py
if errorlevel 1 (
    echo ❌ 完整功能测试失败
    exit /b 1
)
echo.

echo ============================================
echo 所有测试完成！
echo ============================================
