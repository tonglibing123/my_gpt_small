@echo off
REM MiniMind 一键运行脚本 (Windows)
REM 自动执行完整的训练流程

setlocal enabledelayedexpansion

REM 设置颜色（Windows 10+）
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

echo.
echo ======================================================================
echo   MiniMind 一键运行脚本
echo ======================================================================
echo.

REM 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo %RED%错误: 未找到 Python，请先安装 Python 3.8+%NC%
    pause
    exit /b 1
)

echo %BLUE%使用 Python: python%NC%

REM 步骤0: 环境检查
echo.
echo ======================================================================
echo   步骤0: 环境检查
echo ======================================================================
echo.

python tools\check_env.py
if %errorlevel% neq 0 (
    echo %RED%环境检查失败，请先解决环境问题%NC%
    pause
    exit /b 1
)

echo %GREEN%环境检查通过%NC%

REM 询问运行模式
echo.
echo %BLUE%请选择运行模式:%NC%
echo   1) 快速模式 (1000 steps, ~30分钟)
echo   2) 完整模式 (5000 steps, ~2-4小时)
echo   3) 仅推理 (使用已有模型)
set /p mode="请输入选项 [1-3]: "

if "%mode%"=="1" (
    set TOTAL_STEPS=1000
    set MODE_NAME=快速模式
) else if "%mode%"=="2" (
    set TOTAL_STEPS=5000
    set MODE_NAME=完整模式
) else if "%mode%"=="3" (
    echo %BLUE%跳过训练，直接进入推理%NC%
    goto inference
) else (
    echo %YELLOW%无效选项，使用快速模式%NC%
    set TOTAL_STEPS=1000
    set MODE_NAME=快速模式
)

echo %BLUE%运行模式: %MODE_NAME% (%TOTAL_STEPS% steps)%NC%

REM 步骤1: 训练 Tokenizer
echo.
echo ======================================================================
echo   步骤1: 训练 Tokenizer
echo ======================================================================
echo.

if exist "my_tokenizer" (
    echo %YELLOW%检测到已有 tokenizer，跳过训练%NC%
) else (
    echo %BLUE%开始训练 Tokenizer...%NC%
    python src\training\train_tokenizer.py
    if %errorlevel% neq 0 (
        echo %RED%Tokenizer 训练失败%NC%
        pause
        exit /b 1
    )
    echo %GREEN%Tokenizer 训练完成%NC%
)

REM 步骤2: 数据预处理
echo.
echo ======================================================================
echo   步骤2: 数据预处理
echo ======================================================================
echo.

if exist "data\train.bin" (
    echo %YELLOW%检测到已有预处理数据，跳过预处理%NC%
) else (
    echo %BLUE%开始数据预处理...%NC%
    python src\data\pretokenize.py
    if %errorlevel% neq 0 (
        echo %RED%数据预处理失败%NC%
        pause
        exit /b 1
    )
    echo %GREEN%数据预处理完成%NC%
)

REM 步骤3: 数据探索（可选）
echo.
echo ======================================================================
echo   步骤3: 数据探索 (可选)
echo ======================================================================
echo.

set /p run_analyze="是否运行分词器分析? [y/N]: "
if /i "%run_analyze%"=="y" (
    echo %BLUE%运行分词器分析...%NC%
    python tools\tokenizer_analyzer.py
    echo %GREEN%分词器分析完成%NC%
) else (
    echo %BLUE%跳过分词器分析%NC%
)

REM 步骤4: 模型训练
echo.
echo ======================================================================
echo   步骤4: 模型训练
echo ======================================================================
echo.

echo %BLUE%开始训练模型 (%TOTAL_STEPS% steps)...%NC%
echo %YELLOW%这可能需要较长时间，请耐心等待...%NC%

where deepspeed >nul 2>nul
if %errorlevel% equ 0 (
    deepspeed --num_gpus 1 src\training\train_model.py --total_steps %TOTAL_STEPS% --batch_size 16 --save_steps 500 --log_interval 10
) else (
    echo %YELLOW%未找到 deepspeed 命令，使用 Python 直接运行...%NC%
    python src\training\train_model.py --total_steps %TOTAL_STEPS% --batch_size 16 --save_steps 500 --log_interval 10
)

if %errorlevel% neq 0 (
    echo %RED%模型训练失败%NC%
    pause
    exit /b 1
)

echo %GREEN%模型训练完成%NC%

REM 步骤5: 模型评估（可选）
echo.
echo ======================================================================
echo   步骤5: 模型评估 (可选)
echo ======================================================================
echo.

set /p run_eval="是否运行模型评估? [y/N]: "
if /i "%run_eval%"=="y" (
    echo %BLUE%运行模型评估...%NC%
    python tools\model_evaluator.py --checkpoint ckpt\pretrain\final --max_samples 1000
    echo %GREEN%模型评估完成%NC%
) else (
    echo %BLUE%跳过模型评估%NC%
)

REM 步骤6: 可视化（可选）
echo.
echo ======================================================================
echo   步骤6: 训练可视化 (可选)
echo ======================================================================
echo.

set /p run_viz="是否生成可视化报告? [y/N]: "
if /i "%run_viz%"=="y" (
    echo %BLUE%生成可视化报告...%NC%
    python tools\visualizer.py
    echo %GREEN%可视化报告已生成: training_report.html%NC%
) else (
    echo %BLUE%跳过可视化%NC%
)

REM 步骤7: 推理测试
:inference
echo.
echo ======================================================================
echo   步骤7: 推理测试
echo ======================================================================
echo.

if not exist "ckpt\pretrain\final" (
    echo %RED%未找到训练好的模型%NC%
    goto end
)

echo %BLUE%运行推理测试...%NC%
python src\inference\infer.py --checkpoint ckpt\pretrain\final --prompt "人工智能在未来" --max_new_tokens 50 --temperature 0.8

if %errorlevel% neq 0 (
    echo %RED%推理测试失败%NC%
    pause
    exit /b 1
)

echo %GREEN%推理测试完成%NC%

echo.
set /p run_chat="是否启动多轮对话? [y/N]: "
if /i "%run_chat%"=="y" (
    echo %BLUE%启动多轮对话...%NC%
    python src\inference\chat.py --checkpoint ckpt\pretrain\final --max_new_tokens 50
)

REM 完成
:end
echo.
echo ======================================================================
echo   🎉 完整流程运行完成！
echo ======================================================================
echo.

echo %GREEN%所有步骤已完成%NC%
echo.
echo %BLUE%下一步建议:%NC%
echo   1. 查看训练报告: start training_report.html
echo   2. 启动多轮对话: python src\inference\chat.py --checkpoint ckpt\pretrain\final
echo   3. 查看 TensorBoard: tensorboard --logdir=runs
echo.

pause
exit /b 0
