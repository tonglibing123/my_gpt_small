#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具：生成训练过程的可视化图表
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(log_dir="runs", output_dir="visualizations"):
    """绘制训练曲线"""
    print("=" * 60)
    print("📈 绘制训练曲线")
    print("=" * 60)
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("❌ 需要安装 tensorboard: pip install tensorboard")
        return
    
    # 查找 event 文件
    event_files = glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"❌ 在 {log_dir} 目录下未找到 TensorBoard 日志文件")
        return
    
    print(f"找到 {len(event_files)} 个日志文件")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for event_file in event_files:
        print(f"\n处理: {event_file}")
        
        # 加载事件
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # 获取所有标量
        tags = ea.Tags()['scalars']
        print(f"  可用指标: {tags}")
        
        # 1. 训练 Loss
        if 'train/loss' in tags:
            train_loss = ea.Scalars('train/loss')
            steps = [s.step for s in train_loss]
            values = [s.value for s in train_loss]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label='Train Loss', linewidth=2)
            
            # 添加验证 Loss
            if 'val/loss' in tags:
                val_loss = ea.Scalars('val/loss')
                val_steps = [s.step for s in val_loss]
                val_values = [s.value for s in val_loss]
                plt.plot(val_steps, val_values, label='Val Loss', linewidth=2, marker='o')
            
            plt.xlabel('训练步数')
            plt.ylabel('Loss')
            plt.title('训练和验证 Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, "training_loss.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ 保存: {output_path}")
            plt.close()
        
        # 2. 学习率
        if 'train/learning_rate' in tags:
            lr_data = ea.Scalars('train/learning_rate')
            steps = [s.step for s in lr_data]
            values = [s.value for s in lr_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, linewidth=2, color='orange')
            plt.xlabel('训练步数')
            plt.ylabel('学习率')
            plt.title('学习率变化')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            output_path = os.path.join(output_dir, "learning_rate.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ 保存: {output_path}")
            plt.close()
        
        # 3. 训练/验证对比
        if 'train/loss' in tags and 'val/loss' in tags:
            train_loss = ea.Scalars('train/loss')
            val_loss = ea.Scalars('val/loss')
            
            # 对齐步数
            val_dict = {s.step: s.value for s in val_loss}
            train_at_val = [(s.step, s.value, val_dict.get(s.step)) 
                           for s in train_loss if s.step in val_dict]
            
            if train_at_val:
                steps, train_vals, val_vals = zip(*train_at_val)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(train_vals, val_vals, alpha=0.6)
                
                # 添加对角线
                min_val = min(min(train_vals), min(val_vals))
                max_val = max(max(train_vals), max(val_vals))
                plt.plot([min_val, max_val], [min_val, max_val], 
                        'r--', label='完美拟合', linewidth=2)
                
                plt.xlabel('训练 Loss')
                plt.ylabel('验证 Loss')
                plt.title('训练 vs 验证 Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                output_path = os.path.join(output_dir, "train_val_comparison.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  ✅ 保存: {output_path}")
                plt.close()


def plot_checkpoint_comparison(ckpt_dir="ckpt/pretrain", output_dir="visualizations"):
    """比较不同 checkpoint 的性能"""
    print("\n" + "=" * 60)
    print("📊 Checkpoint 性能比较")
    print("=" * 60)
    
    history_file = os.path.join(ckpt_dir, "history.json")
    
    if not os.path.exists(history_file):
        print(f"❌ 未找到 history.json: {history_file}")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    checkpoints = history.get("checkpoints", [])
    
    if not checkpoints:
        print("❌ 没有 checkpoint 记录")
        return
    
    print(f"找到 {len(checkpoints)} 个 checkpoint")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    steps = [c['step'] for c in checkpoints]
    val_losses = [c['val_loss'] for c in checkpoints]
    is_best = [c.get('is_best', False) for c in checkpoints]
    
    # 绘制
    plt.figure(figsize=(12, 6))
    
    # 所有 checkpoint
    plt.plot(steps, val_losses, 'o-', label='验证 Loss', linewidth=2, markersize=8)
    
    # 标记最佳 checkpoint
    best_steps = [s for s, b in zip(steps, is_best) if b]
    best_losses = [l for l, b in zip(val_losses, is_best) if b]
    if best_steps:
        plt.scatter(best_steps, best_losses, color='red', s=200, 
                   marker='*', label='最佳 Checkpoint', zorder=5)
    
    plt.xlabel('训练步数')
    plt.ylabel('验证 Loss')
    plt.title('Checkpoint 性能对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "checkpoint_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {output_path}")
    plt.close()
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  - 最佳验证 Loss: {min(val_losses):.4f} (step {steps[val_losses.index(min(val_losses))]})") 
    print(f"  - 最终验证 Loss: {val_losses[-1]:.4f} (step {steps[-1]})")
    print(f"  - Loss 改进: {val_losses[0] - val_losses[-1]:.4f}")


def plot_evaluation_results(eval_file="evaluation_results.json", output_dir="visualizations"):
    """可视化评估结果"""
    print("\n" + "=" * 60)
    print("📊 评估结果可视化")
    print("=" * 60)
    
    if not os.path.exists(eval_file):
        print(f"❌ 评估结果文件不存在: {eval_file}")
        return
    
    with open(eval_file) as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建指标总览 (3x2 布局)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # 1. 困惑度
    ax = axes[0, 0]
    ppl = results.get('perplexity', 0)
    ax.bar(['困惑度'], [ppl], color='skyblue')
    ax.set_ylabel('Perplexity')
    ax.set_title('模型困惑度')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 准确率
    ax = axes[0, 1]
    acc = results.get('accuracy', 0) * 100
    ax.bar(['准确率'], [acc], color='lightgreen')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('预测准确率')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Loss
    ax = axes[1, 0]
    loss = results.get('loss', 0)
    ax.bar(['Loss'], [loss], color='salmon')
    ax.set_ylabel('Loss')
    ax.set_title('平均 Loss')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 生成样本长度
    ax = axes[1, 1]
    gen_samples = results.get('generation_samples', [])
    if gen_samples:
        lengths = [s['length'] for s in gen_samples]
        ax.hist(lengths, bins=10, color='plum', edgecolor='black')
        ax.set_xlabel('生成长度')
        ax.set_ylabel('样本数')
        ax.set_title('生成文本长度分布')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, '无生成样本', ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 5. Diversity Metrics (Distinct-n)
    ax = axes[2, 0]
    diversity = results.get('diversity_metrics', {})
    if diversity:
        metrics = ['Distinct-1', 'Distinct-2', 'Distinct-3']
        values = [
            diversity.get('distinct_1', 0),
            diversity.get('distinct_2', 0),
            diversity.get('distinct_3', 0)
        ]
        bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('分数')
        ax.set_title('生成多样性 (Distinct-n)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, '无多样性指标', ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 6. 词表使用统计
    ax = axes[2, 1]
    if diversity:
        total_tokens = diversity.get('total_tokens', 0)
        unique_tokens = diversity.get('unique_tokens', 0)
        vocab_usage = diversity.get('vocab_usage', 0)
        
        categories = ['总 Token 数', '唯一 Token 数']
        values = [total_tokens, unique_tokens]
        bars = ax.bar(categories, values, color=['#95E1D3', '#F38181'])
        ax.set_ylabel('数量')
        ax.set_title(f'词表使用 (使用率: {vocab_usage:.2%})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, '无词表统计', ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "evaluation_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {output_path}")
    plt.close()


def generate_html_report(output_dir="visualizations", output_file="training_report.html"):
    """生成 HTML 报告"""
    print("\n" + "=" * 60)
    print("📄 生成 HTML 报告")
    print("=" * 60)
    
    # 查找所有图片
    images = glob(os.path.join(output_dir, "*.png"))
    
    if not images:
        print(f"❌ 在 {output_dir} 目录下未找到图片")
        return
    
    # 生成 HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>训练报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #4CAF50;
            margin-top: 0;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🚀 大模型训练报告</h1>
    <p class="timestamp">生成时间: {timestamp}</p>
"""
    
    import datetime
    html = html.format(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 按类别组织图片
    categories = {
        "训练过程": ["training_loss", "learning_rate", "train_val_comparison"],
        "Checkpoint 分析": ["checkpoint_comparison"],
        "模型评估": ["evaluation_summary"],
        "数据分析": ["token_distribution", "sequence_length"]
    }
    
    for category, keywords in categories.items():
        category_images = [img for img in images 
                          if any(kw in os.path.basename(img) for kw in keywords)]
        
        if category_images:
            html += f'    <div class="section">\n'
            html += f'        <h2>{category}</h2>\n'
            for img in category_images:
                rel_path = os.path.relpath(img, os.path.dirname(output_file))
                img_name = os.path.basename(img).replace('_', ' ').replace('.png', '')
                html += f'        <h3>{img_name}</h3>\n'
                html += f'        <img src="{rel_path}" alt="{img_name}">\n'
            html += '    </div>\n'
    
    html += """
</body>
</html>
"""
    
    # 保存 HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 保存: {output_file}")
    print(f"\n在浏览器中打开: file://{os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description="训练可视化工具")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard 日志目录")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/pretrain", help="Checkpoint 目录")
    parser.add_argument("--eval_file", type=str, default="evaluation_results.json", help="评估结果文件")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="输出目录")
    args = parser.parse_args()
    
    print("\n" + "📊 训练可视化工具".center(60, "="))
    print()
    
    # 1. 训练曲线
    plot_training_curves(args.log_dir, args.output_dir)
    
    # 2. Checkpoint 比较
    plot_checkpoint_comparison(args.ckpt_dir, args.output_dir)
    
    # 3. 评估结果
    if os.path.exists(args.eval_file):
        plot_evaluation_results(args.eval_file, args.output_dir)
    
    # 4. 生成 HTML 报告
    generate_html_report(args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print(f"\n查看生成的文件:")
    print(f"  - {args.output_dir}/")
    print(f"  - training_report.html")
    print()


if __name__ == "__main__":
    main()
