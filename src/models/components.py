# common.py - 共用的基础组件
"""
抽取项目中共用的基础组件：Checkpoint管理等

注意：RMSNorm、RotaryEmbedding、SwiGLU 等模型组件定义在 model_small.py 中，
本文件只包含训练相关的工具组件。

已移除的功能（简化教学项目）：
- EMA: 大模型训练中不常用
- Cosine Restarts 调度器: 过于复杂，使用 DeepSpeed 内置调度器
- Per-Parameter 梯度裁剪: 使用 DeepSpeed 配置中的标准梯度裁剪
"""
import torch


# ========== Checkpoint 管理器 ==========
import os
import json
import shutil


class CheckpointManager:
    """管理checkpoint，保留最近N个和最佳M个"""
    
    def __init__(self, ckpt_dir: str = "ckpt", keep_recent: int = 3, keep_best: int = 2):
        if keep_recent < 1:
            raise ValueError("keep_recent 必须 >= 1")
        if keep_best < 1:
            raise ValueError("keep_best 必须 >= 1")
        self.ckpt_dir = ckpt_dir
        self.keep_recent = keep_recent
        self.keep_best = keep_best
        self.history_file = os.path.join(ckpt_dir, "history.json")
        self.history = self._load_history()
    
    def _load_history(self):
        """加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"checkpoints": [], "best_val_loss": float('inf'), "best_step": None}
    
    def _save_history(self):
        """保存历史记录"""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record(self, step: int, val_loss: float, ckpt_path: str):
        """记录checkpoint信息"""
        self.history["checkpoints"].append({
            "step": step,
            "val_loss": val_loss,
            "path": ckpt_path,
            "is_best": False
        })
        
        # 检查是否是最佳
        if val_loss < self.history["best_val_loss"]:
            self.history["best_val_loss"] = val_loss
            self.history["best_step"] = step
            self.history["checkpoints"][-1]["is_best"] = True
            # 复制到 best 目录
            best_dir = os.path.join(self.ckpt_dir, f"best_step_{step}")
            if os.path.exists(ckpt_path) and not os.path.exists(best_dir):
                shutil.copytree(ckpt_path, best_dir)
        
        self._save_history()
    
    def cleanup(self):
        """清理旧checkpoint，保留最近N个和最佳M个"""
        checkpoints = self.history["checkpoints"]
        if len(checkpoints) <= self.keep_recent:
            return
        
        # 按step排序，找出需要保留的最近N个
        sorted_by_step = sorted(checkpoints, key=lambda x: x["step"], reverse=True)
        recent_steps = {c["step"] for c in sorted_by_step[:self.keep_recent]}
        
        # 按val_loss排序，找出需要保留的最佳M个
        sorted_by_loss = sorted(checkpoints, key=lambda x: x["val_loss"])
        best_steps = {c["step"] for c in sorted_by_loss[:self.keep_best]}
        
        # 合并需要保留的
        keep_steps = recent_steps | best_steps
        
        # 删除不需要保留的
        to_remove = []
        for ckpt in checkpoints:
            if ckpt["step"] not in keep_steps:
                path = ckpt["path"]
                # 检查是否是 best 目录（跳过删除）
                path_basename = os.path.basename(path)
                if os.path.exists(path) and not path_basename.startswith("best_"):
                    try:
                        shutil.rmtree(path)
                    except Exception as e:
                        print(f"警告: 删除 {path} 失败: {e}")
                to_remove.append(ckpt)
        
        # 更新历史记录
        for ckpt in to_remove:
            self.history["checkpoints"].remove(ckpt)
        self._save_history()
    
    def get_best_checkpoint(self):
        """获取最佳checkpoint路径"""
        if self.history["best_step"] is not None:
            best_dir = os.path.join(self.ckpt_dir, f"best_step_{self.history['best_step']}")
            if os.path.exists(best_dir):
                return best_dir
        return None
    
    def get_summary(self):
        """获取checkpoint摘要"""
        return {
            "total_saved": len(self.history["checkpoints"]),
            "best_val_loss": self.history["best_val_loss"],
            "best_step": self.history["best_step"]
        }
