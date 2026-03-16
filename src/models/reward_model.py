# reward_model.py  (简化版)
import torch
import torch.nn as nn
import os, sys

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class RewardModel(nn.Module):
    def __init__(self, mygpt, dropout=0.1):
        """
        Args:
            mygpt: MyGPT 模型实例
            dropout: Dropout 比例
        """
        super().__init__()
        self.backbone = mygpt
        # 冻结 Embedding
        for p in self.backbone.tok_emb.parameters():
            p.requires_grad = False

        self.reward_head = nn.Sequential(
            nn.Linear(mygpt.n_embd, mygpt.n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mygpt.n_embd, 1, bias=False)
        )

    def forward(self, idx, attention_mask=None):
        """
        idx: [B, T]
        attention_mask: [B, T]  1=有效 0=pad
        return: reward [B]
        """
        b, t = idx.size()
        device = idx.device
        
        # 边界检查
        if t == 0:
            return torch.zeros(b, device=device)
        
        # 使用 RoPE 前向传播获取隐藏态
        x = self.backbone.drop(self.backbone.tok_emb(idx))
        
        # 获取 RoPE cos/sin
        cos, sin = self.backbone.rotary_emb(t, device)
        
        # Causal Mask
        mask = torch.tril(torch.ones(t, t, device=device)).unsqueeze(0).unsqueeze(0)
        
        # 通过所有 Block（注意：推理时应该禁用 dropout，但这由 model.eval() 控制）
        for block in self.backbone.blocks:
            x, _ = block(x, cos, sin, mask, None)
        hidden = self.backbone.ln_f(x)  # [B, T, n_embd]

        # 获取最后一个有效 token 的隐藏态
        if attention_mask is None:
            lengths = torch.full((b,), t, device=device, dtype=torch.long)
        else:
            lengths = attention_mask.sum(dim=1).long().clamp(min=1, max=t)
        batch_range = torch.arange(b, device=device)
        last_hidden = hidden[batch_range, lengths - 1]  # [B, n_embd]

        reward = self.reward_head(last_hidden).squeeze(-1)  # [B]
        return reward
