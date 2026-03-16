# model_small.py - 简化版 MyGPT
"""
简化版 MyGPT：只保留核心特性
- RoPE 旋转位置编码
- GQA 分组查询注意力
- RMSNorm
- SwiGLU
- KV-Cache
- 权重共享

移除的高级特性：MoE、μP、NTK/YaRN、Logit Soft-Capping、Label Smoothing、Gradient Checkpointing
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ========== RMSNorm ==========
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 保存输入类型，在 BF16 下转为 float32 计算以保证精度
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight.float()
        return x.to(input_dtype)


# ========== RoPE 旋转位置编码 ==========
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device):
        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ========== SwiGLU ==========
class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(8 / 3 * n_embd)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w_gate(x)) * self.w1(x)))


# ========== GQA 注意力 ==========
class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_head % n_kv_head == 0, f"n_head ({n_head}) 必须能被 n_kv_head ({n_kv_head}) 整除"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        B, T, _ = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV-Cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv_cache = (k, v)
        
        # GQA: 扩展 KV 头
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # 使用 SDPA（PyTorch 2.0+）
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.bool() if mask is not None else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(mask is None), scale=self.scale
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.dropout(self.o_proj(y)), new_kv_cache


# ========== Transformer Block ==========
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = GroupedQueryAttention(n_embd, n_head, n_kv_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, dropout)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), cos, sin, mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


# ========== 主模型 ==========
class MyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = 6, n_head: int = 6,
                 n_kv_head: int = 2, n_embd: int = 384, block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size 必须为正整数，当前值: {vocab_size}")
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) 必须能被 n_head ({n_head}) 整除")
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.head_dim = n_embd // n_head
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=block_size * 2)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_kv_head, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享
        
        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self):
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.o_proj.weight, std=residual_std)
            torch.nn.init.normal_(block.mlp.w2.weight, std=residual_std)

    def forward(self, idx, targets=None, kv_cache=None, return_value=False):
        """
        前向传播
        
        Args:
            idx: 输入 token ids [B, T]
            targets: 目标 token ids [B, T]（用于计算 loss）
            kv_cache: KV 缓存（用于推理加速）
            return_value: 是否返回 value（需要 value_head 属性）
        
        Returns:
            如果 return_value=False: (logits, loss, new_caches)
            如果 return_value=True: (logits, values, new_caches)
        """
        B, T = idx.size()
        device = idx.device
        past_len = kv_cache[0][0].size(2) if kv_cache and kv_cache[0] else 0
        
        cos, sin = self.rotary_emb(past_len + T, device)
        cos, sin = cos[past_len:past_len + T], sin[past_len:past_len + T]
        
        x = self.drop(self.tok_emb(idx))
        
        # 因果掩码
        total_len = past_len + T
        mask = torch.tril(torch.ones(total_len, total_len, device=device))[-T:, :].unsqueeze(0).unsqueeze(0)
        
        new_caches = []
        kv_cache_list = list(kv_cache) if kv_cache else [None] * self.n_layer
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, mask, kv_cache_list[i])
            new_caches.append(cache)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # 如果需要返回 value（用于 RLHF）
        if return_value:
            if hasattr(self, 'value_head') and self.value_head is not None:
                values = self.value_head(x).squeeze(-1)
            else:
                # 如果没有 value_head，创建一个并警告
                import warnings
                warnings.warn("模型没有 value_head，将动态创建。建议在初始化时添加 value_head。")
                self.value_head = nn.Linear(self.n_embd, 1, bias=False).to(device)
                torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
                values = self.value_head(x).squeeze(-1)
            return logits, values, new_caches
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50,
                 top_p: float = None, pad_token_id: int = None, eos_token_id: int = None):
        """
        自回归生成
        
        Args:
            idx: 输入 token ids [B, T]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样（可选）
            pad_token_id: pad token id（用于兼容性）
            eos_token_id: eos token id，遇到时提前停止
        """
        if max_new_tokens <= 0:
            return idx
        
        kv_caches = [None] * self.n_layer
        temperature = max(temperature, 1e-7)
        
        for i in range(max_new_tokens):
            # 检查总序列长度是否超过 block_size
            if idx.size(1) > self.block_size:
                # 截断输入，保留最近的 block_size 个 token
                idx = idx[:, -self.block_size:]
                # 重置 KV cache，从截断后的序列重新开始
                kv_caches = [None] * self.n_layer
                # 重新计算完整序列的 KV cache
                logits, _, kv_caches = self(idx, kv_cache=kv_caches)
                logits = logits[:, -1, :] / temperature
            else:
                idx_cond = idx if i == 0 else idx[:, -1:]
                logits, _, kv_caches = self(idx_cond, kv_cache=kv_caches)
                logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-P (nucleus) 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            # 处理 NaN/Inf 情况（数值稳定性）
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs.sum(dim=-1) == 0).any():
                import warnings
                warnings.warn("检测到 NaN/Inf，使用均匀分布采样")
                probs = torch.ones_like(probs) / probs.size(-1)
            
            # 确保概率和为 1
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            try:
                idx_next = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                # 如果采样失败，使用 argmax
                import warnings
                warnings.warn(f"采样失败: {e}，使用贪婪解码")
                idx_next = probs.argmax(dim=-1, keepdim=True)
            
            idx = torch.cat([idx, idx_next], dim=1)
            
            # EOS 停止条件
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break
        
        return idx

    def get_num_params(self, non_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n
