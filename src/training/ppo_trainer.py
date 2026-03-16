# ppo_trainer.py - PPO 训练器
"""
PPO (Proximal Policy Optimization) 训练器

核心功能:
1. collect(): 收集训练数据（生成响应、计算奖励）
2. ppo_epoch(): PPO更新（策略梯度、价值函数、熵正则化）
3. update_reference(): 更新参考策略

详细说明请参考: docs/PPO算法详解.md
"""
import torch, torch.nn.functional as F, logging, os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


class PPOTrainer:
    """
    PPO训练器
    
    参数说明:
        policy: 策略模型（要训练的模型）
        ref_policy: 参考策略（用于计算KL散度）
        reward_model: 奖励模型（评估生成质量）
        value_model: 价值模型（预测未来奖励，通常与policy共享）
        tokenizer: 分词器
        pad_token_id: padding token的ID
        clip_eps: 策略裁剪参数（默认0.2）
        beta: KL散度惩罚系数（默认0.01）
        eps_v: 价值函数裁剪参数（默认0.2）
    """
    def __init__(self, policy, ref_policy, reward_model, value_model, tokenizer,
                 pad_token_id=None, clip_eps=0.2, beta=0.01, gradient_accumulation_steps=32,
                 eps_v=0.2, share_policy_value=True):
        self.policy_engine = policy
        self.value_engine = value_model
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.clip_eps, self.beta, self.eps_v = clip_eps, beta, eps_v
        self.policy = policy.module if hasattr(policy, 'module') else policy
        self.value_model = value_model.module if hasattr(value_model, 'module') else value_model

    @torch.no_grad()
    def collect(self, prompts, max_new_tokens=128):
        """
        收集训练数据（Rollout阶段）
        
        步骤:
        1. 使用当前策略生成响应
        2. 计算每个token的对数概率和价值
        3. 使用Reward Model评估生成质量
        4. 添加KL散度惩罚（防止偏离参考策略太远）
        
        参数:
            prompts: 输入提示 [B, L_prompt]
            max_new_tokens: 最大生成长度
            
        返回:
            rollouts字典:
                - sequences: 完整序列 [B, L_prompt + L_gen]
                - logprobs: 对数概率 [B, L_gen]
                - values: 价值预测 [B, L_gen]
                - rewards: 奖励 [B, L_gen]
        """
        B = prompts.shape[0]
        if B == 0:
            raise ValueError("prompts batch size 不能为 0")

        # 1. 生成响应
        seq = self.policy.generate(
            prompts, 
            max_new_tokens=max_new_tokens, 
            temperature=1.0,  # 使用采样，保持多样性
            pad_token_id=self.pad_token_id
        )
        
        # 2. 计算对数概率和价值
        logits, values, _ = self.policy(seq, return_value=True)
        
        # 计算每个token的对数概率: log P(token | context)
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1)[:, :-1],  # [B, L-1, V]
            2,
            seq[:, 1:].unsqueeze(2)  # [B, L-1, 1]
        ).squeeze(2)  # [B, L-1]
        
        values = values[:, :-1]  # 对齐维度

        # 3. 计算奖励（使用Reward Model）
        rewards = torch.zeros_like(logprobs)
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        
        for i in range(B):
            # 找到EOS位置（生成结束位置）
            if eos_id:
                eos_idx = (seq[i] == eos_id).nonzero(as_tuple=True)[0]
                gen_len = eos_idx[0].item() if len(eos_idx) > 0 else seq.size(1)
            else:
                gen_len = seq.size(1)
            
            # 序列太短时跳过，rewards 保持为 0
            if gen_len < 2:
                logging.debug(f"样本 {i} 序列长度 {gen_len} < 2，跳过奖励计算")
                continue
            
            # 使用Reward Model评分
            seq_slice = seq[i:i+1, :gen_len]
            r = self.reward_model(seq_slice, (seq_slice != self.pad_token_id).long())
            
            # 将奖励放在生成结束位置
            reward_idx = min(gen_len - 2, rewards.size(1) - 1)
            if reward_idx >= 0:
                rewards[i, reward_idx] = r.item()

        # 4. 添加KL散度惩罚
        # KL(π_new || π_ref) ≈ log π_new - log π_ref
        ref_logits = self.ref_policy(seq)[0]
        ref_logprobs = torch.gather(
            F.log_softmax(ref_logits, dim=-1)[:, :-1], 
            2, 
            seq[:, 1:].unsqueeze(2)
        ).squeeze(2)
        
        # 修正奖励: r' = r - β * KL
        # 这样可以防止策略偏离参考策略太远，保持原始能力
        rewards = rewards - self.beta * (logprobs - ref_logprobs)

        return {
            "sequences": seq, 
            "logprobs": logprobs.detach(), 
            "values": values.detach(), 
            "rewards": rewards.detach()
        }

    def ppo_epoch(self, rollouts, mini_batch=4, gamma=0.99, lam=0.95, 
                  value_loss_coef=0.1, entropy_coef=0.01):
        """
        一轮PPO更新
        
        步骤:
        1. 计算优势函数（GAE）
        2. 标准化优势（提升稳定性）
        3. Mini-batch更新:
           a. Policy Loss (Clipped Objective)
           b. Value Loss (Clipped)
           c. Entropy (鼓励探索)
        
        参数:
            rollouts: collect()返回的数据
            mini_batch: mini-batch大小
            gamma: 折扣因子（默认0.99）
            lam: GAE lambda（默认0.95）
            value_loss_coef: 价值损失系数（默认0.1）
            entropy_coef: 熵正则化系数（默认0.01）
            
        返回:
            统计信息字典:
                - pg_loss: 策略梯度损失
                - vf_loss: 价值函数损失
                - entropy: 熵
                - kl: KL散度
                - clipfrac: 裁剪比例
        """
        seq, old_logp, values, rewards = rollouts["sequences"], rollouts["logprobs"], rollouts["values"], rollouts["rewards"]
        B, T_1 = old_logp.shape
        if B == 0 or T_1 == 0:
            return {}

        # 创建mask（忽略padding）
        mask = (seq[:, 1:] != self.pad_token_id).float()
        if mask.shape[1] > T_1:
            mask = mask[:, :T_1]
        elif mask.shape[1] < T_1:
            mask = F.pad(mask, (0, T_1 - mask.shape[1]), value=1.0)

        # ========== 1. 计算优势函数（GAE） ==========
        # GAE (Generalized Advantage Estimation) 平衡偏差和方差
        # A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        # 其中 δ_t = r_t + γV_{t+1} - V_t
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(T_1)):
            # 下一步的价值
            next_v = values[:, t + 1] if t < T_1 - 1 else torch.zeros(B, device=rewards.device)
            
            # TD误差: δ_t = r_t + γV_{t+1} - V_t
            delta = rewards[:, t] + gamma * next_v - values[:, t]
            
            # GAE递推: A_t = δ_t + γλA_{t+1}
            gae = delta + gamma * lam * gae * mask[:, t]
            advantages[:, t] = gae
        
        # 计算回报: R_t = A_t + V_t
        returns = advantages + values

        # 统计信息
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        num_updates = 0

        # ========== 2. Mini-batch更新 ==========
        perm = torch.randperm(B)
        for start in range(0, B, mini_batch):
            mb_idx = perm[start:min(start + mini_batch, B)]
            mb_ids, mb_mask = seq[mb_idx], mask[mb_idx]
            mb_old_logp, mb_returns = old_logp[mb_idx], returns[mb_idx]
            mb_advantages, mb_old_values = advantages[mb_idx].detach(), values[mb_idx].detach()

            # 前向传播
            logits, new_values, _ = self.policy_engine(mb_ids, return_value=True)
            new_logp = torch.gather(
                F.log_softmax(logits, dim=-1)[:, :-1], 
                2, 
                mb_ids[:, 1:].unsqueeze(2)
            ).squeeze(2)

            # 维度对齐
            min_len = min(new_logp.shape[1], mb_mask.shape[1])
            new_logp, mb_mask = new_logp[:, :min_len], mb_mask[:, :min_len]
            mb_old_logp, mb_returns = mb_old_logp[:, :min_len], mb_returns[:, :min_len]
            mb_advantages, mb_old_values = mb_advantages[:, :min_len], mb_old_values[:, :min_len]
            
            # 对齐 logits
            logits_aligned = logits[:, :-1, :]
            if logits_aligned.shape[1] > min_len:
                logits_aligned = logits_aligned[:, :min_len, :]

            # ========== a. Policy Loss (Clipped Objective) ==========
            # 计算概率比率: r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
            ratio = torch.exp(new_logp - mb_old_logp.detach())
            
            # 两个目标函数:
            # surr1 = r_t * A_t (正常更新)
            # surr2 = clip(r_t, 1-ε, 1+ε) * A_t (裁剪更新)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
            
            # 取最小值（保守更新）
            pg_loss = (-torch.min(surr1, surr2) * mb_mask).sum() / (mb_mask.sum() + 1e-8)

            # ========== b. Value Loss (Clipped) ==========
            V_pred = new_values[:, :-1]
            if V_pred.shape[1] > min_len:
                V_pred = V_pred[:, :min_len]
            elif V_pred.shape[1] < min_len:
                # 截断其他张量以匹配 V_pred
                actual_len = V_pred.shape[1]
                mb_mask = mb_mask[:, :actual_len]
                mb_returns = mb_returns[:, :actual_len]
                mb_old_values = mb_old_values[:, :actual_len]
            
            # 裁剪价值函数更新
            V_clip = mb_old_values + torch.clamp(
                V_pred - mb_old_values, 
                -self.eps_v, 
                self.eps_v
            )
            
            # 取最大值（保守更新）
            vf_loss = (0.5 * torch.max(
                (V_pred - mb_returns).pow(2), 
                (V_clip - mb_returns).pow(2)
            ) * mb_mask).sum() / (mb_mask.sum() + 1e-8)

            # ========== c. 熵正则化（鼓励探索） ==========
            # Entropy = -Σ p(a) * log(p(a))
            # 高熵 = 更随机 = 更多探索
            probs = F.softmax(logits_aligned, dim=-1)
            log_probs = F.log_softmax(logits_aligned, dim=-1)
            entropy = -(probs * log_probs).sum(-1)  # [B, T]
            entropy = (entropy * mb_mask).sum() / (mb_mask.sum() + 1e-8)

            # ========== 监控指标 ==========
            # KL散度（用于监控策略变化）
            kl = (mb_old_logp - new_logp).detach()
            kl = (kl * mb_mask).sum() / (mb_mask.sum() + 1e-8)

            # Clip fraction（用于监控裁剪比例）
            clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float()
            clipfrac = (clipfrac * mb_mask).sum() / (mb_mask.sum() + 1e-8)

            # ========== 总损失 ==========
            # Loss = Policy Loss + c1 * Value Loss - c2 * Entropy
            loss = pg_loss + value_loss_coef * vf_loss - entropy_coef * entropy
            
            # 反向传播
            if hasattr(self.policy_engine, 'backward'):
                self.policy_engine.backward(loss)
                self.policy_engine.step()
            else:
                loss.backward()

            # 累积统计
            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
            total_clipfrac += clipfrac.item()
            num_updates += 1

        # 返回统计信息
        return {
            'pg_loss': total_pg_loss / num_updates,
            'vf_loss': total_vf_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl': total_kl / num_updates,
            'clipfrac': total_clipfrac / num_updates,
            'total_loss': (total_pg_loss + value_loss_coef * total_vf_loss - entropy_coef * total_entropy) / num_updates
        }

    @torch.no_grad()
    def update_reference(self):
        """
        更新参考策略
        
        将当前策略的参数复制到参考策略，用于计算KL散度。
        定期更新可以让策略逐步演化，同时保持稳定性。
        
        建议: 每3-5个训练步骤更新一次
        """
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()
        logging.info("[Ref] Reference model updated")
