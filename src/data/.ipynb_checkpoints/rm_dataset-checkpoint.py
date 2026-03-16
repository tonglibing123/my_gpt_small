# rm_data.py - Reward Model 数据集
import torch, json, numpy as np, os
from typing import List, Tuple, Any


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[str], tokenizer: Any, block_size: int = 256,
                 sample_ratio: float = 1.0, seed: int = 42):
        # 先适配分词器，再验证
        tokenizer = self._adapt_fast_tokenizer(tokenizer)
        self._validate_tokenizer(tokenizer)
        
        if block_size <= 0 or not files:
            raise ValueError("block_size 必须为正整数且 files 不能为空")
        self.block_size = block_size
        self.samples: List[Tuple[torch.Tensor, ...]] = []

        all_samples, total_lines, skipped = [], 0, 0
        for fn in files:
            if not os.path.exists(fn):
                raise FileNotFoundError(fn)
            with open(fn, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        skipped += 1
                        continue
                    try:
                        obj = json.loads(line)
                        if "chosen" not in obj or "rejected" not in obj:
                            raise ValueError("缺少 chosen/rejected")
                        
                        # 拼接上下文
                        ctx = "\n".join(f"{t.get('role','')}: {t.get('text','')}" for t in obj.get("context", []))
                        ctx = ctx + "\n" if ctx else ""
                        
                        chosen = obj['chosen'].get('text', '') if isinstance(obj['chosen'], dict) else str(obj['chosen'])
                        rejected = obj['rejected'].get('text', '') if isinstance(obj['rejected'], dict) else str(obj['rejected'])
                        
                        c = tokenizer.encode_plus(ctx + f"assistant: {chosen}", truncation=True,
                                                  max_length=block_size, padding='max_length', return_tensors='np')
                        r = tokenizer.encode_plus(ctx + f"assistant: {rejected}", truncation=True,
                                                  max_length=block_size, padding='max_length', return_tensors='np')
                        all_samples.append((
                            torch.from_numpy(c['input_ids'][0].astype(np.int64)),
                            torch.from_numpy(c['attention_mask'][0].astype(np.int64)),
                            torch.from_numpy(r['input_ids'][0].astype(np.int64)),
                            torch.from_numpy(r['attention_mask'][0].astype(np.int64))))
                    except Exception as e:
                        skipped += 1
                        print(f"警告: {fn}:{line_num} 跳过: {e}")

        if not all_samples:
            raise ValueError(f"没有成功加载任何样本（共 {total_lines} 行，跳过 {skipped} 行）")
        print(f"成功加载 {len(all_samples)} 个样本")
        
        if not (0 < sample_ratio <= 1.0):
            raise ValueError("sample_ratio 必须在 (0, 1]")
        subset_size = max(1, int(len(all_samples) * sample_ratio))
        indices = np.random.default_rng(seed).choice(len(all_samples), size=subset_size, replace=False)
        indices.sort()
        self.samples = [all_samples[i] for i in indices]

    def _adapt_fast_tokenizer(self, tokenizer):
        """
        适配 PreTrainedTokenizerFast：兼容两种 encode 返回格式（列表/Encoding对象）
        """
        # 仅对没有 encode_plus 的 Fast 分词器进行适配
        if hasattr(tokenizer, 'encode_plus'):
            return tokenizer
        
        # 检查是否是 Fast 分词器（有 encode 方法但无 encode_plus）
        if not hasattr(tokenizer, 'encode'):
            raise ValueError("分词器必须支持 encode 或 encode_plus 方法")
        
        # 封装 encode_plus 方法，兼容 numpy 输出
        def encode_plus_wrapper(
            text,
            truncation=True,
            max_length=None,
            padding='max_length',
            return_tensors=None,
            **kwargs
        ):
            """
            兼容两种 encode 返回格式：
            1. 返回列表（直接是 input_ids）
            2. 返回 Encoding 对象（有 .ids/.attention_mask）
            """
            # ========== 关键修复：分两步获取 input_ids 和 attention_mask ==========
            # 第一步：获取 input_ids（处理截断/长度限制）
            input_ids = tokenizer.encode(
                text,
                truncation=truncation,
                max_length=max_length,
                padding=False,  # 先不padding，手动处理
                **kwargs
            )
            
            # 第二步：单独获取 attention_mask
            # 如果是列表，说明返回的是 input_ids，手动生成 attention_mask
            if isinstance(input_ids, list):
                # 处理 padding（补 pad_token_id 到 max_length）
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                if padding == 'max_length' and max_length is not None:
                    # 截断过长的序列
                    if len(input_ids) > max_length:
                        input_ids = input_ids[:max_length]
                    # 补 pad 到 max_length
                    attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
                    input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
                else:
                    attention_mask = [1] * len(input_ids)
            else:
                # 如果是 Encoding 对象，按原有逻辑处理
                input_ids = input_ids.ids
                attention_mask = input_ids.attention_mask
            
            # 转换为 numpy 格式（匹配你的代码调用逻辑）
            input_ids_np = np.array([input_ids], dtype=np.int64)
            attention_mask_np = np.array([attention_mask], dtype=np.int64)
            
            # 构造返回结果
            result = {
                'input_ids': input_ids_np,
                'attention_mask': attention_mask_np
            }
            
            # 处理 return_tensors 参数
            if return_tensors == 'pt':
                result['input_ids'] = torch.from_numpy(result['input_ids'])
                result['attention_mask'] = torch.from_numpy(result['attention_mask'])
            elif return_tensors is None:
                result['input_ids'] = result['input_ids'][0]
                result['attention_mask'] = result['attention_mask'][0]
            
            return result
        
        # 将封装的方法绑定到分词器对象
        tokenizer.encode_plus = encode_plus_wrapper
        return tokenizer

    def _validate_tokenizer(self, tokenizer):
        if not hasattr(tokenizer, 'encode_plus'):
            raise ValueError("分词器必须支持 encode_plus")
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            raise ValueError("分词器必须设置 pad_token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]