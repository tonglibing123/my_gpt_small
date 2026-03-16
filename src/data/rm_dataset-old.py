# rm_data.py - Reward Model 数据集
import torch, json, numpy as np, os
from typing import List, Tuple, Any


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[str], tokenizer: Any, block_size: int = 256,
                 sample_ratio: float = 1.0, seed: int = 42):
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

    def _validate_tokenizer(self, tokenizer):
        if not hasattr(tokenizer, 'encode_plus'):
            raise ValueError("分词器必须支持 encode_plus")
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            raise ValueError("分词器必须设置 pad_token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
