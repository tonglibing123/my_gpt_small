"""MMap 数据集：从 .bin 文件加载预处理的 token 序列"""
import os, mmap, numpy as np, torch, warnings


class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, bin_files, block_size=256):
        if not bin_files:
            raise ValueError("bin_files 不能为空")
        if block_size <= 0:
            raise ValueError(f"block_size 必须为正整数，当前值: {block_size}")
        self.block_size = block_size
        self.data, self.lengths = [], []
        self._file_handles, self._mmaps = [], []
        self._closed = False
        
        for f in bin_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"文件不存在: {f}")
            if os.path.getsize(f) == 0:
                warnings.warn(f"文件为空，跳过: {f}")
                continue
            fd = os.open(f, os.O_RDONLY)
            self._file_handles.append(fd)
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmaps.append(mm)
            arr = np.frombuffer(mm, dtype=np.uint16)
            self.data.append(arr)
            self.lengths.append((len(arr) - 1) // block_size)
        
        if not self.data:
            raise ValueError("所有数据文件都为空或不存在")
        self.cumlen = np.cumsum([0] + self.lengths)
        if self.cumlen[-1] == 0:
            raise ValueError("数据文件中没有足够的 token")
    
    def close(self):
        if self._closed:
            return
        self._closed = True
        for mm in self._mmaps:
            try: mm.close()
            except: pass
        for fd in self._file_handles:
            try: os.close(fd)
            except: pass
        self._mmaps.clear()
        self._file_handles.clear()
    
    def __del__(self):
        self.close()

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, idx):
        if self._closed:
            raise RuntimeError("数据集已关闭")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围")
        file_idx = np.searchsorted(self.cumlen, idx + 1) - 1
        local_idx = idx - self.cumlen[file_idx]
        start = local_idx * self.block_size
        tokens = torch.tensor(self.data[file_idx][start:start + self.block_size + 1], dtype=torch.long)
        return tokens[:-1], tokens[1:]
