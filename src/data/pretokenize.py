#!/usr/bin/env python3
"""
预处理脚本：将 pretrain_hq.jsonl 转换为 .bin 格式的 token 序列
- Phase1: 多进程 tokenize → 临时 shard
- Phase2: 合并 + 全局 shuffle → train.bin
"""
import os, numpy as np, multiprocessing as mp, signal, shutil, json
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_JSON = "my_tokenizer/tokenizer.json"
DATA_FILE = "minimind_dataset/pretrain_hq.jsonl"
BLOCK_SIZE = 256
OUT_DIR = "data"
N_WORKER = 8  # RTX 4090 优化：使用 8 个工作进程
SHARD_SIZE = 10_000_000  # 每个 shard 的 token 数量

if not os.path.exists(TOKENIZER_JSON):
    raise FileNotFoundError(f"Tokenizer 文件不存在: {TOKENIZER_JSON}\n请先运行 src/training/train_tokenizer.py")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}\n请确保已下载 pretrain_hq.jsonl 到 minimind_dataset/ 目录")

tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
eos_id = tokenizer.token_to_id("</s>")
if eos_id is None:
    raise RuntimeError("请在 tokenizer 特殊 token 里加入 </s>")

os.makedirs(OUT_DIR, exist_ok=True)
temp_dir = os.path.join(OUT_DIR, "temp")
os.makedirs(temp_dir, exist_ok=True)


def load_all_texts():
    """加载所有文本数据到内存"""
    texts = []
    print(f">>> 正在加载数据: {DATA_FILE}")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj and obj["text"]:
                    texts.append(obj["text"])
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    print(f">>> 成功加载 {len(texts)} 条文本")
    return texts


def _write_shard(args):
    """单个 worker 处理一个 shard"""
    worker_id, shard_id, texts, start_idx, end_idx = args
    fname = os.path.join(temp_dir, f"shard_{shard_id:05d}.bin")
    if os.path.exists(fname):
        return
    
    fout = None
    try:
        fout = open(fname, "wb")
        tokens_written = 0
        
        for idx in range(start_idx, end_idx):
            text = texts[idx]
            ids = tokenizer.encode(text).ids + [eos_id]
            
            # 按 BLOCK_SIZE 切分
            for i in range(0, len(ids), BLOCK_SIZE):
                chunk = ids[i:i+BLOCK_SIZE]
                if len(chunk) == BLOCK_SIZE:
                    fout.write(np.array(chunk, dtype=np.uint16).tobytes())
                    tokens_written += BLOCK_SIZE
        
        print(f"[worker {worker_id}] shard {shard_id} 完成, {tokens_written} tokens")
    except Exception as e:
        print(f"[worker {worker_id}] shard {shard_id} 错误: {e}")
        if fout:
            fout.close()
            fout = None
        if os.path.exists(fname):
            os.remove(fname)
        raise
    finally:
        if fout:
            fout.close()


def phase1_tokenize():
    print(">>> Phase1: tokenize + shard")
    
    # 加载所有文本
    all_texts = load_all_texts()
    if not all_texts:
        raise RuntimeError("没有加载到任何文本数据")
    
    # 计算每个 shard 处理的文本数量
    n_texts = len(all_texts)
    texts_per_shard = max(1, n_texts // (N_WORKER * 4))  # 每个 worker 处理多个 shard
    n_shards = (n_texts + texts_per_shard - 1) // texts_per_shard
    
    print(f">>> 总文本数: {n_texts}, 每个 shard: {texts_per_shard} 条文本, 总 shard 数: {n_shards}")
    
    # 准备任务
    tasks = []
    for shard_id in range(n_shards):
        start_idx = shard_id * texts_per_shard
        end_idx = min(start_idx + texts_per_shard, n_texts)
        worker_id = shard_id % N_WORKER
        tasks.append((worker_id, shard_id, all_texts, start_idx, end_idx))
    
    # 多进程处理
    pool = mp.Pool(N_WORKER)
    for _ in tqdm(pool.imap_unordered(_write_shard, tasks), total=len(tasks), ncols=80):
        pass
    pool.close()
    pool.join()


def phase2_merge_shuffle():
    print(">>> Phase2: merge & shuffle")
    files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.endswith(".bin")]
    if not files:
        raise RuntimeError("临时目录中没有找到 .bin 文件")
    
    blocks_per_file = [os.path.getsize(f) // 2 // BLOCK_SIZE for f in files]
    n_blocks = sum(blocks_per_file)
    if n_blocks == 0:
        raise RuntimeError("数据量不足")
    
    print(f">>> 总共 {n_blocks} 个完整 block (来自 {len(files)} 个文件)")
    mmaps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
    out = np.memmap(os.path.join(OUT_DIR, "train.bin"), dtype=np.uint16, mode="w+", shape=(n_blocks * BLOCK_SIZE,))

    block_indices = [(fid, blk) for fid, n_blk in enumerate(blocks_per_file) for blk in range(n_blk)]
    np.random.shuffle(block_indices)

    offset = 0
    for file_id, local_blk in tqdm(block_indices, ncols=80, desc="合并"):
        local_start = local_blk * BLOCK_SIZE
        out[offset:offset+BLOCK_SIZE] = mmaps[file_id][local_start:local_start+BLOCK_SIZE]
        offset += BLOCK_SIZE

    out.flush()
    del mmaps, out
    shutil.rmtree(temp_dir)
    print(f">>> 已生成 {OUT_DIR}/train.bin")
    print(f">>> 总 tokens: {n_blocks * BLOCK_SIZE:,}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: exit(0))
    print("=" * 60)
    print("数据预处理: pretrain_hq.jsonl → train.bin")
    print("=" * 60)
    phase1_tokenize()
    phase2_merge_shuffle()
    print("=" * 60)
    print("预处理完成！")
    print("=" * 60)
