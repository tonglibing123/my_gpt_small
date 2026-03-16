#!/usr/bin/env python3
"""
项目一致性检查工具
检查所有配置文件、脚本和代码的参数一致性、路径引用等
"""
import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ConsistencyChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def add_error(self, msg):
        self.errors.append(f"❌ 错误: {msg}")
        
    def add_warning(self, msg):
        self.warnings.append(f"⚠️  警告: {msg}")
        
    def add_info(self, msg):
        self.info.append(f"ℹ️  信息: {msg}")
    
    def check_config_yaml(self):
        """检查 config.yaml 配置一致性"""
        print("\n" + "="*60)
        print("检查 configs/config.yaml")
        print("="*60)
        
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            self.add_error(f"配置文件不存在: {config_path}")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查模型参数一致性
        model_cfg = config.get('model', {})
        tokenizer_cfg = config.get('tokenizer', {})
        
        if model_cfg.get('vocab_size') != tokenizer_cfg.get('vocab_size'):
            self.add_error(
                f"模型和分词器的 vocab_size 不一致: "
                f"model={model_cfg.get('vocab_size')}, "
                f"tokenizer={tokenizer_cfg.get('vocab_size')}"
            )
        
        # 检查路径存在性
        paths_to_check = [
            ('data.dataset_path', config.get('data', {}).get('dataset_path')),
            ('tokenizer.save_path', config.get('tokenizer', {}).get('save_path')),
        ]
        
        for name, path in paths_to_check:
            if path and not os.path.exists(path):
                self.add_warning(f"{name} 路径不存在: {path} (需要先训练/下载)")
        
        # 检查 DeepSpeed 配置文件
        ds_configs = [
            config.get('pretrain', {}).get('deepspeed_config'),
            config.get('reward_model', {}).get('deepspeed_config'),
            config.get('ppo', {}).get('deepspeed_config'),
        ]
        
        for ds_config in ds_configs:
            if ds_config and not os.path.exists(ds_config):
                self.add_error(f"DeepSpeed 配置文件不存在: {ds_config}")
        
        print("✓ config.yaml 检查完成")
        return config
    
    def check_deepspeed_configs(self, config):
        """检查 DeepSpeed 配置一致性"""
        print("\n" + "="*60)
        print("检查 DeepSpeed 配置")
        print("="*60)
        
        # 预训练配置
        ds_pretrain = "configs/deepspeed_zero2.json"
        if os.path.exists(ds_pretrain):
            with open(ds_pretrain, 'r') as f:
                ds_cfg = json.load(f)
            
            yaml_cfg = config.get('pretrain', {})
            
            # 检查学习率
            ds_lr = ds_cfg.get('optimizer', {}).get('params', {}).get('lr')
            yaml_lr = yaml_cfg.get('learning_rate')
            if ds_lr != yaml_lr:
                self.add_warning(
                    f"预训练学习率不一致: "
                    f"deepspeed={ds_lr}, config.yaml={yaml_lr}"
                )
            
            # 检查 warmup_steps
            ds_warmup = ds_cfg.get('scheduler', {}).get('params', {}).get('warmup_num_steps')
            yaml_warmup = yaml_cfg.get('warmup_steps')
            if ds_warmup != yaml_warmup:
                self.add_warning(
                    f"预训练 warmup_steps 不一致: "
                    f"deepspeed={ds_warmup}, config.yaml={yaml_warmup}"
                )
            
            # 检查 total_steps
            ds_total = ds_cfg.get('scheduler', {}).get('params', {}).get('total_num_steps')
            yaml_total = yaml_cfg.get('total_steps')
            if ds_total != yaml_total:
                self.add_warning(
                    f"预训练 total_steps 不一致: "
                    f"deepspeed={ds_total}, config.yaml={yaml_total}"
                )
        
        # Reward Model 配置
        ds_rm = "configs/deepspeed_zero2_rm.json"
        if os.path.exists(ds_rm):
            with open(ds_rm, 'r') as f:
                ds_cfg = json.load(f)
            
            yaml_cfg = config.get('reward_model', {})
            
            ds_lr = ds_cfg.get('optimizer', {}).get('params', {}).get('lr')
            yaml_lr = yaml_cfg.get('learning_rate')
            if ds_lr != yaml_lr:
                self.add_warning(
                    f"奖励模型学习率不一致: "
                    f"deepspeed={ds_lr}, config.yaml={yaml_lr}"
                )
        
        # PPO 配置
        ds_ppo = "configs/deepspeed_zero2_ppo.json"
        if os.path.exists(ds_ppo):
            with open(ds_ppo, 'r') as f:
                ds_cfg = json.load(f)
            
            yaml_cfg = config.get('ppo', {})
            
            ds_lr = ds_cfg.get('optimizer', {}).get('params', {}).get('lr')
            yaml_lr = yaml_cfg.get('learning_rate')
            if ds_lr != yaml_lr:
                self.add_warning(
                    f"PPO 学习率不一致: "
                    f"deepspeed={ds_lr}, config.yaml={yaml_lr}"
                )
        
        print("✓ DeepSpeed 配置检查完成")
    
    def check_training_scripts(self):
        """检查训练脚本的默认参数"""
        print("\n" + "="*60)
        print("检查训练脚本默认参数")
        print("="*60)
        
        scripts = {
            'train_model.py': 'src/training/train_model.py',
            'train_tokenizer.py': 'src/training/train_tokenizer.py',
            'train_reward_model.py': 'src/training/train_reward_model.py',
            'train_ppo.py': 'src/training/train_ppo.py',
        }
        
        params = {}
        
        for name, path in scripts.items():
            if not os.path.exists(path):
                self.add_error(f"训练脚本不存在: {path}")
                continue
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取默认参数
            script_params = {}
            
            # vocab_size
            match = re.search(r'--vocab_size.*default=(\d+)', content)
            if match:
                script_params['vocab_size'] = int(match.group(1))
            
            # n_layer
            match = re.search(r'--n_layer.*default=(\d+)', content)
            if match:
                script_params['n_layer'] = int(match.group(1))
            
            # n_head
            match = re.search(r'--n_head.*default=(\d+)', content)
            if match:
                script_params['n_head'] = int(match.group(1))
            
            # n_embd
            match = re.search(r'--n_embd.*default=(\d+)', content)
            if match:
                script_params['n_embd'] = int(match.group(1))
            
            # block_size
            match = re.search(r'--block_size.*default=(\d+)', content)
            if match:
                script_params['block_size'] = int(match.group(1))
            
            # batch_size
            match = re.search(r'--batch_size.*default=(\d+)', content)
            if match:
                script_params['batch_size'] = int(match.group(1))
            
            params[name] = script_params
        
        # 检查模型参数一致性
        model_params = ['vocab_size', 'n_layer', 'n_head', 'n_embd', 'block_size']
        reference = params.get('train_model.py', {})
        
        for param in model_params:
            ref_value = reference.get(param)
            if ref_value is None:
                continue
            
            for script_name, script_params in params.items():
                if script_name == 'train_tokenizer.py':
                    continue  # 分词器脚本参数不同
                
                value = script_params.get(param)
                if value is not None and value != ref_value:
                    self.add_error(
                        f"模型参数 {param} 不一致: "
                        f"train_model.py={ref_value}, {script_name}={value}"
                    )
        
        print("✓ 训练脚本参数检查完成")
        return params
    
    def check_shell_scripts(self):
        """检查 shell 脚本"""
        print("\n" + "="*60)
        print("检查 Shell 脚本")
        print("="*60)
        
        scripts = [
            'run_all.sh',
            'run_all.bat',
            'run_4090.sh',
            'run_tests.sh',
            'run_tests.bat',
        ]
        
        for script in scripts:
            if not os.path.exists(script):
                self.add_warning(f"脚本不存在: {script}")
                continue
            
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查引用的文件是否存在
            # Python 脚本引用
            py_files = re.findall(r'python[3]?\s+([^\s]+\.py)', content)
            for py_file in py_files:
                # 清理路径（移除 Windows 路径分隔符）
                py_file = py_file.replace('\\', '/')
                if not os.path.exists(py_file):
                    self.add_error(f"{script} 引用的文件不存在: {py_file}")
            
            # 检查目录引用
            dirs = re.findall(r'(?:ckpt|data|my_tokenizer|minimind_dataset)/[^\s"\']+', content)
            for dir_path in dirs:
                # 只检查根目录是否存在
                root_dir = dir_path.split('/')[0]
                if root_dir not in ['ckpt', 'data', 'my_tokenizer']:  # 这些是训练后才有的
                    if not os.path.exists(root_dir):
                        self.add_warning(f"{script} 引用的目录不存在: {root_dir}")
        
        print("✓ Shell 脚本检查完成")
    
    def check_data_paths(self):
        """检查数据路径一致性"""
        print("\n" + "="*60)
        print("检查数据路径")
        print("="*60)
        
        # 预训练数据
        pretrain_data = "minimind_dataset/pretrain_hq.jsonl"
        if not os.path.exists(pretrain_data):
            self.add_warning(f"预训练数据不存在: {pretrain_data} (需要下载)")
        
        # RLHF 数据
        rlhf_train = "minimind_dataset/hh_rlhf_cn/helpful_base_cn_train.jsonl"
        rlhf_test = "minimind_dataset/hh_rlhf_cn/helpful_base_cn_test.jsonl"
        
        if not os.path.exists(rlhf_train):
            self.add_warning(f"RLHF 训练数据不存在: {rlhf_train} (需要下载)")
        
        if not os.path.exists(rlhf_test):
            self.add_warning(f"RLHF 测试数据不存在: {rlhf_test} (需要下载)")
        
        print("✓ 数据路径检查完成")
    
    def check_checkpoint_paths(self):
        """检查 checkpoint 路径引用一致性"""
        print("\n" + "="*60)
        print("检查 Checkpoint 路径")
        print("="*60)
        
        # 检查 train_ppo.py 中的路径
        ppo_script = "src/training/train_ppo.py"
        if os.path.exists(ppo_script):
            with open(ppo_script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查奖励模型路径
            rm_match = re.search(r'--rm_ckpt.*default="([^"]+)"', content)
            if rm_match:
                rm_path = rm_match.group(1)
                expected_rm_path = "ckpt/rm/pytorch_model.bin"
                if rm_path != expected_rm_path:
                    self.add_warning(
                        f"PPO 脚本中奖励模型路径: {rm_path}, "
                        f"建议使用: {expected_rm_path}"
                    )
            
            # 检查 SFT 模型路径
            sft_match = re.search(r'--sft_ckpt.*default="([^"]+)"', content)
            if sft_match:
                sft_path = sft_match.group(1)
                expected_sft_path = "ckpt/pretrain/final/pytorch_model.bin"
                if sft_path != expected_sft_path:
                    self.add_warning(
                        f"PPO 脚本中 SFT 模型路径: {sft_path}, "
                        f"建议使用: {expected_sft_path}"
                    )
        
        print("✓ Checkpoint 路径检查完成")
    
    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "="*60)
        print("检查摘要")
        print("="*60)
        
        if self.errors:
            print(f"\n发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.info:
            print(f"\n信息:")
            for info in self.info:
                print(f"  {info}")
        
        if not self.errors and not self.warnings:
            print("\n✅ 所有检查通过，未发现问题！")
        elif not self.errors:
            print(f"\n✅ 未发现错误，但有 {len(self.warnings)} 个警告需要注意")
        else:
            print(f"\n❌ 发现 {len(self.errors)} 个错误需要修复")
        
        print("="*60)
        
        return len(self.errors) == 0


def main():
    print("="*60)
    print("项目一致性检查工具")
    print("="*60)
    
    checker = ConsistencyChecker()
    
    # 执行所有检查
    config = checker.check_config_yaml()
    if config:
        checker.check_deepspeed_configs(config)
    checker.check_training_scripts()
    checker.check_shell_scripts()
    checker.check_data_paths()
    checker.check_checkpoint_paths()
    
    # 打印摘要
    success = checker.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
