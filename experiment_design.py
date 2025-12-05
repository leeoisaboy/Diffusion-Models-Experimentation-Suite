"""
扩散模型实验设计
包含超参数实验、采样算法对比、引导生成实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from diffusion_utilities import ContextUnet, CustomDataset, ddpm_add_noise, sample_ddpm, sample_ddim
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 简单的SGD优化器实现
class SimpleSGD:
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        # 添加param_groups以兼容PyTorch标准接口
        self.param_groups = [{'params': self.params, 'lr': self.lr}]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        # 确保使用最新的学习率
        self.lr = self.param_groups[0]['lr']
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

class ExperimentManager:
    def __init__(self, data_path='sprites_1788_16x16.npy',
                 label_path='sprite_labels_nc_1788_16x16.npy'):
        """初始化实验管理器"""
        self.data_path = data_path
        self.label_path = label_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载数据
        self.load_data()

        # 创建结果目录
        self.results_dir = Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)

    def load_data(self):
        """加载数据集"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 直接使用文件路径创建数据集
        self.dataset = CustomDataset(self.data_path, self.label_path, transform, null_context=False)

        # 获取加载的数据用于其他用途
        self.sprites = self.dataset.sprites
        self.sprite_labels = self.dataset.slabels

    def create_model(self, n_feat=64, n_cfeat=5, height=16):
        """创建模型"""
        return ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(self.device)

    def hyperparameter_experiment(self, configs):
        """
        超参数实验
        configs: 超参数配置列表
        """
        results = {}

        # 尝试加载已有的实验结果
        results_file = self.results_dir / "hyperparameter_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"加载了 {len(results)} 个已有的实验结果")
            except Exception as e:
                print(f"加载已有结果失败: {e}")

        for i, config in enumerate(configs):
            exp_key = f"exp_{i}"

            # 如果这个实验已经完成，跳过
            if exp_key in results:
                print(f"跳过已完成的实验 {i+1}/{len(configs)}: {config}")
                continue

            print(f"运行实验 {i+1}/{len(configs)}: {config}")

            try:
                # 创建模型和优化器
                model = self.create_model(n_feat=config['n_feat'])

                # 根据配置选择优化器
                optimizer_type = config.get('optimizer', 'adam')
                if optimizer_type == 'sgd':
                    optimizer = SimpleSGD(model.parameters(), lr=config['lr'])
                else:  # adam
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

                # 训练
                train_losses = self.train_model(
                    model, optimizer,
                    batch_size=config['batch_size'],
                    n_epoch=config['n_epoch']
                )

                # 采样测试
                samples, sample_time = self.sample_ddpm(model, timesteps=config['timesteps'])

                # 保存结果到内存
                results[exp_key] = {
                    'config': config,
                    'train_losses': train_losses,
                    'sample_time': sample_time,
                    'final_loss': train_losses[-1] if train_losses else None
                }

                # 实时保存到JSON文件
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"实验 {i+1} 结果已实时保存")

                # 保存模型和样本
                torch.save(model.state_dict(), self.results_dir / f"model_exp_{i}.pth")
                self.save_samples(samples, f"samples_exp_{i}")

            except Exception as e:
                print(f"实验 {i+1} 失败: {e}")
                # 保存失败信息
                results[exp_key] = {
                    'config': config,
                    'error': str(e),
                    'status': 'failed'
                }
                # 仍然保存失败信息到JSON
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

        print("所有实验完成")
        return results

    def sampling_algorithm_comparison(self, model_paths):
        """
        DDPM vs DDIM采样算法对比
        """
        results = {}

        for model_name, model_path in model_paths.items():
            print(f"测试模型: {model_name}")

            # 加载模型
            model = self.create_model()
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # DDPM采样
            ddpm_samples, ddpm_time = self.sample_ddpm(model, timesteps=500)

            # DDIM采样
            ddim_samples, ddim_time = self.sample_ddim(model, timesteps=25)

            # 计算质量指标
            ddpm_metrics = self.calculate_metrics(ddpm_samples)
            ddim_metrics = self.calculate_metrics(ddim_samples)

            results[model_name] = {
                'ddpm': {
                    'time': ddpm_time,
                    'metrics': ddpm_metrics
                },
                'ddim': {
                    'time': ddim_time,
                    'metrics': ddim_metrics
                }
            }

            # 保存样本
            self.save_samples(ddpm_samples, f"{model_name}_ddpm")
            self.save_samples(ddim_samples, f"{model_name}_ddim")

        # 保存结果
        with open(self.results_dir / "sampling_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def guided_generation_experiment(self, model_path, context_configs):
        """
        引导生成实验
        """
        results = {}

        # 加载模型
        model = self.create_model()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        for config_name, context_config in context_configs.items():
            print(f"测试引导配置: {config_name}")

            # 生成样本
            samples = self.sample_guided(model, context_config, timesteps=25)

            # 保存结果
            results[config_name] = {
                'context_config': context_config
            }

            # 保存样本
            self.save_samples(samples, f"guided_{config_name}")

        # 保存结果
        with open(self.results_dir / "guided_generation.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def train_model(self, model, optimizer, batch_size: int = 100, n_epoch: int = 10):
        """训练模型"""
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        model.train()
        losses: List[float] = []

        for ep in range(n_epoch):
            print(f'epoch {ep}')

            # GPU内存监控
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f'  GPU内存使用: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB')

            # 线性学习率衰减
            optimizer.param_groups[0]['lr'] = 1e-3 * (1 - ep / n_epoch)

            for x, c in dataloader:
                x = x.to(self.device)
                c = c.to(self.device)

                # 上下文掩码
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(self.device)
                c = c * context_mask.unsqueeze(-1)

                # 扩散过程
                t = torch.randint(0, 500, (x.shape[0],)).to(self.device)  # timesteps=500
                x_noisy, noise = ddpm_add_noise(x, t)

                # 预测噪声
                pred_noise = model(x_noisy, t / 500, c)  # timesteps=500

                # 计算损失
                loss = F.mse_loss(pred_noise, noise)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

        return losses

    def sample_ddpm(self, model, timesteps=500):
        """DDPM采样"""
        start_time = time.time()
        samples, _ = sample_ddpm(model, 32, (3, 16, 16), self.device, timesteps=timesteps)
        sample_time = time.time() - start_time
        return samples, sample_time

    def sample_ddim(self, model, timesteps=25, context=None):
        """DDIM采样"""
        start_time = time.time()
        samples = sample_ddim(model, 32, (3, 16, 16), self.device, n=timesteps, context=context, timesteps=500, eta=0.0)
        sample_time = time.time() - start_time
        return samples, sample_time

    def sample_guided(self, model, context_config: List[float], timesteps: int = 25):
        """引导采样

        Args:
            model: 扩散模型
            context_config: 上下文配置，例如 [1, 0, 0, 0, 0] 表示 human
            timesteps: DDIM采样步数
        """
        # 创建上下文向量 - 确保数据类型和设备正确
        context = torch.tensor([context_config] * 32, dtype=torch.float32).to(self.device)

        print(f"  Context shape: {context.shape}, Context: {context_config}")

        # 使用DDIM采样，eta=0.0确保确定性采样
        samples = sample_ddim(model, 32, (3, 16, 16), self.device, n=timesteps, context=context, timesteps=500, eta=0.0)
        return samples

    def calculate_metrics(self, samples):
        """计算图像质量指标"""
        # 这里可以添加FID、IS等指标的计算
        # 暂时返回基础统计信息
        return {
            'mean': float(samples.mean().item()),
            'std': float(samples.std().item()),
            'min': float(samples.min().item()),
            'max': float(samples.max().item())
        }

    def save_samples(self, samples, name: str):
        """保存样本图像"""
        samples_dir = self.results_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        # 转换为numpy并保存
        samples_np = samples.cpu().numpy()
        np.save(samples_dir / f"{name}.npy", samples_np)

        # 保存可视化图像
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                # 关键修复：使用固定的归一化范围 [-1, 1] -> [0, 1]
                # 模型输出在[-1,1]范围内，不应该对每个图像独立归一化
                img = np.clip((img + 1) / 2, 0, 1)  # 从[-1,1]映射到[0,1]
                ax.imshow(img)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(samples_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        plt.close()

def run_all_experiments():
    """运行所有实验"""
    exp_manager = ExperimentManager()

    # # 1. 超参数实验
    # print("=== 开始超参数实验 ===")
    # print("=== 检测到CUDA设备，使用GPU优化配置 ===")

    # # GPU优化配置 - 充分利用RTX 4060 8GB显存
    # hyper_configs = [
    #     # 基础配置 - 增加批大小充分利用GPU
    #     {'n_feat': 128, 'lr': 1e-3, 'batch_size': 128, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},

    #     # 模型大小探索 - 测试更大模型
    #     {'n_feat': 64, 'lr': 1e-3, 'batch_size': 256, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},
    #     {'n_feat': 256, 'lr': 1e-3, 'batch_size': 64, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},
    #     {'n_feat': 512, 'lr': 1e-3, 'batch_size': 32, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},

    #     # 学习率探索
    #     {'n_feat': 128, 'lr': 5e-4, 'batch_size': 128, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},
    #     {'n_feat': 128, 'lr': 2e-3, 'batch_size': 128, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},

    #     # 批大小极限测试
    #     {'n_feat': 128, 'lr': 1e-3, 'batch_size': 256, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'adam'},

    #     # 优化器对比
    #     {'n_feat': 128, 'lr': 1e-3, 'batch_size': 128, 'n_epoch': 30, 'timesteps': 500, 'optimizer': 'sgd'},

    #     # 时间步数探索
        
    #     {'n_feat': 128, 'lr': 1e-3, 'batch_size': 128, 'n_epoch': 30, 'timesteps': 250, 'optimizer': 'adam'},
    # ]
    # hyper_results = exp_manager.hyperparameter_experiment(hyper_configs)

    #2. 采样算法对比
    print("=== 开始采样算法对比实验 ===")
    model_paths = {
        'baseline': 'weights/context_model_31.pth',
        'com_mod_1': 'weights/context_model_trained.pth',
        'com_mod_2': 'weights/model_trained.pth'
    }
    sampling_results = exp_manager.sampling_algorithm_comparison(model_paths)

    #3. 引导生成实验
    # print("=== 开始引导生成实验 ===")
    # context_configs = {
    #     'human': [1, 0, 0, 0, 0],      # 人类
    #     'food': [0, 0, 1, 0, 0],       # 食物
    #     'spell': [0, 0, 0, 1, 0],      # 法术
    #     'mixed_human_food': [1, 0, 1, 0, 0],  # 人类+食物
    #     'mixed_spell_side': [0, 0, 1, 1, 1],  # 法术+侧脸
    # }
    # guided_results = exp_manager.guided_generation_experiment(
    #     'weights/context_model_31.pth', context_configs
    # )

    print("=== 所有实验完成 ===")
    return {
        #'hyperparameter': hyper_results,
        'sampling': sampling_results,
        #'guided_generation': guided_results
    }

if __name__ == "__main__":
    # 运行实验
    all_results = run_all_experiments()
    print("实验完成！结果保存在 experiment_results/ 目录中")