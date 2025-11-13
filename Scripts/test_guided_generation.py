"""
测试优化后的引导生成功能
"""

import torch
from experiment_design import ExperimentManager
from pathlib import Path

def test_guided_generation():
    """测试引导生成"""
    print("=== 测试优化后的引导生成 ===")

    # 初始化实验管理器
    exp_manager = ExperimentManager()

    # 检查模型文件
    model_path = 'weights/context_model_31.pth'
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("可用的模型文件:")
        weights_dir = Path('weights')
        if weights_dir.exists():
            for f in weights_dir.glob('*.pth'):
                print(f"  - {f}")
        return

    print(f"加载模型: {model_path}")

    # 测试配置 - 从简单到复杂
    context_configs = {
        'human': [1, 0, 0, 0, 0],           # 人类
        'non_human': [0, 1, 0, 0, 0],       # 非人类
        'food': [0, 0, 1, 0, 0],            # 食物
        'spell': [0, 0, 0, 1, 0],           # 法术
        'side_facing': [0, 0, 0, 0, 1],     # 侧脸
        'unconditional': [0, 0, 0, 0, 0],   # 无条件生成
        'mixed_human_food': [1, 0, 1, 0, 0],      # 人类+食物
        'mixed_spell_side': [0, 0, 0, 1, 1],      # 法术+侧脸
    }

    results = exp_manager.guided_generation_experiment(model_path, context_configs)

    print("\n=== 生成完成 ===")
    print(f"结果保存在: {exp_manager.results_dir / 'samples'}")

    # 显示生成的文件
    samples_dir = exp_manager.results_dir / 'samples'
    print("\n生成的图像文件:")
    for f in sorted(samples_dir.glob('guided_*.png')):
        print(f"  - {f.name}")

if __name__ == "__main__":
    test_guided_generation()
