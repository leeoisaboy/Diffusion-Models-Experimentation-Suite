"""
诊断模型和采样过程
"""

import torch
import numpy as np
from diffusion_utilities import ContextUnet, ddpm_add_noise
import matplotlib.pyplot as plt

def diagnose_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型
    model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=16).to(device)
    model.load_state_dict(torch.load('weights/context_model_31.pth', weights_only=False))
    model.eval()

    print("\n=== 模型参数检查 ===")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 测试1: 检查模型输入输出
    print("\n=== 测试1: 模型前向传播 ===")
    batch_size = 4
    x = torch.randn(batch_size, 3, 16, 16).to(device)
    t = torch.tensor([0.5]).repeat(batch_size)[:, None, None, None].to(device)
    c = torch.tensor([[1, 0, 0, 0, 0]] * batch_size, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x, t, c)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output mean: {output.mean():.3f}, std: {output.std():.3f}")

    # 测试2: 检查不同上下文的输出差异
    print("\n=== 测试2: 不同上下文的输出差异 ===")
    contexts = {
        'human': [1, 0, 0, 0, 0],
        'food': [0, 0, 1, 0, 0],
        'unconditional': [0, 0, 0, 0, 0]
    }

    x_test = torch.randn(1, 3, 16, 16).to(device)
    t_test = torch.tensor([[0.5]]).to(device)[:, None, None, None]

    outputs = {}
    with torch.no_grad():
        for name, context in contexts.items():
            c_test = torch.tensor([context], dtype=torch.float32).to(device)
            out = model(x_test, t_test, c_test)
            outputs[name] = out
            print(f"{name}: mean={out.mean():.3f}, std={out.std():.3f}")

    # 计算输出差异
    diff_human_food = (outputs['human'] - outputs['food']).abs().mean()
    diff_human_uncond = (outputs['human'] - outputs['unconditional']).abs().mean()
    print(f"\nDifference (human vs food): {diff_human_food:.3f}")
    print(f"Difference (human vs unconditional): {diff_human_uncond:.3f}")

    if diff_human_food < 0.01:
        print("⚠️  警告: 不同上下文的输出几乎相同！模型可能没有正确学习条件信息。")

    # 测试3: 检查噪声调度
    print("\n=== 测试3: 噪声调度检查 ===")
    from diffusion_utilities import ab_t
    print(f"ab_t[0]: {ab_t[0]:.6f}")
    print(f"ab_t[250]: {ab_t[250]:.6f}")
    print(f"ab_t[500]: {ab_t[500]:.6f}")

    # 测试4: 模拟一步去噪
    print("\n=== 测试4: 模拟去噪过程 ===")
    # 创建一个真实图像并加噪
    real_img = torch.zeros(1, 3, 16, 16).to(device)
    real_img[:, :, 4:12, 4:12] = 1.0  # 创建一个白色方块

    t_step = torch.tensor([250]).to(device)
    noisy_img, actual_noise = ddpm_add_noise(real_img, t_step)

    print(f"Original image range: [{real_img.min():.3f}, {real_img.max():.3f}]")
    print(f"Noisy image range: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
    print(f"Actual noise range: [{actual_noise.min():.3f}, {actual_noise.max():.3f}]")

    # 让模型预测噪声
    t_normalized = torch.tensor([[250/500]]).to(device)[:, None, None, None]
    c_test = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_noise = model(noisy_img, t_normalized, c_test)

    print(f"Predicted noise range: [{pred_noise.min():.3f}, {pred_noise.max():.3f}]")

    # 计算预测误差
    noise_error = torch.nn.functional.mse_loss(pred_noise, actual_noise)
    print(f"Noise prediction MSE: {noise_error:.6f}")

    if noise_error > 1.0:
        print("⚠️  警告: 噪声预测误差很大！模型可能没有训练好。")

    # 测试5: 完整的采样测试（少量步数）
    print("\n=== 测试5: 采样测试 (50步) ===")
    from diffusion_utilities import sample_ddim

    context_test = torch.tensor([[0, 0, 1, 0, 0]] * 4, dtype=torch.float32).to(device)  # food
    samples = sample_ddim(model, 4, (3, 16, 16), device, n=50, context=context_test, timesteps=500, eta=0.0)

    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Samples mean: {samples.mean():.3f}, std: {samples.std():.3f}")

    # 保存诊断结果图像
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(4):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip((img + 1) / 2, 0, 1)  # 从[-1,1]归一化到[0,1]
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('diagnosis_samples.png', dpi=150, bbox_inches='tight')
    print("诊断样本已保存到 diagnosis_samples.png")

    # 检查样本是否有结构
    variance_per_sample = samples.var(dim=[1, 2, 3])
    print(f"\nVariance per sample: {variance_per_sample}")

    if variance_per_sample.mean() < 0.01:
        print("⚠️  警告: 样本方差太小，可能生成了均匀的颜色而不是有意义的图像。")

if __name__ == "__main__":
    diagnose_model()
