"""
对比训练数据和生成结果
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusion_utilities import ContextUnet, sample_ddim

# 加载训练数据
sprites = np.load('sprites_1788_16x16.npy')
labels = np.load('sprite_labels_nc_1788_16x16.npy')

# 找到食物类别的样本
food_label = [0, 0, 1, 0, 0]
food_indices = []
for i in range(len(labels)):
    if np.array_equal(labels[i], food_label):
        food_indices.append(i)
        if len(food_indices) >= 8:
            break

print(f"找到 {len(food_indices)} 个食物样本")

# 创建对比图
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Training Data (Food) - Real Sprites', fontsize=14)

for idx, ax in enumerate(axes.flat[:len(food_indices) * 4]):
    row = idx // 8
    if row < 2:  # 前两行显示训练数据
        sample_idx = (idx % 8) + (row * 8)
        if sample_idx < len(food_indices):
            img = sprites[food_indices[sample_idx]]
            ax.imshow(img)
            ax.axis('off')

plt.tight_layout()
plt.savefig('comparison_training_food.png', dpi=150, bbox_inches='tight')
print("训练数据对比图已保存")

# 加载生成的样本
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=16).to(device)
model.load_state_dict(torch.load('weights/context_model_31.pth', weights_only=False))
model.eval()

print("\n生成新样本用于对比...")
context = torch.tensor([[0, 0, 1, 0, 0]] * 16, dtype=torch.float32).to(device)  # food

# 尝试不同的采样步数
for n_steps in [25, 50, 100]:
    print(f"\n使用 {n_steps} 步 DDIM 采样...")
    samples = sample_ddim(model, 16, (3, 16, 16), device, n=n_steps, context=context, timesteps=500, eta=0.0)

    print(f"生成样本范围: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"生成样本均值: {samples.mean():.3f}, 标准差: {samples.std():.3f}")

    # 保存对比图
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle(f'Generated Food Sprites (DDIM {n_steps} steps)', fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            img = samples[idx].permute(1, 2, 0).cpu().numpy()
            img = np.clip((img + 1) / 2, 0, 1)
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'comparison_generated_food_{n_steps}steps.png', dpi=150, bbox_inches='tight')
    print(f"生成结果已保存到 comparison_generated_food_{n_steps}steps.png")

# 检查模型是否能从真实数据重建
print("\n\n=== 测试模型重建能力 ===")
from diffusion_utilities import ddpm_add_noise, transform

# 取一个真实的食物精灵
real_sprite = sprites[food_indices[0]].astype(np.float32) / 255.0  # 转换到[0,1]
real_sprite = real_sprite * 2 - 1  # 转换到[-1,1]，与训练时一致

# 转换为torch tensor
real_sprite_torch = torch.from_numpy(real_sprite).permute(2, 0, 1).unsqueeze(0).float().to(device)

print(f"真实精灵范围: [{real_sprite_torch.min():.3f}, {real_sprite_torch.max():.3f}]")

# 添加噪声然后去噪
t_test = torch.tensor([250]).to(device)
noisy_sprite, actual_noise = ddpm_add_noise(real_sprite_torch, t_test)

# 预测噪声
context_single = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32).to(device)
t_normalized = torch.tensor([[250/500]]).to(device)[:, None, None, None]

with torch.no_grad():
    pred_noise = model(noisy_sprite, t_normalized, context_single)

# 计算去噪后的图像
from diffusion_utilities import ab_t
denoised = (noisy_sprite - (1 - ab_t[250]).sqrt() * pred_noise) / ab_t[250].sqrt()

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
titles = ['Original', 'Noisy (t=250)', 'Predicted Noise', 'Denoised']
images = [real_sprite_torch, noisy_sprite, pred_noise, denoised]

for ax, img, title in zip(axes, images, titles):
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np + 1) / 2, 0, 1)
    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('reconstruction_test.png', dpi=150, bbox_inches='tight')
print("重建测试已保存到 reconstruction_test.png")
