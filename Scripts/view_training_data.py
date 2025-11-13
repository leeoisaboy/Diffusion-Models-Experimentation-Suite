"""
查看训练数据的实际样子
"""

import numpy as np
import matplotlib.pyplot as plt

# 加载训练数据
sprites = np.load('sprites_1788_16x16.npy')
labels = np.load('sprite_labels_nc_1788_16x16.npy')

print(f"数据形状: {sprites.shape}")
print(f"标签形状: {labels.shape}")
print(f"数据范围: [{sprites.min()}, {sprites.max()}]")

# 显示一些训练样本
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    idx = i * 50  # 每隔50个样本显示一个
    if idx < len(sprites):
        img = sprites[idx]
        label = labels[idx]
        ax.imshow(img)
        ax.set_title(f"{label}", fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig('training_data_samples.png', dpi=150, bbox_inches='tight')
print("训练数据样本已保存到 training_data_samples.png")

# 显示特定类别的样本
print("\n=== 按类别显示样本 ===")
categories = {
    'Human': [1, 0, 0, 0, 0],
    'Non-Human': [0, 1, 0, 0, 0],
    'Food': [0, 0, 1, 0, 0],
    'Spell': [0, 0, 0, 1, 0],
    'Side-Facing': [0, 0, 0, 0, 1]
}

fig, axes = plt.subplots(5, 8, figsize=(16, 10))
fig.suptitle('Training Data by Category', fontsize=14)

for row_idx, (cat_name, cat_label) in enumerate(categories.items()):
    # 找到匹配该类别的样本
    matches = []
    for i in range(len(labels)):
        if np.array_equal(labels[i], cat_label):
            matches.append(i)
        if len(matches) >= 8:
            break

    print(f"{cat_name}: 找到 {len(matches)} 个样本")

    for col_idx in range(8):
        ax = axes[row_idx, col_idx]
        if col_idx < len(matches):
            img = sprites[matches[col_idx]]
            ax.imshow(img)
        ax.axis('off')
        if col_idx == 0:
            ax.set_ylabel(cat_name, fontsize=10, rotation=0, ha='right', va='center')

plt.tight_layout()
plt.savefig('training_data_by_category.png', dpi=150, bbox_inches='tight')
print("分类训练数据已保存到 training_data_by_category.png")
