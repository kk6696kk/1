import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

# =========================
# 1. 加载模型
# =========================
model = FeatExtNet_LDIntegrated(base_channels=16, num_stage=3)
model.eval()  # 推理模式（关闭BN统计）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =========================
# 2. 准备输入图像
# =========================
# 替换为你的图像路径
img_path = "demo_image.jpg"

image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# 3. 前向传播
# =========================
with torch.no_grad():
    outputs = model(input_tensor)

# =========================
# 4. 选择要可视化的特征图 key
# =========================
keys_to_vis = ["stage1_f", "stage2_f", "stage3_f", "stage_fine"]

# =========================
# 5. 创建保存目录
# =========================
save_dir = "feature_vis"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 6. 绘制每一层的特征图
# =========================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('LDConv Feature Maps', fontsize=28)
plt.subplots_adjust(hspace=0.2, wspace=0.1)

for ax, key in zip(axs.flat, keys_to_vis):
    feat = outputs[key].squeeze(0).detach().cpu()  # (C, H, W)

    # 归一化到0~1
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

    # 将多个通道压缩为单通道（取均值）
    feat_map = feat_norm.mean(dim=0).numpy()

    ax.imshow(feat_map, cmap='viridis')
    ax.set_title(f"{key}", fontsize=20)
    ax.axis('off')

plt.tight_layout()
plt.show()

# =========================
# 7. 如果需要保存每个阶段的可视化图
# =========================
for key in keys_to_vis:
    feat = outputs[key].squeeze(0).detach().cpu()
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    feat_map = feat_norm.mean(dim=0).numpy()
    plt.imsave(os.path.join(save_dir, f"{key}.png"), feat_map, cmap='viridis')
