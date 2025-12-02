# -*- coding: utf-8 -*-
"""
Latent Code 采样与可视化
1. 基于任务(2)训练好的模型对 latent code 进行均匀采样
2. 利用解码器对采样结果进行恢复
3. 展示 latent code 的分布、采样范围和采样结果
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model import AutoEncoder
from dataloader import get_dataloader


def visualize_latent_distribution(model, data_loader, device, save_path='results'):
    """
    可视化所有测试数据在latent space中的分布
    """
    model.eval()

    all_latent = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            latent, _ = model(images)
            all_latent.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latent = np.concatenate(all_latent, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 绘制latent space分布
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_latent[:, 0], all_latent[:, 1],
                         c=all_labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Distribution of MNIST Digits')
    plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'latent_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Latent分布图已保存到 {save_path}/latent_distribution.png")

    x_min, x_max = all_latent[:, 0].min(), all_latent[:, 0].max()
    y_min, y_max = all_latent[:, 1].min(), all_latent[:, 1].max()
    print(f"\nLatent空间范围:")
    print(f"  维度1: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  维度2: [{y_min:.2f}, {y_max:.2f}]")

    # 返回latent范围
    return x_min, x_max, y_min, y_max


def sample_and_decode(model, device, x_range, y_range, n_samples=20, save_path='results'):
    """
    在latent space中均匀采样，并用decoder生成图像
    """
    model.eval()

    x_min, x_max = x_range
    y_min, y_max = y_range

    # 生成均匀采样的网格
    x_values = np.linspace(x_min, x_max, n_samples)
    y_values = np.linspace(y_min, y_max, n_samples)

    # 创建画布
    fig, axes = plt.subplots(n_samples, n_samples, figsize=(15, 15))
    fig.suptitle(f'Latent Space Sampling\nX: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}]',
                 fontsize=14)

    with torch.no_grad():
        for i, y in enumerate(reversed(y_values)):
            for j, x in enumerate(x_values):
                latent = torch.tensor([[x, y]], dtype=torch.float32).to(device)
                decoded = model.decoder(latent)
                decoded = decoded.view(28, 28).cpu().numpy()
                axes[i, j].imshow(decoded, cmap='gray')
                axes[i, j].axis('off')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'latent_sampling.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"采样结果已保存到 {save_path}/latent_sampling.png")


def main():
    data_dir = './dataset'
    save_path = './results'
    model_path = os.path.join(save_path, 'autoencoder.pth')
    n_samples = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = AutoEncoder().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return

    test_loader = get_dataloader(data_dir, batch_size=256, train=False, shuffle=False)
    print(f"测试集大小: {len(test_loader.dataset)}")

    print("\n1. 可视化Latent Space分布...")
    x_min, x_max, y_min, y_max = visualize_latent_distribution(model, test_loader, device, save_path)

    print(f"\n2. 在Latent Space中均匀采样 ({n_samples}x{n_samples})...")
    margin = 0.5
    sample_and_decode(
        model, device,
        x_range=(x_min - margin, x_max + margin),
        y_range=(y_min - margin, y_max + margin),
        n_samples=n_samples,
        save_path=save_path
    )

    print("\n完成！")


if __name__ == '__main__':
    main()
