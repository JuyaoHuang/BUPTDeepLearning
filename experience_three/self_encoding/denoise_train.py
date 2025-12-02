"""
降噪自编码器
- 噪声因子为0.4
- 在输入图像上叠加均值为0且方差为1的标准高斯白噪声, 训练降噪自编码器
- 给出数字从0到9的10张图片的原始图片、加噪图片和重建图片
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

from model import AutoEncoder
from dataloader import get_dataloader, get_samples

def add_noise(images, noise_factor=0.4):
    """
    添加高斯噪声

    Args:
        images: 输入图像 tensor
        noise_factor: 噪声因子

    Returns:
        加噪后的图像（截断到[0,1]）
    """
    # 生成均值为0，方差为1的高斯噪声
    noise = torch.randn_like(images)

    noisy_images = images + noise_factor * noise
    # 将像素值截断到[0, 1]范围
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images


def train_denoising(model, train_loader, loss_fn, optimizer, scheduler, device, epochs=50, noise_factor=0.4):
    """训练降噪自编码器"""
    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # 添加噪声
            noisy_images = add_noise(images, noise_factor)

            _, reconstructed = model(noisy_images)

            # 计算损失：与原始干净图像对比
            loss = loss_fn(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Learning Rate: {lr:.6f}')

    return loss_history


def visualize_denoising(model, data_dir, device, noise_factor=0.4, save_path='results'):
    """可视化降噪效果：原始图片、加噪图片、重建图片"""
    model.eval()

    # 获取0-9的样本
    samples, labels = get_samples(data_dir)
    samples = samples.to(device)

    # 添加噪声
    torch.manual_seed(42)
    noisy_samples = add_noise(samples, noise_factor)

    with torch.no_grad():
        _, reconstructed = model(noisy_samples)

    # 转换为numpy用于绘图
    original = samples.cpu().numpy()
    noisy = noisy_samples.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 绘制图像
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    fig.suptitle('Denoising Autoencoder Results\nOriginal (top) / Noisy (middle) / Reconstructed (bottom)',
                 fontsize=12)

    for i in range(10):
        # 原始图片
        axes[0, i].imshow(original[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(str(i))

        # 加噪图片
        axes[1, i].imshow(noisy[i, 0], cmap='gray')
        axes[1, i].axis('off')

        # 重建图片
        axes[2, i].imshow(reconstructed[i, 0], cmap='gray')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'denoising_result.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"降噪结果已保存到 {save_path}/denoising_result.png")


def plot_loss(loss_history, save_path='results'):
    """绘制训练损失曲线"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Denoising Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'denoise_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"损失曲线已保存到 {save_path}/denoise_loss_curve.png")


def main():
    batch_size = 256
    learning_rate = 1e-3
    epochs = 120
    noise_factor = 0.4
    data_dir = './dataset'
    save_path = './results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader = get_dataloader(data_dir, batch_size=batch_size, train=True)
    print(f"训练集大小: {len(train_loader.dataset)}")

    model = AutoEncoder().to(device)
    print("\n模型结构:")
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print(f"\n开始训练降噪自编码器 (噪声因子: {noise_factor})...")
    loss_history = train_denoising(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        noise_factor=noise_factor)

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'denoise_autoencoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到 {model_path}")

    plot_loss(loss_history, save_path)
    visualize_denoising(model, data_dir, device, noise_factor, save_path)


if __name__ == '__main__':
    main()
