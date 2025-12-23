"""
训练深度自编码器
- 使用BCELoss作为损失函数
- bottleneck层维度为2
- 展示0-9数字的原始图片和重建图片
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import AutoEncoder
from dataloader import get_dataloader, get_samples

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def train(model, train_loader, loss_fn, op, scheduler, device, epochs=50):
    """训练自编码器"""
    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        # _ 是标签，自编码器训练不需要使用标签（无监督学习）
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # 前向传播
            _, reconstructed = model(images)

            loss = loss_fn(reconstructed, images)
            op.zero_grad()
            loss.backward()
            op.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # 将每一轮的平均损失值传给学习率调度器
        scheduler.step(avg_loss)
        lr = op.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Learning Rate: {lr:.6f}')

    return loss_history


def visualize_reconstruction(model, data_dir, device, save_path='results'):
    """可视化0-9数字的原始图片和重建图片"""
    model.eval()

    # 获取0-9的样本
    samples, labels = get_samples(data_dir)
    samples = samples.to(device)

    with torch.no_grad():
        _, reconstructed = model(samples)

    # 转换为numpy用于绘图
    original = samples.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    os.makedirs(save_path, exist_ok=True)

    # 绘制对比图
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fig.suptitle('Original(top) vs Reconstruction(bottom)', fontsize=14)

    for i in range(10):
        axes[0, i].imshow(original[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(str(i))

        axes[1, i].imshow(reconstructed[i, 0], cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reconstruction.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"重建结果已保存到 {save_path}/reconstruction.png")


def plot_loss(loss_history, save_path='results'):
    """绘制训练损失曲线"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"损失曲线已保存到 {save_path}/loss_curve.png")


def main():
    batch_size = 256
    learning_rate = 1e-3
    epochs = 100
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

    print("\n开始训练...")
    loss_history = train(model, train_loader, loss_fn, optimizer, scheduler,device, epochs)

    # 保存模型
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'autoencoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到 {model_path}")

    # 可视化结果
    plot_loss(loss_history, save_path)
    visualize_reconstruction(model, data_dir, device, save_path)


if __name__ == '__main__':
    main()
