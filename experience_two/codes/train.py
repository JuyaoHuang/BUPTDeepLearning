import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import get_dataloader, CLASS_NAMES
from transforms import train_transform, test_transform
from model import FoodCNN


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    一个 epoch 的训练
    :return:
    epoch_loss: 该批次的平均损失
    epoch_acc: 该批次的平均准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc='Training', leave=True):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    """
    epoch_loss: 该批次的平均损失
    epoch_acc: 该批次的平均准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train(config):
    """完整训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, val_loader, _ = get_dataloader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_transform=train_transform,
        test_transform=test_transform
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    model = FoodCNN(num_classes=11).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # TensorBoard
    writer = SummaryWriter(log_dir=config['log_dir'])

    # 记录模型结构
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 学习率调度
        scheduler.step()

        # TensorBoard 记录
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    writer.close()
    print(f"\n训练结束.\n最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存到: {config['save_path']}")
    print(f"TensorBoard 日志: {config['log_dir']}")
    print(f"运行 'tensorboard --logdir={config['log_dir']}' 查看训练曲线")

    return model


if __name__ == '__main__':
    config = {
        'data_dir': 'datasets/food11',
        'batch_size': 64,
        'epochs': 30,
        'lr': 0.001,
        'log_dir': 'runs/food_cnn',
        'save_path': 'best_food_cnn.pth'
    }

    # 创建日志目录
    os.makedirs(config['log_dir'], exist_ok=True)

    train(config)
