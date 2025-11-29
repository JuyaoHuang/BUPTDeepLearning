import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import get_dataloader
from transforms import train_transform, test_transform
from model_vgg import VGG16Food


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
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


def validate(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train(config):
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

    # VGG16 模型 (使用预训练权重)
    model = VGG16Food(num_classes=11, pretrained=True).to(device)
    print("已加载 VGG16 预训练权重")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TensorBoard
    writer = SummaryWriter(log_dir=config['log_dir'])

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    writer.close()
    print(f"\n训练结束.\n最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存到: {config['save_path']}")

    return model


if __name__ == '__main__':
    config = {
        'data_dir': 'datasets/food11',
        'batch_size': 64,
        'epochs': 15, # 二次训练轮数可以低一点(显卡照样顶不住)
        'lr': 0.0001,
        'log_dir': 'runs/vgg16',
        'save_path': 'best_vgg16.pth'
    }

    os.makedirs(config['log_dir'], exist_ok=True)
    train(config)
