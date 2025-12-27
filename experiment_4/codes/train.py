"""
训练语义匹配模型
实验步骤(5)：模型训练与验证
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import get_dataloader
from Net import SemanticMatchingModel


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch
    :return: epoch_loss, epoch_acc
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits, _ = model(input_ids, token_type_ids, attention_mask)

        # 计算损失
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型
    :return: epoch_loss, epoch_acc
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits, _ = model(input_ids, token_type_ids, attention_mask)

            # 计算损失
            loss = criterion(logits, labels)

            # 统计
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train(config):
    """完整训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, val_loader, _, vocab = get_dataloader(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        max_len=config['max_len']
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"词表大小: {len(vocab)}")

    model = SemanticMatchingModel(
        vocab_size=len(vocab),
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        max_len=config['max_len'],
        num_classes=2,
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )

    writer = SummaryWriter(log_dir=config['log_dir'])

    # 记录模型结构（可选）
    try:
        dummy_input_ids = torch.randint(0, len(vocab), (1, config['max_len'])).to(device)
        dummy_token_type_ids = torch.zeros(1, config['max_len'], dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones(1, config['max_len'], dtype=torch.long).to(device)
        writer.add_graph(model, (dummy_input_ids, dummy_token_type_ids, dummy_attention_mask))
    except Exception as e:
        print(f"无法记录模型结构: {e}")

    best_val_acc = 0.0
    patience = 0
    max_patience = config.get('early_stop_patience', 10)

    print("\n开始训练...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, config['save_path'])
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n早停: 验证准确率 {max_patience} 个 epoch 未提升")
                break

    writer.close()
    print(f"\n训练结束")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存到: {config['save_path']}")
    print(f"TensorBoard 日志: {config['log_dir']}")
    print(f"\n运行以下命令查看训练曲线:")
    print(f"  tensorboard --logdir={config['log_dir']}")

    return model


if __name__ == '__main__':
    # 训练配置
    config = {
        'data_path': 'dataset',
        'batch_size': 64,
        'max_len': 64,
        'epochs': 40,
        'lr': 2e-3,
        'weight_decay': 0.005,
        'model_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'ffn_dim': 1024,
        'dropout': 0.05,
        'log_dir': 'runs/semantic_matching_6layers_v2',
        'save_path': 'best_model_6layers_v2.pth',
        'early_stop_patience': 10
    }

    # 创建日志目录
    os.makedirs(config['log_dir'], exist_ok=True)

    # 训练
    train(config)
