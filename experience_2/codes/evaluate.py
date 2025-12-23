import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from dataloader import FoodDataset, CLASS_NAMES
from transforms import test_transform
from model import FoodCNN

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate(model, val_loader, device):
    """评估模型，返回预测结果和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('混淆矩阵', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存到: {save_path}")


def main():
    data_dir = 'datasets/food11'
    model_path = 'best_food_cnn.pth'
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    val_dataset = FoodDataset(
        root=os.path.join(data_dir, 'validation'),
        transform=test_transform,
        mode='val'
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"验证集样本数: {len(val_dataset)}")

    model = FoodCNN(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"已加载模型: {model_path}")
    model = model.to(device)

    predictions, labels = evaluate(model, val_loader, device)

    accuracy = (predictions == labels).sum() / len(labels) * 100
    print(f"\n验证集准确率: {accuracy:.2f}%")

    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=CLASS_NAMES))

    plot_confusion_matrix(labels, predictions, CLASS_NAMES)


if __name__ == '__main__':
    main()
