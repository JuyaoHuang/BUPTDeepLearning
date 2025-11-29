import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataloader import FoodDataset, CLASS_NAMES
from transforms import test_transform
from model_vgg import VGG16Food

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

def main():
    data_dir = 'datasets/food11'
    model_path = 'best_vgg16.pth'
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

    model = VGG16Food(num_classes=11,pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"已加载模型: {model_path}")
    model = model.to(device)

    predictions, labels = evaluate(model, val_loader, device)

    accuracy = (predictions == labels).sum() / len(labels) * 100
    print(f"\n验证集准确率: {accuracy:.2f}%")

    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=CLASS_NAMES))


if __name__ == '__main__':
    main()
