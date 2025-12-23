import os
import torch
import pandas as pd
from tqdm import tqdm

from dataloader import FoodDataset
from transforms import test_transform
from model_vgg import VGG16Food


def predict(model, test_loader, device):
    model.eval()
    predictions = []
    filenames = []

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy())
            filenames.extend(names)

    return filenames, predictions


def main():
    data_dir = 'datasets/food11'
    model_path = 'best_vgg16.pth'
    output_path = 'ans_vgg.csv'
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    test_dataset = FoodDataset(
        root=os.path.join(data_dir, 'evaluation'),
        transform=test_transform,
        mode='test'
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"测试集样本数: {len(test_dataset)}")

    model = VGG16Food(num_classes=11, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已加载模型: {model_path}")

    filenames, predictions = predict(model, test_loader, device)

    df = pd.DataFrame({
        'Id': filenames,
        'Category': predictions
    })
    df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存到: {output_path}")
    print(f"共 {len(df)} 条预测")

    print("\n前10条预测结果:")
    print(df.head(10))


if __name__ == '__main__':
    main()
