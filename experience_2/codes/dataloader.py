import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 11个食物类别
CLASS_NAMES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
    "Vegetable/Fruit"
]


class FoodDataset(Dataset):
    """
    Food-11 数据集的自定义 Dataset 类
    文件名格式: [类别]_[编号].jpg
    例如: 0_123.jpg 表示类别0(Bread)的第123张图片
    """

    def __init__(self, root, transform=None, mode='train'):
        """
        Args:
            root: 数据集目录路径 (如 datasets/food11/training)
            transform: 图像变换操作
            mode: 'train', 'val' 或 'test'，用于区分是否需要标签
        """
        self.root = root
        self.transform = transform
        self.mode = mode

        # 获取所有图片文件
        self.images = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])

        if mode != 'test':
            self.labels = []
            for img_name in self.images:
                # 文件名格式: [类别]_[编号].jpg
                label = int(img_name.split('_')[0])
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image, self.images[idx]  # 测试集返回图片和文件名
        else:
            return image, self.labels[idx]  # 训练/验证集返回图片和标签

    def get_class_name(self, label):
        """根据标签获取类别名称"""
        return CLASS_NAMES[label]


def get_dataloader(data_dir, batch_size=64, train_transform=None, test_transform=None):
    """
    创建训练集、验证集和测试集的 DataLoader

    Args:
        data_dir: 数据集根目录 (如 datasets/food11)
        batch_size: 批次大小
        train_transform: 训练集变换
        test_transform: 验证集/测试集变换

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = FoodDataset(
        root=os.path.join(data_dir, 'training'),
        transform=train_transform,
        mode='train',

    )

    val_dataset = FoodDataset(
        root=os.path.join(data_dir, 'validation'),
        transform=test_transform,
        mode='val'
    )

    test_dataset = FoodDataset(
        root=os.path.join(data_dir, 'evaluation'),
        transform=test_transform,
        mode='test'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from torchvision import transforms
    from collections import Counter

    data_dir = 'datasets/food11'

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = FoodDataset(
        root=os.path.join(data_dir, 'training'),
        transform=simple_transform,
        mode='train'
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"类别数: {len(CLASS_NAMES)}")

    image, label = train_dataset[0]
    print(f"\n第一个样本:")
    print(f"图片形状: {image.shape}")
    print(f"标签: {label} ({train_dataset.get_class_name(label)})")

    label_counts = Counter(train_dataset.labels)
    print(f"\n各类别样本数:")
    for label, count in sorted(label_counts.items()):
        print(f"{label} ({CLASS_NAMES[label]}): {count}")
