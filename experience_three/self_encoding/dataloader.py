import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct

class MNISTDataset(Dataset):
    """自定义MNIST数据集加载器，从idx文件读取"""

    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir: 数据集目录路径
            train: True加载训练集，False加载测试集
            transform: 可选的数据变换
        """
        self.transform = transform

        if train:
            images_file = os.path.join(data_dir, 'train-images.idx3-ubyte')
            labels_file = os.path.join(data_dir, 'train-labels.idx1-ubyte')
        else:
            images_file = os.path.join(data_dir, 't10k-images.idx3-ubyte')
            labels_file = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

        # 读取图像
        self.images = self.read_images(images_file)
        # 读取标签
        self.labels = self.read_labels(labels_file)

    def read_images(self, filepath):
        """读取idx3-ubyte格式的图像文件"""
        with open(filepath, 'rb') as f:
            # 读取magic number和维度信息
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            # 读取图像数据
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, 1, rows, cols)
            # 归一化到 [0, 1]
            images = images.astype(np.float32) / 255.0
        return images

    def read_labels(self, filepath):
        """读取idx1-ubyte格式的标签文件"""
        with open(filepath, 'rb') as f:
            # 读取magic number和数量
            magic, num = struct.unpack('>II', f.read(8))
            # 读取标签数据
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].copy())
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(data_dir, batch_size=256, train=True, shuffle=True, num_workers=4):
    """
    获取MNIST数据加载器

    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        train: 是否为训练集
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数

    Returns:
        DataLoader对象
    """
    dataset = MNISTDataset(data_dir, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def get_samples(data_dir, train=False):
    """
    获取每个数字(0-9)的一个样本

    Args:
        data_dir: 数据集目录
        train: 是否从训练集获取

    Returns:
        images: shape (10, 1, 28, 28) 的tensor
        labels: 0-9的标签列表
    """
    dataset = MNISTDataset(data_dir, train=train)

    samples = {}
    for i in range(len(dataset)):
        image, label = dataset[i]
        if label not in samples:
            samples[label] = image
        if len(samples) == 10:
            break

    # 按标签排序
    images = torch.stack([samples[i] for i in range(10)])
    labels = list(range(10))

    return images, labels


if __name__ == '__main__':
    # 测试数据加载
    data_dir = './dataset'

    # 测试训练集加载
    train_loader = get_dataloader(data_dir, batch_size=64, train=True)
    print(f"训练集批次数: {len(train_loader)}")

    # 获取一个batch查看
    images, labels = next(iter(train_loader))
    print(f"图像batch形状: {images.shape}")
    print(f"标签batch形状: {labels.shape}")
    print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")

    # 测试获取0-9样本
    samples, sample_labels = get_samples(data_dir)
    print(f"0-9样本形状: {samples.shape}")
