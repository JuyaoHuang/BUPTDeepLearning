from torchvision import transforms

# 图像尺寸
IMG_SIZE = 224

# 训练集变换
train_transform = transforms.Compose([
    # 1. Resize: 统一图片尺寸
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # 2. RandomHorizontalFlip: 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 3. RandomRotation: 随机旋转
    transforms.RandomRotation(15),
    # 4. ColorJitter: 颜色抖动
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # 5. RandomAffine: 随机仿射变换
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集/测试集变换 (不做数据增强)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 用于可视化的变换 (不含Normalize，方便显示)
visual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])


if __name__ == '__main__':
    print("Train Transform 包含以下变换:")
    for i, t in enumerate(train_transform.transforms, 1):
        print(f"  {i}. {t.__class__.__name__}")
    print(f"\n共 {len(train_transform.transforms)} 种变换")
