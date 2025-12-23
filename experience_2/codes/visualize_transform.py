import os
import matplotlib.pyplot as plt
from PIL import Image
from transforms import train_transform, visual_transform, IMG_SIZE
from torchvision import transforms

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def denormalize(tensor):
    """反归一化，将 Normalize 后的图片还原用于显示"""
    """使用 ImageNet 的均值和方差值"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_transforms(image_path, num_samples=5):
    """
    可视化 transform 效果
    展示原图和多次增强后的结果
    """
    origin_img = Image.open(image_path).convert('RGB')

    fig, axes = plt.subplots(2, num_samples + 1, figsize=(15, 6))

    # 第一行：原图 + 不带Normalize的增强效果
    axes[0, 0].imshow(origin_img)
    axes[0, 0].set_title('原图', fontsize=12)
    axes[0, 0].axis('off')

    for i in range(num_samples):
        transformed = visual_transform(origin_img)
        img_np = transformed.permute(1, 2, 0).numpy()
        axes[0, i + 1].imshow(img_np)
        axes[0, i + 1].set_title(f'增强效果 {i + 1}', fontsize=12)
        axes[0, i + 1].axis('off')

    # 第二行：带 Normalize 的效果,反归一化
    resize_normal = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    axes[1, 0].imshow(resize_normal(origin_img).permute(1, 2, 0).numpy())
    axes[1, 0].set_title('Resize后', fontsize=12)
    axes[1, 0].axis('off')

    for i in range(num_samples):
        transformed = train_transform(origin_img).clone()
        # 反归一化
        img_denorm = denormalize(transformed)
        img_denorm = img_denorm.clamp(0, 1)  # 限制在 [0, 1]
        img_np = img_denorm.permute(1, 2, 0).numpy()
        axes[1, i + 1].imshow(img_np)
        axes[1, i + 1].set_title(f'包含Normalize {i + 1}', fontsize=12)
        axes[1, i + 1].axis('off')

    plt.suptitle('上：不含Normalize | 下：包含Normalize且反归一化', fontsize=14)
    plt.tight_layout()
    plt.savefig('transform_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 选择一张训练图片进行可视化
    data_dir = 'datasets/food11/training'
    sample_images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    sample_image_path = os.path.join(data_dir, sample_images[8])

    visualize_transforms(sample_image_path, num_samples=5)
