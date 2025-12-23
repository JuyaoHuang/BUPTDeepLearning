"""
任务2: 测试不同迭代次数的模型效果
使用 cornell.jpg (content) + woman_with_hat_matisse.jpg (style)
生成 10%, 50%, 80%, 100% 迭代次数的风格迁移结果
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization


def test_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, default='input/content/cornell.jpg',
                    help='Content image path')
parser.add_argument('--style', type=str, default='input/style/woman_with_hat_matisse.jpg',
                    help='Style image path')
parser.add_argument('--model_dir', type=str, default='./experiments',
                    help='Directory containing trained decoder models')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth',
                    help='VGG model path')
parser.add_argument('--output', type=str, default='./output_iterations',
                    help='Output directory')
parser.add_argument('--max_iter', type=int, default=10000,
                    help='Max iterations used during training (to calculate percentages)')
parser.add_argument('--content_size', type=int, default=512,
                    help='Content image size')
parser.add_argument('--style_size', type=int, default=512,
                    help='Style image size')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建输出目录
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# 计算需要测试的迭代次数 (10%, 50%, 80%, 100%)
iterations = {
    '10%': int(args.max_iter * 0.1),
    '50%': int(args.max_iter * 0.5),
    '80%': int(args.max_iter * 0.8),
    '100%': args.max_iter
}
print(f"Testing iterations: {iterations}")

vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg, weights_only=True))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
vgg.eval()

# 加载图像
content_tf = test_transform(args.content_size)
style_tf = test_transform(args.style_size)

content = content_tf(Image.open(args.content).convert('RGB'))
style = style_tf(Image.open(args.style).convert('RGB'))
content = content.to(device).unsqueeze(0)
style = style.to(device).unsqueeze(0)

print(f"Content image: {args.content}")
print(f"Style image: {args.style}")

# 保存原始 content 和 style 图像供迁移对比
save_image(content, output_dir / 'content.jpg')
save_image(style, output_dir / 'style.jpg')

# 对每个迭代次数的模型进行测试
for percentage, iter_num in iterations.items():
    model_path = Path(args.model_dir) / f'decoder_iter_{iter_num}.pth.tar'

    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}")
        continue

    print(f"[{percentage}] Loading model: {model_path}")

    decoder = net.decoder
    decoder.load_state_dict(torch.load(model_path, weights_only=True))
    decoder.to(device)
    decoder.eval()

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, alpha=1.0)

    output_name = output_dir / f'stylized_{percentage}_iter{iter_num}.jpg'
    save_image(output, str(output_name))
    print(f"[{percentage}] Saved: {output_name}")

print(f"\nAll results saved to: {output_dir}")
