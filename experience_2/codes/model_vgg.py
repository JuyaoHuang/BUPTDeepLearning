import torch
import torch.nn as nn
from torchvision import models


class VGG16Food(nn.Module):
    """
    VGG16 模型用于 Food-11 分类
    修改最后一层全连接层输出为 11 类
    """

    def __init__(self, num_classes=11, pretrained=True):
        super(VGG16Food, self).__init__()

        if pretrained:
            self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            self.vgg = models.vgg16(weights=None)

        # 修改最后一层全连接层 (原本输出1000类 -> 11类)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)


class VGG19Food(nn.Module):
    """
    VGG19 模型用于 Food-11 分类
    修改最后一层全连接层输出为 11 类
    """

    def __init__(self, num_classes=11, pretrained=True):
        super(VGG19Food, self).__init__()

        if pretrained:
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            self.vgg = models.vgg19(weights=None)

        # 修改最后一层全连接层
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)


def print_model_summary(model, model_name):
    """打印模型结构和参数"""
    print(f"{model_name} 模型结构")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{model_name} 参数统计")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    # 打印 VGG16 和 VGG19 模型结构
    vgg16 = VGG16Food(num_classes=11, pretrained=False)
    print_model_summary(vgg16, "VGG16")

    x = torch.randn(2, 3, 224, 224)
    output = vgg16(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    print("\n")
    vgg19 = VGG19Food(num_classes=11, pretrained=False)
    print_model_summary(vgg19, "VGG19")
