## 📋 Project Overview

2025 年北京邮电大学 BUPT 神经网络与深度学习课程实验合集。

> [!NOTE]
> 源码仅供参考和学习使用，请自主完成实验。
## ✨ Structure

每一个实验内容包含：
1. 实验报告
2. 源代码

## 📋Content

1. [CIFAR-10 图像分类问题](https://github.com/JuyaoHuang/BUPTDeepLearning/blob/main/experience_1/report/CIFAR.md)
2. [CV基础](https://github.com/JuyaoHuang/BUPTDeepLearning/blob/main/experience_2/report/experience2.md)：CNN 中的 VGG 模型应用
3. [自编码器的实现](https://github.com/JuyaoHuang/BUPTDeepLearning/blob/main/experience_3/report/experience3.md)：SAE、VAE
4. [图像风格迁移的应用](https://github.com/JuyaoHuang/BUPTDeepLearning/blob/main/experience_3/report/experience3.md)
5. [Transformer实现语义分析](./)

## 🔗 Pretrained Model

以百度网盘的方式提供训练好的模型。

[网盘链接](https://pan.baidu.com/s/1SIA-L_P-mVItnBFleWeolQ?pwd=2fdk)：

通过网盘分享的文件：深度学习训练模型
链接: https://pan.baidu.com/s/1SIA-L_P-mVItnBFleWeolQ?pwd=2fdk 提取码: 2fdk

### 1.实验一 CIFAR-10 

该部分有数个预训练模型，适配不同的实验要求：
| 模型名                      | 备注                  | 卷积层数 | Cov 层神经元                        | 全连接层神经元           | 激活函数  |
| --------------------------- | --------------------- | -------- | --------------------------------- | :----------------------- | --------- |
| **cifar_net.pth**           | 原始模型              | 三层     | 3 -> 16 -> 32 -> 64               | 1024 -> 120 -> 84 -> 10  | ReLU      |
| **CIFAR-5layer-CNN.pth**    | 模型对比的五层CNN模型 | 五层     | 3 -> 16 -> 32 -> 64 -> 128 -> 128 | 同上                     | ReLU      |
| **CIFAR-CNN-LeakyReLU.pth** |                       | 三层     | 同一                              | 同上                     | LeakyReLU |
| **CIFAR-CNN-ReLU.pth**      | 原始模型              | 三层     | 同一                              | 同上                     | ReLU      |
| **CIFAR-CNN-Sigmoid.pth**   |                       | 三层     | 同一                              | 同上                     | Sigmoid   |
| **CIFAR_wideCov.pth**       | 更多的卷积层神经元    | 三层     | 32 -> 64-> 128                    | 同上                     | LeakyReLU |
| **CIFAR_wideFC.pth**        | 更多的全连接层神经元  | 三层     | 同一                              | 1024 -> 256 -> 128 -> 10 | LeakyReLU |
| **CIFAR_with_softmax.pth**  | 最后一层使用 softmax  | 三层     | 同一                              | 同一                     | ReLU      |

> ![NOTE]
> 模型对比[参考此篇文章](https://www.juayohuang.top/posts/projects/dl/ex1/cifar)

### 2. 实验二 CNN

1. **best_food_cnn.pth**：自定义 4 层 Cov + 三层 FC 的预训练模型
2. **best_vgg16.pth**：预训练 VGG16 模型
   
### 3. 实验三 自编码器与风格迁移

> ![NOTE]
> 自编码器预训练模型未提供，但比较简单， RTX 4060 5分钟左右就能训练一个不错的效果

1. **解码模型**：style_transfer/pytorch-AdaIN/experiments/ 为解码器模型训练时，不同迭代轮次保存的模型。命名格式：`decoder_iter_<epoches>.pth.tar`
2. **vgg_normalised.pth**：VGG 预训练模型

## 📄 License

MIT License - See the [LICENSE](./LICENCE.md) file for details.

