import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

def get_dataloader_workers():
    """使用 4 个进程读取数据"""
    return 4

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始输入: [B, 3, 32, 32]
        # 第1个卷积块
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第2个卷积块
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 第3个卷积块
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # # 第四个卷积块
        # self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # self.pool4 = nn.MaxPool2d(2, 2)
        # # 第五个卷积块
        # self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        # self.pool5 = nn.MaxPool2d(2, 2)

        # 全连接层
        # 经过 3次池化后，特征图大小为 4x4，通道数为64
        # 因此 flatten后的向量维度是 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 顺序通过卷积块
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x)))
        # x = self.pool5(F.relu(self.conv5(x)))

        # 将所有维度展平成一维
        x = torch.flatten(x, 1)

        # 全连接层
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x


def train(model, device, trainloader, loss_function, optimizer, epoches, writer):
    """模型训练"""
    """
    return:
        model trained, history of training loss and training accuracy
    """
    model = model.to(device)

    history = {'train_loss': [], 'train_accuracy': []}

    # 切换为训练模式
    model.train()
    for epoch in range(epoches):

        train_loss = 0.0
        acc_cnt = 0
        for X,y in trainloader:
            X,y = X.to(device), y.to(device)

            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(X)
            # 计算loss
            loss = loss_function(outputs, y)
            loss.backward()
            # 更新权重
            optimizer.step()

            train_loss += loss.item()
            acc_cnt += (outputs.argmax(1) == y).type(torch.float).sum().item()

        train_loss = train_loss / len(trainloader)
        accuracy = acc_cnt / len(trainloader.dataset)

        print(f"Epoch [{epoch + 1}/{epoches}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Accuracy: {accuracy:.4f}")

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(accuracy)

        # 在每个 epoch 结束后，使用 writer 记录 loss 和 accuracy
        # writer.add_scalar(tag, scalar_value, global_step)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)

    print('训练完毕')

    return model,history


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print('Save Successfully')


def pred_model_in_testset(model, testloader, device):
    """在测试集上评测最终模型性能"""
    model = model.to(device)
    model.eval()

    acc_cnt = 0
    with torch.no_grad():
        for X, y in testloader:
            X,y = X.to(device), y.to(device)
            output = model(X)
            acc_cnt += (output.argmax(1) == y).type(torch.float).sum().item()

    accuracy = acc_cnt / len(testloader.dataset)
    return accuracy


def collect_pred_labels(model, dataloader, device):
    """
    遍历 Dataloader，收集模型的所有预测结果和真实标签
    :param model: 模型
    :param dataloader: 数据加载器
    :param device:
    :return:
        labels:numpy, preds:numpy
    """
    labels = torch.tensor([],device=device)
    preds = torch.tensor([],device=device)
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            output = model(X)
            predicts = output.argmax(dim=1, keepdim=True)

            # 拼接当前批次的预测值和标签到列表里
            labels = torch.cat((labels,y),dim=0)
            preds = torch.cat((preds,predicts),dim=0)

    # 直接返回 numpy并转移到cpu上，方便后续 seaborn绘图
    return labels.cpu().numpy(), preds.cpu().numpy()


def plot_confusion_matrix(cm, class_names, img_name, title='Confusion matrix'):
    """
    使用seaborn绘制混淆矩阵
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=class_names,yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{img_name}_CM.png',dpi=300)
    plt.show()


def display_confusion_matrix(PATH):

    batch_size = 256

    # 数据加载区块
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # shuffle = False 以保持顺序
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=get_dataloader_workers())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,

                                             shuffle=False, num_workers=get_dataloader_workers())
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = Net()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    from sklearn.metrics import confusion_matrix
    # 绘制训练集的混淆矩阵
    print('训练集混淆矩阵绘制：')
    train_labels, train_preds = collect_pred_labels(model, trainloader, device)
    cm_train = confusion_matrix(train_labels, train_preds)
    plot_confusion_matrix(cm=cm_train,
                          class_names=classes,
                          title='Train Set Confusion Matrix',
                          img_name='train_set')

    # 绘制测试集的混淆矩阵
    print('测试集混淆矩阵绘制：')
    test_labels, test_preds = collect_pred_labels(model, testloader, device)
    cm_test = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm=cm_test,
                          class_names=classes,
                          title='Test Set Confusion Matrix',
                          img_name='test_set')


def display_loss_curve(history: dict):
    """
    绘制历史的训练损失和准确率的折线图
    :param history: dict, {"train_loss":[loss values], "train_accuracy":[accuracy values]}
    :return: none
    """
    # 转为 DF，方便送入 seaborn
    df = pd.DataFrame(history)
    df['epoch'] = range(1, len(df)+1)

    sns.set_style('darkgrid')

    fig, ax = plt.subplots(1,2,figsize=(10,5))

    # 绘制训练损失
    sns.lineplot(
        data=df,
        x='epoch',
        y='train_loss',
        ax=ax[0],
        color='b',
        marker='o'
    )
    # 在坐标点上打印文字
    for index,row in df.iterrows():
        ax[0].text(
            row['epoch'],
            row['train_loss'],
            f"{row['train_loss']:.3f}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='k'
        )
    ax[0].set_title('Train Loss With Softmax')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    # 绘制训练准确率
    sns.lineplot(
        data=df,
        x='epoch',
        y='train_accuracy',
        color='r',
        marker='o'
    )
    for index,row in df.iterrows():
        ax[1].text(
            row['epoch'],
            row['train_accuracy'],
            f"{row['train_accuracy']:.3f}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='k'
        )

    ax[1].set_title('Train Accuracy With Softmax')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('with_softmax_net_loss.png',dpi=300)
    plt.show()


def valiation(PATH):
    """评估模块"""
    # 简单的图像评估区块
    model = Net()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    print('加载模型成功')

    # def imshow(img):
    #     img = img / 2 + 0.5  # 反标准化
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # data_iter = iter(testloader)
    # images, labels = next(data_iter)
    #
    # show_images = 8
    # images_to_show = images[:show_images]
    # labels_to_show = labels[:show_images]
    #
    # print('GroundTruth: ', ' '.join(f'{classes[labels_to_show[j]]:5s}' for j in range(show_images)))
    # # 预测标签
    # with torch.no_grad():
    #     outputs = model(images_to_show)
    #     _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(show_images)))
    #
    # imshow(torchvision.utils.make_grid(images_to_show))

    # 模型评估区块 (在测试集上评估)
    # def pred_model(model, testloader, loss_function, device):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracy = pred_model_in_testset(
        model=model,
        testloader=testloader,
        device=device
    )
    print(f'测试集准确率：{test_accuracy * 100:.2f}%')

    # 具体分类评估区块：
    # 在每一个分类上进行评估，观察每一个分类的准确率
    # 从已持久化模型加载模型
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # 累加每个类别的准确率计算变量
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Class: {classname:5s} Accuracy: {accuracy:.1f} %')

def train_model(PATH):

    # 训练设备选择区块
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'选择训练的设备：{device}')

    print(f'开始构建神经网络:')
    model = Net()
    print('神经网络结构：')
    print(model)
    print('神经网络构建完成')

    # 神经网络训练区块
    print('开始训练神经网络：')
    # 1.定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 2.定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 3.传入训练函数
    # def train(model, device, trainloader, loss_function, optimizer, epoches):
    # 使用 tensorboard 可视化训练过程
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./logs')
    trained_model,history = train(
        model=model,
        device=device,
        trainloader=trainloader,
        loss_function=loss_function,
        optimizer=optimizer,
        epoches=10,
        writer=writer
    )

    writer.close()
    print('结束Summary的写入')

    # 存储训练数据
    import json

    file_path = 'train_with_softmax.json'
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    # 模型持久化区块

    save_model(trained_model,PATH)

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 超参数定义区块
    batch_size = 256

    # 数据加载区块
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=get_dataloader_workers())

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=get_dataloader_workers())

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型训练函数
    # train_model(PATH='./CIFAR_with_softmax.pth')

    # 模型预测和验证函数
    # valiation(PATH='./CIFAR_with_softmax.pth')

    # # 损失曲线绘制函数
    # import json
    # file_path = 'train_with_softmax.json'
    # with open(file_path, "r", encoding="utf-8") as f:
    #     history = json.load(f)
    #
    # display_loss_curve(history=history)








