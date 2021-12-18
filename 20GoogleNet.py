import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.utils import data

# 含有并行连结的网络
# 文章的观点是，使用不同大小的卷积核的组合是有利的
# Inception块
# 4个路径从不同层面抽取信息，然后再输出通道维合并
# 输出和输入等高等宽，但通道数不同

#            1*1卷积层
# 输入  ->    1*1卷积层  ->  3*3卷积层，填充1       ->  通道合并层
#            1*1卷积层  ->  5*5卷积层，填充2
#            3*3最大池层，填充1  ->  1*1卷积层

class Inception(nn.Module):
    # `c1`--`c4` 是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最⼤汇聚层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# GoogLeNet模型
# 7*7卷积层  ->  3*3最大池层  ->  1*1卷积层  ->  3*3卷积层  ->  3*3最大池层  ->
# 2个Inception  ->  3*3最大池层  ->  5个Inception  ->  3*3最大池层  ->
# 2个Inception  ->  全局平均池层  ->  全连接层

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten())
# 所有的超参数都是试出来的
net = nn.Sequential(b1,b2,b3,b4,b5, nn.Linear(1024, 10))


x = torch.rand(size=(1,1,96,96))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)


lr, num_epochs, batch_size = 0.1, 10, 64
trans = [torchvision.transforms.ToTensor()]
trans.insert(0, torchvision.transforms.Resize(96))
trans = torchvision.transforms.Compose(trans)

mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

d2l.train_ch6(net, train_iter, test_iter, num_epochs,
            lr, d2l.try_gpu())



# Inception后续有各种变种，包括使用批量规范化，使用特殊的卷积层，使用残差连接等