import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 网络中的网络
# 卷积层后的第一个全连接层参数量巨大：通道数*图片高*图片宽*下一层输入数
# NiN块，是一个卷积层，后面跟着两个1*1的卷积层（相当于全连接层）
# NiN架构，交替使用NiN块和步幅为2的最大池化层（每次高宽减半）
# 最后使用全局平均池化从得到输出（输出通道数就是类别数）
from torch.utils import data


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

net = nn.Sequential(nin_block(1, 96, kernel_size=11, strides=4, padding=0),
                    nn.MaxPool2d(3, stride=2),
                    nin_block(96,256,kernel_size=5,strides=1,padding=2),
                    nn.MaxPool2d(3, stride=2),
                    nin_block(256,384,kernel_size=3,strides=1,padding=1),
                    nn.MaxPool2d(3, stride=2),
                    nn.Dropout(0.5),
                    nin_block(384,10,kernel_size=3,strides=1,padding=1),
                    nn.AdaptiveAvgPool2d((1,1)),  # target output size of n*m*k*l -> n*m*1*1，kernel_size和stride自动计算
                    nn.Flatten())

x = torch.rand(size=(1,1,224,224))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)

lr, num_epochs, batch_size = 0.1, 10, 64
# 将Fashion-MNIST图像分辨率扩为224*224
trans = [torchvision.transforms.ToTensor()]
trans.insert(0, torchvision.transforms.Resize(224))
trans = torchvision.transforms.Compose(trans)

mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

d2l.train_ch6(net, train_iter, test_iter, num_epochs,
            lr, d2l.try_gpu())
