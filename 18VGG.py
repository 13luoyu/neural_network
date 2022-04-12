import torch
import torchvision
from torch import nn
from d2l import torch as d2l


# VGG就是使用可重复使用的卷积块来构建深度神经网络
# 不同的卷积块个数和超参数得到不同复杂度的变种
from torch.utils import data


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

# 经典设计模式：高宽减半，通道数*2
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_channels, out_channels
        ))
        in_channels = out_channels

    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         nn.Linear(out_channels*7*7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096,4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096,10))

net=vgg(conv_arch)

x = torch.randn(size=(1,1,224,224))
for blk in net:
    x = blk(x)
    print(blk.__class__.__name__,'output shape:\t',x.shape)

# 更小的网络训练
ratio = 4  #  //表示除法向下取整
small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 64
# 将Fashion-MNIST图像分辨率扩为224*224
trans = [torchvision.transforms.ToTensor()]
trans.insert(0, torchvision.transforms.Resize(224))
trans = torchvision.transforms.Compose(trans)  # 作用：几个图像变换组合在一起,按照给出的transform顺序进行计算。

mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

d2l.train_ch6(net, train_iter, test_iter, num_epochs,
              lr, d2l.try_gpu())

