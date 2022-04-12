import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 深的神经网络不一定会有更好的效果，有可能训练偏了。
# 解决办法是更深的网络包含原来的模型，这样结果至少不会更坏
# 残差块：f(x) = x + g(x)
#
# x    ->    g(x)(Weight layer, Activation function...)   ->   f(x)
#             ->        x(或者1*1卷积，调整通道数)
from torch.utils import data


class Residual(nn.Module):
    """残差块"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)  # 作用是改变通道数，使得x和y通道数一样
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y+=x
        return F.relu(y)

blk = Residual(3,3)
x = torch.rand(4,3,6,6)  # 四个维度分别是图片数量，每个图片的通道数，行，列
y = blk(x)
print(y.shape)

# 高宽减半，通道加倍
blk = Residual(3,6,use_1x1conv=True,strides=2)
y = blk(x)
print(y.shape)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """如果first_block为True，则不将高宽减半通道加倍，反之..."""
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True,
                                strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))

net = nn.Sequential(b1,b2,b3,b4,b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512,10))

x = torch.rand(size=(1,1,224,224))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)


lr, num_epochs, batch_size = 0.05, 10, 16
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
d2l.plt.show()


# 为什么restnet能训练1000层的模型
# 一般，对一层，有y=f(x), 其梯度为 dy/dw
# 对深度网络，y=g(f(x)), 其梯度为 dy/dw = dy/df * df/dw，这样的链式
# 很明显，上层训练好以后，dy/df会很小，如此，影响的整体梯度会越来越小，下层难以更新
# RestNet中，y=f(x) + g(f(x))，此时dy/dw = (1+dy/df) * df/dw
# 保证了下层梯度不会太小