import torch
import torchvision
from torch import nn
from d2l import torch as d2l



# LeNet是早期成功的神经网络，用于识别MNIST手写数字数据集
# LeNet先使用卷积层来学习图片空间信息，使用池化层降低图片敏感度
# 然后使用全连接层来转换到类别空间

# 32 * 32 image
# convolution:  6 * 28 * 28  C1 feature map
# pooling:      6 * 14 * 14  S2 feature map
# convolution:  16 * 10 * 10 C3 feature map
# pooling:      16 * 5 * 5   S4 feature map
# full:         120
# full:         84
# full:         10
from torch.utils import data


class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=(5,5), padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=(5,5)),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

x = torch.rand(size=(1,1,28,28), dtype=torch.float32)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:\t', x.shape)


batch_size = 256

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

lr, num_epochs = 0.9, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()














