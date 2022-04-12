import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data


# 批量归一化
# 因为损失出现在最后，所以后面的层训练的快，前面的慢，这导致如果前面的层
# 变化，后面的即使训练快也得变，得学习多次，导致收敛很慢
# 我们希望在学习底部层的时候避免变化顶部层
# 一个可行的方法是改变每一层的输入分布，这样，即使底部调整，因为变化后的每层输入
# 仍然是在指定的分布，这样训练收敛会很快。
# 批量规范化层控制输入数据的均值和方差
# 初始，计算输入的均值μ和方差σ
# 在调整时，调整后的x = γ* (x-μ)/σ + β，其中γ为本层方差，β为本层均值
# γ和β为要学习的参数
# 批量归一化层可以作用在全连接层和卷积层以及激活函数的输入上
# 对于全连接层，作用在特征维上，（行维度，维度1）
# 对于卷积层，作用在通道维上，（输出通道维度，维度1）

# 总之，批量归一化固定小批量中的均值和方差，然后学习出适合的偏移和缩放
# 可以加快模型的收敛速度，同时起到正则化作用，但一般不改变模型精度（lr可以更大）

def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentntum):
    """输入，参数γ，参数β，全局均值，全局方差，用来防止除0的值，用来更新全局变量方差的值"""
    # 从所有批量中计算本层全局的均值方差，（而不是一个批量的均值方差）

    # 如果不是在算梯度（不是在训练）
    if not torch.is_grad_enabled():
        x_hat = (x-moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2,4)  # 2为全连接层，4为卷积层
        if len(x.shape) == 2:
            mean = x.mean(dim=0)
            var = ((x-mean)**2).mean(dim=0)  # 均值和方差都是按行求均值，结果行向量
        else:
            mean = x.mean(dim=(0,2,3), keepdim=True)  # 对输出通道的所有批量求均值，keepdim表示保留维度，结果为1*n*1*1
            var = ((x-mean)**2).mean(dim=(0,2,3), keepdim=True)
        x_hat = (x-mean) / torch.sqrt(var+eps)
        moving_mean = momentntum * moving_mean + (1.0 - momentntum) * mean
        moving_var = momentntum * moving_var + (1.0 - momentntum) * var
    y = gamma * x_hat + beta
    return y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    """BatchNorm层"""
    def __init__(self, num_features, num_dims):
        """如果为卷积层，num_dims=4，num_features=输出通道数
           如果为全连接层，num_dims=2，num_features=列数"""
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))  # nn.Parameter表示要训练的参数
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps = 1e-5, momentntum = 0.9)
        return y


# 应用BatchNorm于LeNet模型
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 64
trans = torchvision.transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
print(net[1].gamma)
print(net[1].beta)





print("--------------------------")

# 简洁实现
# nn.BatchNorm2d()和nn.BatchNorm1d()
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()