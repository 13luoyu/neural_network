import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data

# 丢弃法
# 好的模型需要对输入数据的扰动鲁棒。使用有噪音的数据等价于Tikhonov正则(平滑)，丢弃法是在层之间加入噪音
# 对x加入噪音得到x'，我们希望E[x']=x，其中E为期望
# x' = 0 (概率p), x/(1-p) otherwise
# 通常将丢弃法作用于隐藏全连接层的输出上
# h = σ(w1x + b1), h' = dropout(h), o = w2h' + b2, y = softmax(o)
# 正则项只在训练中使用，用于更新模型参数，在测试中，应该不起作用

# 丢弃法将一些输出项随机置0来控制模型复杂度，最终取平均
# 常作用在多层感知机的隐藏层输出上
# 丢弃概率p是控制模型复杂度的超参数


def dropout_layer(x, dropout):
    """丢弃法"""
    assert  0 <= dropout <=1
    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return x
    mask = (torch.rand(x.shape) > dropout).float()  # (0,1)均匀分布
    return mask * x / (1.0-dropout)

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.lin1(x.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            h1 = dropout_layer(h1, dropout1)
        h2 = self.relu(self.lin2(h1))
        if self.training == True:
            h2 = dropout_layer(h2, dropout2)
        out = self.lin3(h2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)  # 4进程读
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()




print('-------------------------------------')

# 简洁实现

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(256,256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()






