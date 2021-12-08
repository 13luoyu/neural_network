import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
import torchvision

# 感知机
# 给定输入x，权重w和偏移b，感知机输出o=σ(<w,x>+b)，其中σ为激活函数
# 例如σ(x)=1 if x>0, =-1 otherwise
# 早期的感知机等价于使用批量大小为1的梯度下降，
# 并且损失函数l(y,x,w)=max(0, -y(<w,x>+b))
# 因其不能拟合XOR函数，只能产生线性分割面，被遗弃

# 多层感知机
# 既然XOR一条分割线分割不了，就使用两条分割线，先学习两条分割线，然后学习XOR
# 输入层、隐藏层、输出层。隐藏层大小是超参数，隐藏层数量也是超参数。
# 输入x ∈ R^n，隐藏层w1∈R^m*n，b1∈R^m，输出层w2∈R^m*k，b2∈R^k
# h = σ(w1*x + b1)   o = w2.T*h + b2  T转置
# 激活函数σ必须是非线性函数，否则输出o的表达式仍然是线性函数

# sigmoid激活函数 sigmoid(x) = 1 / (1 + exp(-x)) 值域(0,1)
# tanh激活函数 tanh(x) = (1 - exp(-2x))/(1 + exp(-2x))  值域(-1,1)
# ReLU(rectified linear unit)激活函数 ReLU(x) = max(x,0)

# 多隐藏层
# h1 = σ(w1*x+b1)  h2 = σ(w2*h1+b2)  h3 = σ(w3*h2+b3)  o = w4*h3+b4
# 一般设置多隐藏层数目时，h1略大于x，然后逐渐减小为o的数目


batch_size = 256
trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)  # 4进程读
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4)
num_inputs = 784
num_outputs = 10
num_hiddens = 256

w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01)
b1 = torch.zeros(num_hiddens, requires_grad=True)
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01)

b2 = torch.zeros(num_outputs, requires_grad=True)
params = [w1, b1, w2, b2]

def relu(x):
    """ReLU激活函数"""
    a=torch.zeros_like(x)
    return torch.max(x,a)

def net(x):
    """感知机神经网络"""
    x=x.reshape(-1, num_inputs)
    h = relu(x @ w1 + b1)  # @矩阵叉乘，相当于torch.matmul()
    o = h @ w2 + b2
    return o

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()


print('--------------------------------------------------')

# 简洁实现
net = nn.Sequential(nn.Flatten(), nn.Linear(784,256),
                    nn.ReLU(), nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
