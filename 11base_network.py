import torch
from torch import nn
from torch.nn import functional as F
import os

# 模型构造
net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

x=torch.rand(2,20)
print(net(x))

class MLP(nn.Module):  # MLP类继承Module
    """实现上述神经网络"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)  # 两个全连接层
        self.out = nn.Linear(256,10)

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

net=MLP()
print(net(x))

class MySequential(nn.Module):
    """实现nn.Sequential()"""
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x

net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(x))


class FixedHiddenMLP(nn.Module):
    """更灵活的神经网络计算"""
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20), requires_grad=False)  # 不计算梯度，不参与训练
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        x = self.linear(x)
        while x.abs().sum()>1:
            x/=2
        return x.sum()

net = FixedHiddenMLP()
print(net(x))


print("------------------------------------------")

# 参数管理
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
x=torch.rand(size=(2,4))
print(net(x))
print(net[2].state_dict())  # OrderedDict类，包含weight和bias
print(type(net[2].bias))  # Parameter类，可以训练的值
print(net[2].bias)
print(net[2].bias.data)  # 值
print(net[2].bias.grad == None)  # 梯度，因为还没训练，所以为None

# 访问所有参数
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['0.weight'])

# 可以查看网络构造
print(net)

# 内置初始化
def init_normal(m):  # 正态分布，m为module
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)  # 对于net所有神经网络，调用这个函数
print(net[0].weight.data)
print(net[0].bias.data)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)  # 全1
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data)
print(net[0].bias.data)

#对不同层使用不同初始化函数
def xavier(m):  # 见9numerical_stability，xavier均匀分布
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

net[2].apply(xavier)
print(net[0].weight.data)
print(net[2].weight.data)

# 参数绑定，希望两个层共享参数
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared)



# 自定义linear层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self,x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5,3)
print(dense.weight)

print(dense(torch.rand(3,5)))

print("--------------------------")
# 读写文件，保存训练好的神经网络
# 存储张量
x = torch.arange(4)  # 0 1 2 3
torch.save(x, 'temp')
x2 = torch.load('temp')
print(x2)

# 存储张量列表
y = torch.zeros(4)
torch.save([x,y], 'temp2')
x2, y2 = torch.load('temp2')
print(x2)
print(y2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net=MLP()
x=torch.randn(size=(2,20))
y=net(x)

torch.save(net.state_dict(), 'temp3')
clone = MLP()
clone.load_state_dict(torch.load('temp3'))
print(clone)
y_clone = clone(x)
print(y == y_clone)

os.remove('temp')
os.remove('temp2')
os.remove('temp3')





