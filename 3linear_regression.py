import random
import torch
from d2l import torch as d2l
from torch.utils import data  # 数据处理
from torch import nn  # 神经网络

# 回归，一般有多个输入，一个输出，损失定义为输出值和真实输出的差
# y = xw+b，x为向量，y为标量


# 生成数据
# y = xw + b + c  其中w是权重，b是附加值，c是噪声
def synthetic_data(w, b, num_examples):
    x = torch.normal(0,1,(num_examples, len(w)))  # 正态分布随机数
    y = torch.matmul(x,w)+b  # 矩阵乘法
    y += torch.normal(0,0.01,y.shape)  # 添加噪声
    return x, y.reshape(-1,1)

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# d2l.set_figsize()
# d2l.plt.scatter(features[:,1].numpy(),
#                 labels.numpy(), 1)
# d2l.plt.show()

# 随机选batch_size个样本
def data_iter(batch_size, features, labels):
    """每次调用该函数，返回batch_size个样本"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 打乱样本
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
# for x,y in data_iter(batch_size,features,labels):
#     print(x,'\n',y)
#     break




# 定义初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 定义模型
def linreg(x,w,b):
    return torch.matmul(x,w)+b
# 定义损失函数
def squared_loss(y_hat, y):
    """ (y'-y)^2/2 """
    return (y_hat - y.reshape(y_hat.shape))**2/2
# 定义优化算法
def sgd(params, lr, batch_size):
    """参数列表，学习率，样本数"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 这里要/batch_size，因为batch_size个样本计算了batch_size次梯度
            param.grad.zero_()

lr=0.03
num_epochs=3
net=linreg
loss=squared_loss

# 训练
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, features, labels):
        l = loss(net(x,w,b), y)  # 唯一计算梯度的部分
        # l.shape = (batch_size, 1)
        l.sum().backward()
        sgd([w,b], lr, batch_size)  # 不计算梯度
    with torch.no_grad():
        train_1 = loss(net(features, w, b), labels)
        print(f'epoch {epoch}, loss {float(train_1.mean()):f}')


# 训练值和真实值之间的差距
print(f'w: {true_w - w.reshape(true_w.shape)}')
print(f'b: {true_b - b}')


print('---------------------------------')


true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size=10
data_iter=load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

net=nn.Sequential(nn.Linear(2, 1))  # 线性神经网络
net[0].weight.data.normal_(0, 0.01)  # w=正态分布
net[0].bias.data.fill_(0)  # b=0
loss=nn.MSELoss()  # L2范数误差
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 优化算法
num_epochs=3

for epoch in range(num_epochs):
    for x,y in data_iter:
        l=loss(net(x), y)
        trainer.zero_grad()  #
        l.backward()  # 计算梯度
        trainer.step()  # 模型更新
    l=loss(net(features), labels)
    print(f'epoch {epoch}, loss {l:f}')

print(f'w: {net[0].weight.data}')
print(f'b: {net[0].bias.data}')





