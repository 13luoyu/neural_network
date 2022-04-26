import torch
from torch import nn
from d2l import torch as d2l

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
d2l.plt.show()





# 注意力汇聚：Nadaraya-Watson核回归
# 给定成对的输入-输出数据集{(x1,y1),...,(xn,yn)}，如何学习f预测任意新输入x的输出y^=f(x)
# f(x) = Σ(i=1~n) (K(x-xi) / Σ(j=1~n)K(x-xj)) * yi，其中K为核函数
# 意思是对每个输入x，我们将其输出f(x)表示为yi的加权平均，将x和键xi的关系建模为注意力权重
# 考虑高斯核K(x) = 1/根号下(2Π) * exp(-u^2 / 2)
# f(x) = ... = Σ(i=1~n) softmax(-1/2 * (x-xi)^2) * yi


n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
# w_train为排好序的0-5的数组，_记录每个元素在原数组中的位置
def f(x):
    return 2 * torch.sin(x) + x ** 0.8

y_train = f(x_train) + torch.normal(0.0,0.5,(n_train,))
x_test = torch.arange(0,5,0.1)
y_truth = f(x_test)
n_test = len(x_test)  # 50

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])  # 测试集真实值和预测值曲线
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)  # 训练集分布点图
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
d2l.plt.show()

# 非参数注意力汇聚
x_repeat = x_test.repeat_interleave(n_train).reshape((n_test, n_train))
# 注意力权重
attention_weights = nn.functional.softmax(-(x_repeat-x_train)**2/2, dim=1)  # 形状为(n_test, n_train)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
d2l.plt.show()
# 查看注意力权重热力图
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs',
              ylabel='Sorted testing inputs')
d2l.plt.show()

# batch矩阵乘法bmm
x = torch.ones((2,1,4))
y = torch.ones((2,4,6))
print(torch.bmm(x,y).shape)  # (2,1,6)

weights = torch.ones((2, 10)) * 0.1  # 2个batch，meigebatch10个数
values = torch.arange(20.0).reshape((2, 10))  # 同上
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

# 带参数注意力汇聚
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        # w控制的是核函数平滑度

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
#  任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
d2l.plt.show()

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
d2l.plt.show()

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
d2l.plt.show()








