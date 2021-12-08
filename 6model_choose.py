import math
import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch import nn
from d2l import torch as d2l

# 三种数据集：训练数据集（训练模型参数）、验证数据集（选择模型超参数）、测试数据集
# 验证数据集用于验证训练好坏，一般从训练数据中取部分训练，部分验证
# 测试数据集不能用于调参数，只能使用一次（不能反复测试，调参，那是验证数据集应该做的）
# 如果数据不多，使用k-折交叉验证，将训练数据分为k块，k次迭代，每次1块验证，其余训练，最后
# 报告l个验证集误差平均，常用k=10、5

# 过拟合和欠拟合
# 当数据较为简单，应该使用低容量模型，否则高容量产生过拟合。
# 当数据较为复杂，应该使用高容量模型，否则低容量模型产生欠拟合
# 模型容量是拟合各种函数的能力，低容量模型难以拟合训练数据，高容量模型能够记住所有数据
# 随着模型容量升高，训练误差逐渐降低，然而泛化误差（测试集）会先低后高，因此要选择一个合适的模型容量
# 因此，一个模型要考虑：1参数的个数，2参数值的选择范围


# y = 5 + 1.2x - 3.4(x^2)/2! + 5.6(x^3)/3! + b, b~N(0,0.01)
max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4]=np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train+n_test,1))
np.random.shuffle(features)  # 200*1, 200个x
ploy_feature = np.power(features, np.arange(max_degree).reshape(1,-1))
# ploy_feature 200 * 20, 每行为x^0, x^1,...,x^19
for i in range(max_degree):
    ploy_feature[:,i] /= math.gamma(i+1)  # 阶乘
labels = np.dot(ploy_feature, true_w)  # 矩阵乘法
labels += np.random.normal(scale=0.1, size=labels.shape)  # 干扰b

true_w, features, ploy_feature, labels = [torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, ploy_feature, labels]]

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和、样本数量
    for x,y in data_iter:
        out=net(x)
        y=y.reshape(out.shape)
        l=loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
    num_epochs = 400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]  # 4
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
        xlim=[1,num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch+1)%20 == 0:  # 每20次训练绘1次图
            animator.add(epoch+1, (evaluate_loss(net, train_iter, loss),
                evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# 正常拟合
train(ploy_feature[:n_train, :4], ploy_feature[n_train:,:4],
    labels[:n_train], labels[n_train:])
d2l.plt.show()


# 线性函数拟合（欠拟合），只从多项式特征中选择前两个维度（1和x，没有x^2, x^3）
train(ploy_feature[:n_train, :2], ploy_feature[n_train:,:2],
    labels[:n_train], labels[n_train:])
d2l.plt.show()

# 高阶多项式函数拟合（过拟合）
train(ploy_feature[:n_train, :], ploy_feature[n_train:, :],
    labels[:n_train], labels[n_train:], num_epochs=1500)
d2l.plt.show()


