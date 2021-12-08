import torch
from torch import nn
from d2l import torch as d2l

# 为了解决过度拟合问题，可以通过限制参数数目，或者限制参数值的选择范围来控制模型容量
# 权重衰退是后者
# 现在的损失函数为l(w,b) + λ/2 * ||w||^2(w的L2范数的平方)
# λ为超参数，控制了正则项的重要程度，如果=0，无作用，如果无穷，w->0，模型过于简单
# 下面解释对最优解的影响
# 为了使得新损失函数小，不仅要使得l(w,b)小，还要使得||w||小，这就要求所有w都变小，
# w对模型的作用减弱，模型容量减小。
# 参数更新法则
# 计算梯度
# d(l(w,b) + λ/2 * ||w||^2)/dw = dl/dw + λw
# w = (1-ηλ)w - η(dl/dw)
# 可以看到，在更新w时，衰减了原来的w的权重


# y = 0.05 + Σ0.01*xi + b, b~N(0,0.01^2)
# 训练数据集大小，测试数据集大小，输入维度，一次训练/测试样本大小
n_train, n_test, num_inputs, batch_size = 20,100,200,5
true_w, true_b = torch.ones((num_inputs, 1))*0.01,0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    """初始化参数"""
    w = torch.normal(0, 1, size=(num_inputs,1), requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w):
    """L2范数惩罚"""
    return torch.sum(w.pow(2))/2

def train(lambd):
    w,b=init_params()
    # 线性神经网络，L2损失函数
    net,loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
        xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 权重衰退
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)  # 优化
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

train(0)
d2l.plt.show()
train(3)
d2l.plt.show()


print('---------------------------')


def train_concise(wd):
    """使用torch库实现权重衰退"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()  # L2损失函数
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减，只有权重衰减
    trainer = torch.optim.SGD([  #!!!
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
        xlim=[5, num_epochs], legend=['train', 'test']) 
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

train_concise(0)
d2l.plt.show()
train_concise(3)
d2l.plt.show()