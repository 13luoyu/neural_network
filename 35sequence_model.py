import torch
from torch import nn
from d2l import torch as d2l

# 时序模型，当前的数据和之前观察到的数据相关
# 对于观察到的x1,x2,...,xt，认为它们是不独立的随机变量: (x1,x2,...) ~ p(x)
# 则p(x) = p(x1)*p(x2|x1)*p(x3|x1,x2)*...*p(xt|x1,...,xt-1)
# 1自回归模型对之前的数据建模，p(xt|x1,...,xt-1)=p(xt|f(x1,...,xt-1))
# 上面说明对之前的数据训练一个模型，然后预测新的xt
# 2马尔科夫模型，假设当前数据只和过去τ个数据相关，则
# p(xt|x1,...,xt-1)=p(xt|xt-τ,...,xt-1)=p(xt|f(xt-τ,...,xt-1))
# 这样，就可以在有限的数据上训练一个MLP或线性回归模型
# 3潜变量模型，引入潜变量ht概括过去的信息ht=f(x1,...,xt-1)
# 之后，xt=f(xt-1, ht), ht+1=g(xt, ht)

T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)  # 1,2,...
# 正弦波+噪声
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1,1000], figsize=(6,3))
d2l.plt.show()

# 使用马尔可夫模型
tau=4
features = torch.zeros((T-tau, tau))  # 每次只关注前tau个数据
for i in range(tau):
    features[:, i] = x[i:T-tau+i]
labels = x[tau:].reshape((-1,1))
# features每4个元素得到一个labels，即xt=f(xt-1,xt-2,xt-3,xt-4)

batch_size, n_train = 16, 600  # 使用前600个数据训练
train_iter = d2l.load_array((features[:n_train],labels[:n_train]),
                            batch_size, is_train=True)

# 拥有两个全连接层的多层感知机
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def get_net():
    net = nn.Sequential(nn.Linear(4,10),nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights)
    return net
loss = nn.MSELoss()

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            l=loss(net(x),y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch+1}',
              f'loss: {d2l.evaluate_loss(net, train_iter, loss)}')

net=get_net()
train(net, train_iter, loss, epochs=5, lr=0.01)

# 预测，给出前τ个数据，预测下一个
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
          [x.detach().numpy(), onestep_preds.detach().numpy()],
         'time','x',legend=['data','1-step preds'],
         xlim=[1,1000], figsize=(6,3))
d2l.plt.show()

# 多步预测,给出前600个点，预测后400个，逐渐离谱，因为
# 每下一个预测都使用上一个预测（而不是上面的准确值），误差逐渐大
multistep_preds = torch.zeros(T)
multistep_preds[:n_train+tau] = x[:n_train+tau]
for i in range(n_train+tau,T):
    multistep_preds[i]=net(multistep_preds[i-tau:i].reshape((1,-1)))

d2l.plot([time, time[tau:], time[n_train+tau:]],
         [x.detach().numpy(),onestep_preds.detach().numpy(),
          multistep_preds[n_train+tau:].detach().numpy()],
         'time','x',legend=['data','1-step preds','multistep preds'],
         xlim=[1,1000],figsize=(6,3))
d2l.plt.show()



max_steps = 64

# features[i,j]表示当要预测未来j-tau个点时的预测序列的第i个输入
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):  # 大量计算，每个步长逐步计算
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)  # 给4个点，预测未来step个
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()