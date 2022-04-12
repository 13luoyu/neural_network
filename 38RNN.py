import torch
from d2l import torch as d2l
import math
from torch import nn
from torch.nn import functional as F


# 证明 x*w_xh + h*w_hh = x和h的拼接 * w_xh和w_hh的拼接
x, w_xh = torch.normal(0,1,(3,1)), torch.normal(0,1,(1,4))
h, w_hh = torch.normal(0,1,(3,4)), torch.normal(0,1,(4,4))
print(torch.mm(x,w_xh) + torch.mm(h,w_hh))
print(torch.mm(torch.cat((x,h),1), torch.cat((w_xh,w_hh),0)))

batch_size, num_steps = 32, 35  # 每个批量大小，一个样本的词数（xt的t的最大值）
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 独热编码one_hot函数将分别生成多个给定长度的张量，其中指定位为1，其余为0
print(F.one_hot(torch.tensor([0,2]), len(vocab)))
# 小批量的形状为（批量大小，时间步数），为了方便计算，将其转置
x = torch.arange(10).reshape((2,5))  # 2批量5时间步
print(F.one_hot(x.T,28).shape)  # 生成5*2*28张量，这样每个时间步可以取所有批量值，方便计算

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size  # 输入输出都为28，因为输入是一个字符，输出也是一个字符
    def normal(shape):  # torch.randn标准正态分布(0~1)
        return torch.randn(size=shape, device=device)*0.01
    # 隐藏层参数
    w_xh = normal((num_inputs, num_hiddens))
    w_hh = normal((num_hiddens, num_hiddens))  # 关键！与MLP的区别在于记录了之前的隐状态
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)
    # 隐藏状态放在tuple里，因为有的模型会有两个隐藏状态

def rnn(inputs, state, params):
    # 在一个时间步内计算隐藏状态和输出
    # input为三维(时间步数，批量数，特征数)，state为元组，每个元素为二维(批量数，隐藏层大小)
    w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        h = torch.tanh(torch.mm(x, w_xh) + torch.mm(h, w_hh) + b_h)
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h,)

class RNNModelScratch:
    """循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        # 词表的大小，隐藏层大小，设备，初始化参数函数，初始化状态函数，forward函数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):  # 获得初始化状态
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(),
                      get_params, init_rnn_state, rnn)
# x = torch.arange(10).reshape((2,5))  # 2批量5时间步
state = net.begin_state(x.shape[0], d2l.try_gpu())
y, new_state = net(x.to(d2l.try_gpu()), state)
print(y.shape)  # 10*28，10是因为y将每个时间步在行上（dim=0）连接起来
print(len(new_state))  # 1
print(new_state[0].shape)  # 2*512

# 定义预测函数
def predict_ch8(prefix, num_preds, net, vocab, device):
    # 在`prefix`后面生成新字符
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # 第0个词的索引
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 将输入和状态输入，更新状态
        outputs.append(vocab[y])  # 将真实的下一步输出放入outputs
    for _ in range(num_preds):
        # 真正的预测
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

y = predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
print(y)

# 梯度剪裁
# 在上面定义的循环神经网络中，有35个时间步（num_steps），因此有35个矩阵乘法
# 这样会导致梯度爆炸问题。为了解决，梯度剪裁定义为 g = min(1, θ/||g||) * g
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if(norm > theta):
        for param in params:
            param.grad[:] *= theta / norm

# torch.detach()和detach_()
# detach()
# 官方文档中，对这个方法是这么介绍的。
# 返回一个新的 从当前图中分离的 Variable。
# 返回的 Variable 永远不会需要梯度
# 如果 被 detach 的Variable  volatile=True， 那么 detach 出来的 volatile 也为 True
# 还有一个注意事项，即：返回的 Variable 和 被 detach 的Variable 指向同一个 tensor
# detach_()
# 官网给的解释是：将 Variable 从创建它的 graph 中分离，把它作为叶子节点。
# 从源码中也可以看出这一点
# 将 Variable 的grad_fn 设置为 None，这样，BP 的时候，到这个 Variable 就找不到 它的 grad_fn，所以就不会再往后BP了。
# 将 requires_grad 设置为 False。这个感觉大可不必，但是既然源码中这么写了，如果有需要梯度的话可以再手动 将 requires_grad 设置为 true


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for x, y in train_iter:
        if state is None or use_random_iter:  # 将state初始化为0，当且仅当刚刚开始训练一个batch，
            # 或者batch之间是打乱的，前后之间没有连续
            state = net.begin_state(batch_size=x.shape[0], device=device)
        else:  # 若两个batch之间是连续的，或在训练一个batch的中间
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()  # 指的是这个state计算梯度不影响
            else:
                for s in state:
                    s.detach_()
        y = y.T.reshape(-1)  # y为（时间步数 * 批量数）
        x, y = x.to(device), y.to(device)
        y_hat, state = net(x, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)  # 梯度裁剪
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l*y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()  # 使用指数计算损失

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
d2l.plt.show()



# 随机抽样的方法
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps, use_random_iter=True)
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
d2l.plt.show()


print("-----------------------------------------")

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)  # 输入输出大小，隐藏层大小
state = torch.zeros((1, batch_size, num_hiddens))  # 上面的实现中state为元组，这里多加了一个维度
x = torch.rand(size=(num_steps, batch_size, len(vocab)))
y, state_new = rnn_layer(x, state)
print(y.shape)
print(state_new.shape)

# 上述nn.RNN仅包含隐藏层，我们还要创建一个输出层
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        # 如果RNN是双向的，num_directions为2，否则为1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        x = F.one_hot(inputs.T.long(), self.vocab_size)
        x = x.to(torch.float32)
        y, state = self.rnn(x, state)
        # 全连接层，首先将y形状改为（时间步数*批量大小，隐藏单元数），输出形状为（时间步数*批量大小，词表大小）
        output = self.linear(y.reshape((-1, y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):  # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions*self.rnn.num_layers,
                                batch_size,self.num_hiddens), device=device)
        else:  # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions*self.rnn.num_layers,
                                 batch_size,self.num_hiddens), device=device),
                    torch.zeros((self.num_directions*self.rnn.num_layers,
                                batch_size,self.num_hiddens), device=device))

device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller ', 10, net, vocab, device)

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()














