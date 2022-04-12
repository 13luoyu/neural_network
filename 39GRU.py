import torch
from torch import nn
from d2l import torch as d2l


# GRU门控循环单元
# 背景：不是每个观察值都是同等重要，要在过去与现在之间有所取舍
# 更新门：能关注的机制；重置门：能遗忘的机制
# 重置门Rt=σ(XtWxr + Ht-1Whr + br)， 形状和状态Ht一样
# 更新门Zt=σ(XtWxz + Ht-1Whz + bz)
# 候选隐状态H~t = tanh(XtWxh + (Rt · Ht-1)Whh + bh)，·为点乘
# Rt属于(0,1)，若Rt=0，将上一次状态设置为0，遗忘过去状态；Rt=1，则将之前的状态全部拿出来。
# 隐状态Ht = Zt · Ht-1 + (1-Zt) · H~t
# Zt属于(0,1)若Zt=1，则表示忽略掉了当前元素，完全沿用过去状态，否则Zt=0，表示忽略过去的状态
# 综上，两个门控制了极端状态Ht范围为（只和Xt相关  ~  和Xt与Ht-1相关（RNN）  ~  只和Ht-1相关）


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outpus = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    w_xz, w_hz, b_z = three()  # 更新门
    w_xr, w_hr, b_r = three()  # 重置门
    w_xh, w_hh, b_h = three()  # 候选隐状态
    w_hq = normal((num_hiddens, num_outpus))
    b_q = torch.zeros(num_outpus, device=device)
    params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device))

def gru(inputs, state, params):
    w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        z = torch.sigmoid((x @ w_xz) + (h @ w_hz) + b_z)
        r = torch.sigmoid((x @ w_xr) + (h @ w_hr) + b_r)
        h_tilda = torch.tanh((x @ w_xh) + ((r * h) @ w_hh) + b_h)
        h = z * h + (1 - z) * h_tilda
        y = h @ w_hq + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (h,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()


print("------------------------------------------")


num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()




















