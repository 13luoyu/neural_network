import torch
from d2l import torch as d2l
from torch import nn


# 深度循环神经网络，拥有多个隐藏层
# 每个隐状态都连续地传递给当前层的下一个时间步和下一层的当前时间步

# 第l层第t步的隐状态的表达式为 Ht(l)=σ(Ht(l-1)*Wxh(l) + Ht-1(l)*Whh(l) + Bh(l))
# 深度循环神经网络使用多个隐藏层来获得更多的非线性性质


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2  # 2层
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()