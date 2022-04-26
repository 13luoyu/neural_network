
# 双向循环神经网络，既关注正向顺序，也关注逆向的顺序
# 某个位置的词也可以取决于未来的上下文，填不一样的词
# Ht-> = X * W + Ht-1 * W + b
# Ht<- = X * W + Ht+1 * W + b
# Ht = [Ht->, Ht<-]
# Ot = Ht * W + b

# 它的主要作用是对句子进行特征提取、填空、文本分类、翻译等，它不适合推理，因为不知道未来是什么

import torch
from torch import nn
from d2l import torch as d2l

# 加载数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers,
                     bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()# 训练困惑度很小，但预测一定不准确
