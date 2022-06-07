# 应用： 情感分析（单文本）、自然语言推断（文本对）
# 架构： MLP， CNN， RNN， Attention
# 预训练：word2vec， GloVe， 子词嵌入， BERT


import torch
from torch import nn
from d2l import torch as d2l

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)

# 定义情感分析RNN
# 使用LSTM训练，并提取第一个和最后一个时间步的output，使用全连接层输出
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 好 or 坏
    def forward(self, inputs):
        # inputs为(batch_size, num_steps)
        embeddings = self.embedding(inputs.T)  # (num_steps, batch_size, embed_size)
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
        self.encoder.flatten_parameters()
        outputs, state = self.encoder(embeddings)  # (num_steps, batch_size, 2*num_hiddens)
        # 将LSTM第一个时间步（句子开头）和最后一个时间步连接起来
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)  # (batch_size, 4*num_hiddens)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)

x = torch.randint(0, len(vocab), (5, 10))
print(net(x).shape)  # (batch_size, 2)



# 加载预训练的词向量
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]  # vocab.idx_to_token是一个个token的数组
print(embeds.shape)  # vocab中所有词转为词向量，(len(vocab), 100)
# 预训练的词向量载入，并设置不训练
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

# 训练
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer,
               num_epochs, devices)
d2l.plt.show()

# 预测情感
def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

print(predict_sentiment(net, vocab, 'this movie is so great'))
print(predict_sentiment(net, vocab, 'this movie is so bad'))
