import torch
from torch import nn
from d2l import torch as d2l

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)

# 因为自然语言没有高宽之分，都是一维时间步，所以用一维卷积
def corr1d(X, K):
    # 一维卷积操作，其中x和k都是一维的，注意：之前实现的都是二维卷积
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y

X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
print(corr1d(X, K))

def corr1d_multi_in(X, K):
    # 首先，遍历'X'和'K'的第0维（通道维）。然后，把它们加在一起
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])  # X加上了batch_size维度
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
print(corr1d_multi_in(X, K))


# 卷积神经网络模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes,
                 num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 平均时间汇聚层，经过后n个输入输出1个值
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))
    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs),
                                self.constant_embedding(inputs)),
                               dim=2)  # (batch_size, num_steps, 2*embed_size)
        embeddings = embeddings.permute(0,2,1)  # (批量数，通道数，时间步)
        # 多个卷积层，每层卷积，池化，激活，结果为(批量数，通道数，1)
        # 删除最后一个维度并沿通道维将其连接起来，
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(
            conv(embeddings))), dim=-1) for conv in self.convs])
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

# 加载预训练词向量
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False

lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
d2l.plt.show()

d2l.predict_sentiment(net, vocab, 'this movie is so great')
d2l.predict_sentiment(net, vocab, 'this movie is so bad')