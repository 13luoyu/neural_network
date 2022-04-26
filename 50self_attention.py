import math
import torch
from torch import nn
from d2l import torch as d2l


# 自注意力机制，每个查询都会关注所有的键－值对并生成一个注意力输出。
# 即query，key，value为同一集合，yi = f(xi, (x1,x1),...,(xn,xn))


num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
print(attention.eval())

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)  # (batch_size, num_queries, num_hiddens)


# CNN: 时间复杂度O(knd^2)，k为核长度，n为序列长度，d为序列每个值的维度（通道数）；
# 可以并行，因为每个卷积操作是独立的；最大路径长度为O（n/k），最大路径长度指的是从输入最多经过几层得到一个输出
# RNN：时间复杂度O(nd^2)，不能并行，最大路径长度O(n)
# 自注意力：时间复杂度O(n^2d)，可以并行，最大路径长度O(1)
# 卷积神经网络和自注意力都拥有并行计算的优势， 而且自注意力的最大路径长度最短。
# 但是因为其计算复杂度是关于序列长度的二次方，所以在很长的序列中计算会非常慢。


# 位置编码，循环神经网络是逐个的重复地处理词元的， 而自注意力则因为并行计算而放弃了顺序操作。 为了使用序列的顺序信息，
# 我们通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。
# 位置编码可以通过学习得到也可以直接固定得到。 接下来，我们描述的是基于正弦函数和余弦函数的固定位置编码
# 定义：X∈R(n*d)为一个序列n个词元的d维嵌入表示，位置编码P和X形状相同，编码后结果为X+P!!!是+起来不是concat
# P(i,2j) = sin(i/10000^(2j/d))
# P(i,2j+1) = cos(i/10000^(2j/d))


#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # [0::2] = [from=0;from!=end;from+=2]
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
d2l.plt.show()







