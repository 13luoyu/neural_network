import torch
import math
from torch import nn
from d2l import torch as d2l
import pandas as pd


# Transformer架构，基于纯注意力和编码器-解码器架构来处理序列对
# 编码器和解码器都有n个transformer块
# 每个块里使用多头自注意力、基于位置的前馈网络，和层归一化

# 多头注意力，对于同一query，key，value，我们希望抽取不同的信息（注意力输出不同）
# 多头注意力使用h个独立的注意力池化，然后合并各个头输出得到最终输出

# decoder中有掩码的多头注意力
# 解码器中对序列的一个元素输出时，不应该考虑该元素之后的元素
# 解决办法是将计算xi的输出时，假装当前序列长为i，设置valid_lens实现

# 前馈神经网络
# 将输入形状由(b,n,d)变为(bn,d)，之后经过两个全连接层，再由(bn d')变为(b,n,d')

# 层归一化
# 批量归一化对每个特征/通道维度进行归一化，然而，NLP中序列长度会变化，导致序列越长，每个值越小的问题
# 层归一化对每个样本里的元素进行归一化（同一时间步的所有batch归一化）

# 信息传递
# 编码器输出y1，...，yn，将其作为价码中第i个Transformer块的多头注意力的key和value，
# 同注意力的query则来自目标序列
# 意味着编码器和解码器中块的个数和输出维度都是一样的（输入=输出）

# 预测第t+1个输出时，解码器中输入前t个预测值，其中前t个预测值作为key和value，第k个预测值最为query




# 多头注意力实现
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)  # 点注意力分数
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  # 多头注意力，使用了相同的WqWk和Wv，与理论不符
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # 输入形状(batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens / num_heads)
        # 输出形状(batch_size * num_heads, 查询个数, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        # 形状(batch_size, 查询个数，num_hiddens)
        return self.W_o(output_concat)



def transpose_qkv(X, num_heads):
    # 为了多头注意力的并行计算而改变形状
    # 输入X形状为(batch_size, 查询或者“键－值”对的个数, num_hiddens)
    # 下面X的形状为(batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 下面X的形状为(batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0,2,1,3)
    # 输出X形状为(batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    # 逆转transpose_qkv的操作
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                               num_heads, 0.5)
attention.eval()
batch_size, num_queries, num_kvpairs, valid_lens = 2,4,6,torch.tensor([3,2])
x = torch.ones((batch_size, num_queries, num_hiddens))
y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(x,y,y,valid_lens).shape)




# Transformer实现
# 基于位置的前馈网络
# transformer模型中基于位置的前馈网络使用同一个多层感知机，作用是对所有序列位置的表示进行转换。
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        # 这里x为三维的，pytorch会将x前面的维度合为1个维度
        return self.dense2(self.relu(self.dense1(x)))

ffn = PositionWiseFFN(4,4,8)
ffn.eval()
print(ffn(torch.ones((2,3,4))))  # 只对序列表示的维度进行改变


# 对比batchnorm和layernorm
ln = nn.LayerNorm(2)  # 每个layer均值方差相同
bn = nn.BatchNorm1d(2)  # 每个batch均值方差相同
x = torch.tensor([[1,2],[2,3]], dtype=torch.float32)
print('layer norm: ', ln(x), '\nbatch norm: ', bn(x))

# 使用残差连接和层归一化
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

add_norm = AddNorm([3,4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2,3,4)), torch.ones((2,3,4))).shape)
# 不会改变形状


# 编码器中的一个层
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout,
                                            bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, x, valid_lens):
        y = self.addnorm1(x, self.attention(x,x,x,valid_lens))
        return self.addnorm2(y, self.ffn(y))

x = torch.ones((2,100,24))
valid_lens = torch.tensor([3,2])
encoder_blk = EncoderBlock(24,24,24,24,[100,24],24,48,8,0.5)
encoder_blk.eval()
# 输出和输入形状一样
print(encoder_blk(x, valid_lens).shape)


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)  # 位置编码
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, x, valid_lens, *args):
        # 因为位置编码值在-1和1之间，而embedding层输出和维度数d有关，d越大，每个值越小（d个数平方和为1）
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return x

encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
# 这里输入2维（批量数，序列长）
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
# Transformer编码器的输出是(batch_size, num_steps, num_hiddens)



class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:  # 将本步输入X和之前的输入X连接累计起来
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:  # 训练阶段，当前时间步不看之后（下一时间步）的输入
            # 这样做的原因是预测中并不知道整个序列，而训练时自注意机制却知道整个序列
            # 为了保持训练预测的一致而设计这种机制
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)



class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights



# 训练
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
d2l.plt.show()


# 测试
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')









