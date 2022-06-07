# NLP中的迁移学习
# 使用于训练好的模型来抽取词、句子特征，不更新与训练好的模型，而是构造新的网络抓取新任务需要的信息

# BERT：基于微调的NLP模型，预训练的模型抽取了足够的信息，新的任务只需要增加一个简单的输出层
# BERT是一个只有编码器的transformer，有两个版本
# Base版本：blocks=12, hidden size=768, heads(多头注意力)=12, parameters=110M
# Large版本: blocks=24, hidden size=1024, heads=16, parameters=340M
# 在大规模数据上训练（3B（三十亿）以上）

# BERT预训练：
# 对输入的修改：1每个样本是一个句子对：<cls> this movie is great <sep> i like it <sep>
# 2加入额外的片段嵌入，用以区分两个句子；
# 3位置编码可学习
# BERT预训练任务：
# 1Transformer编码器是双向的，而标准语言模型要求单向（前面不能看到后面的词）
# BERT随机掩蔽词元并使用来自双向上下文的词元以自监督的方式预测掩蔽词元
# BERT是带掩码的语言，每次随机将一些词元（15%）替换为<mask>
# 实际上，BERT的做法是：对选中的词元，80%的时间替换为<mask>，10%的时间替换为随机词元，10%的时间不变
# 2下一句子预测，预测一个句子对中两个句子是不是相邻的
# 训练样本中，50%相邻，50%不相邻


import torch
from torch import nn
from d2l import torch as d2l
# bert实现

def get_tokens_and_segments(token_a, token_b=None):
    """获得bert的输入，包含两部分，tokens为词组，segments标识第一个单词和第二个单词"""
    tokens = ['<cls>'] + token_a + ['<seq>']
    segments = [0] * (len(token_a) + 2)
    if token_b is not None:
        tokens += token_b + ['<seq>']
        segments += [1] * (len(token_b) + 1)
    return tokens, segments
    # segments是将<cls>第一句<seq>标记为0，将第二句<seq>标记为1

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)  # 将vocab_size个词编码
        self.segment_embedding = nn.Embedding(2, num_hiddens)  # 将输入（0或1）编码
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,  # max_len为位置编码最大长度(也就是序列x的最大长度)
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        # 词元嵌入，段嵌入（段嵌入用以区分两句话）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]  # 位置编码
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


# normalized_shape
# 如果传入整数，比如4，则被看做只有一个整数的list，此时LayerNorm会对输入的最后一维进行归一化，这个int值需要和输入的最后一维一样大。
#
# 假设此时输入的数据维度是[3, 4]，则对3个长度为4的向量求均值方差，得到3个均值和3个方差，分别对这3行进行归一化（每一行的4个数字都是均值为0，方差为1）；LayerNorm中的weight和bias也分别包含4个数字，重复使用3次，对每一行进行仿射变换（仿射变换即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
# 如果输入的是个list或者torch.Size，比如[3, 4]或torch.Size([3, 4])，则会对网络最后的两维进行归一化，且要求输入数据的最后两维尺寸也是[3, 4]。
#
# 假设此时输入的数据维度也是[3, 4]，首先对这12个数字求均值和方差，然后归一化这个12个数字；weight和bias也分别包含12个数字，分别对12个归一化后的数字进行仿射变换（仿射变换即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
# 假设此时输入的数据维度是[N, 3, 4]，则对着N个[3,4]做和上述一样的操作，只是此时做仿射变换时，weight和bias被重复用了N次。
# 假设此时输入的数据维度是[N, T, 3, 4]，也是一样的，维度可以更多。
# 注意：显然LayerNorm中weight和bias的shape就是传入的normalized_shape。
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print(encoded_X.shape)  # (batch_size, 序列长, num_hiddens)



class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        # X为BERTEncoder的输出，pred_positions为哪些地方要去预测(batch_size, 要预测的地方数)
        # 这些要去预测的地方最初输入为<mask>，x为其编码
        num_pred_positions = pred_positions.shape[1]  # 一个batch中要去预测的地方数
        pred_positions = pred_positions.reshape(-1)  # batch_size*num_pred_positions
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]），就是将batch_idx重复为和pred_positions同样的形状
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]  # 要预测的X，形状为(总预测地方数,num_hiddens)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))  # (batch_size, num_pred_positions, num_hiddens)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
print(mlm_Y_hat.shape)  # (batch_size, num_pred_positions, vocab_size)

# 损失使用交叉熵函数计算
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')  # reduction=none表示loss为所有损失和，
# reduction=mean(默认)表示loss为所有节点平均损失即
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))  # loss(batch_size*num_pred_positions*vocab_size, batch_size*num_pred_positions)
print(mlm_l.shape)  # batch_size * num_pred_positions







class NextSentencePred(nn.Module):
    """BERT的下一句预测任务，二分类模型"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)

# 由于Transformer编码器中的自注意力，特殊词元“<cls>”的BERT表示已经对输入的两个句子进行了编码
# 使用它作为输入预测
encoded_X_cls = encoded_X[:, 0, :]
# NSP的输入形状:(batchsize，num_hiddens * 序列长)
nsp = NextSentencePred(encoded_X_cls.shape[-1])
nsp_Y_hat = nsp(encoded_X_cls)
print(nsp_Y_hat.shape)  # batch_size, 2

nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
print(nsp_l.shape)







class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)

        # 掩蔽语言模型用
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        # 预测下一句用
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat