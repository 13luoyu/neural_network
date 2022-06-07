import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 使用注意力进行自然语言推断
# 预训练模型：GloVe
# 架构：MLP、Attention


# 我们将一个文本序列中的词元和另一个文本序列中的每个词元对齐（使用Attention），
# 之后比较对其的词元，并聚合比较的结果，表示前提和假设之间的逻辑关系


# 对齐（注意）
# I do need sleep 和 I am tired，我们希望将两句中的I对其，将tired和sleep或need sleep对其
# 我们使用注意力描述上述过程，假设A=(a1,a2,...)，推断B=(b1,b2,...)那么注意力权重计算为
# wij = f(ai)T * f(bj)，f为mlp()函数定义的多层感知机
# 对其结果中，前提中索引为i的词元对齐：βi = 求和(j=1 to n) (softmax(eij) * bj)
# 假设中索引为j的词元对齐：αj = 求和(i=i to m) (softmax(eij) * ai)

def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

# 软对齐，使用加权平均
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B):
        f_A = self.f(A)  # f_A,f_B形状为(batch_size, num_steps, num_hiddens)
        f_B = self.f(B)
        # (batch_size, A的词元数, B的词元数)
        e = torch.bmm(f_A, f_B.permute(0,2,1))
        # beta的形状：（批量大小，序列A的词元数，embed_size），
        # 意味着序列B被软对齐到序列A的每个词元(beta的第1个维度)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # alpha的形状：（批量大小，序列B的词元数，embed_size），
        # 意味着序列A被软对齐到序列B的每个词元(alpha的第1个维度)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


# 比较
# 上面的对齐（注意）已经确定了"need"和"sleep"与"tired"对齐，那么需要将它们进行比较
# 比较先将来自一个序列词元的连接，和来自另一个序列的对齐送入函数g
# vA,i = g([ai, βi]), i=1 to m
# vB,j = g([bj, αj]), j=1 to n
# vA,i是，所有假设中的词元与前提中词元i软对齐，再与词元i的比较
# vB,j是，所有前提中的词元与假设中词元j软对齐，再与词元j的比较

class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B, beta, alpha):
        # V_A为(批量数，A长度，num_hiddens)，表示A和与A对齐的词元的比较
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


# 聚合，现在我们有两组比较向量vA和vB，我们聚合它们推断逻辑关系
# 我们首先求和这两组向量，然后将求和结果放入函数h，或的分类结果

class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)  # 输入就是两维的，flatten与否都行
        self.linear = nn.Linear(num_hiddens, num_outputs)
    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim=1)  # (batch_size, num_hiddens)，将所有词的比较加起来
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat

# 模型
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)

lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
d2l.plt.show()




#@save
def predict_snli(net, vocab, premise, hypothesis):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.']))




