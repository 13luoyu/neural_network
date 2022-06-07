import math
import torch
from torch import nn
from d2l import torch as d2l

dataset = __import__("62_1glove-dataset")

# 我们在一个小的数据集上训练了一个word2vec模型，并使用它为一个输入词寻找语义相似的词

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = dataset.load_data_ptb(batch_size, max_window_size,
                                         num_noise_words)

def h(x, c=2, a=0.75):
    return (torch.min(x, torch.tensor(c).to(x.device)) / c) ** a

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    # center为(batch_size,1)，contexts_and_negatives为(batch_size, max_len)
    # 这两个变量先转化为想来给你，然后矩阵点积，相当于P(wo|wc) = vc * uo
    # 返回形状为(batch_size, 1, max_len)
    v, b1, c1 = embed_v(center)
    u, b2, c2 = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0,2,1)).reshape(contexts_and_negatives.shape)
    # pred:batch_size,embed_size, b1:batch_size,1, b2:batch_size,上下文和无关词数
    return pred + b1 + c2

# 二元交叉熵损失函数，返回为(batch_size)
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, inputs, target, xij, mask=None):
        # inputs包括uj*vi + bi + cj
        # 若uj为上下文中的词，xij>0，要求inputs大，否则，xij=0，要求inputs小
        # out = h(xij) * (inputs - torch.log(xij + 1)) ** 2  # 论文做法
        out = (inputs - torch.log(xij + 1)) ** 2  # +1是与论文不同的做法，体现的是若xij=0，我们希望inputs为0
        # mask掩蔽了填充词，保留上下文词和非上下文词的损失
        if mask is not None:
            out *= mask
        return out.mean(dim=1)


loss = SigmoidBCELoss()


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.b = nn.Parameter(torch.normal(0, 0.01, (num_embeddings,)))
        self.c = nn.Parameter(torch.normal(0, 0.01, (num_embeddings,)))

    def forward(self, x):
        # x(batch_size, 1) or (batch_size, n)
        y = self.embed(x)
        b = self.b[x.reshape(-1)].reshape(x.shape)
        c = self.c[x.reshape(-1)].reshape(x.shape)
        return y, b, c


embed_size = 100
# 训练的就是nn.Embedding的权重w
net = nn.Sequential(Embedding(num_embeddings=len(vocab),
                              embedding_dim=embed_size),
                    Embedding(num_embeddings=len(vocab),
                              embedding_dim=embed_size))

def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label, xij = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), xij, mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
d2l.plt.show()


# 在训练word2vec模型之后，我们可以使用训练好模型中词向量的余弦相似度来从词表中找到与输入单词语义最相似的单词。
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 不包含输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


get_similar_tokens('chip', 3, net[0].embed)
