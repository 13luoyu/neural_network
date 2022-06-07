import math
import torch
from torch import nn
from d2l import torch as d2l

# 我们在一个小的数据集上训练了一个word2vec模型，并使用它为一个输入词寻找语义相似的词

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)


# 回忆Embedding
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)  # 字典大小，每个向量维度
x = torch.tensor([[1,2,3],[4,5,6]])
print(embed(x).shape)


# 定义前向传播
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    # center为(batch_size,1)，contexts_and_negatives为(batch_size, max_len)
    # 这两个变量先转化为想来给你，然后矩阵点积，相当于P(wo|wc) = vc * uo
    # 返回形状为(batch_size, 1, max_len)
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0,2,1))
    return pred

pred = skip_gram(torch.ones((2,1), dtype=torch.long), torch.ones((2,4), dtype=torch.long),
          embed, embed)
print(pred.shape)

# 二元交叉熵损失函数，返回为(batch_size)
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask,
                                                             reduction='none')
        return out.mean(dim=1)

loss = SigmoidBCELoss()

pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1,1,1,1], [1,1,0,0]])
# 原来求平均都是/1个batch中所有元素数量，现在改为有几个有效元素/几
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))


# 几种损失函数，x为预测，y为真实标签
# nn.MSELoss(x,y) = (x-y)^2
# 二分类交叉熵损失，x为真实标签，y为预测
# nn.BCELoss(x,y) = - (x*log(y) + (1-x) * log(1-y))
# nn.BCEWithLogitsLoss(x,y) = nn.BCELoss(x, sigmoid(y))  上类中就是用的这个函数
# nn.CrossEntropyLoss(x,y) = -log(exp(x[y]) / 求和(i)(exp(x[i])))，相当于其中包含softmax


# 手动实现nn.functional.binary_cross_entropy_with_logits（实现的一般）
# 测试数据根据原理进行取反
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))
print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')


embed_size = 100
# 训练的就是nn.Embedding的权重w
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
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
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
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
# train(net, data_iter, lr, num_epochs)
# d2l.plt.show()

# 在训练word2vec模型之后，我们可以使用训练好模型中词向量的余弦相似度来从词表中找到与输入单词语义最相似的单词。
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 不包含输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])