import math
import os
import random
import torch
from d2l import torch as d2l
# 用于预训练词嵌入的数据集

#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """将PTB数据集加载到文本行的列表中"""
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
print(f'# sentences数: {len(sentences)}')
print(sentences[0])
vocab = d2l.Vocab(sentences, min_freq=10)
print(f'vocab size: {len(vocab)}')

# 下采样
# 文本数据中the、a、in等高频词，他们出现次数多，但提供的有用信息很少，不如实体词提供更多信息。
# 此外，大量（高频）单词训练速度慢。
# 因此，当训练词嵌入模型时，对高频单词下采样，数据集中每个词wi有概率被丢弃，概率为P(wi) = max(rand(0,1)-根号下(t/f(wi)), 0)
# 其中f(wi)为wi的出现次数与总词数的比率，t为超参数（1e-4），只有f(wi)>t，高频词wi才被丢弃
def subsample(sentences, vocab):
    # 去掉<unk>
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    # 统计出现次数
    counter = d2l.count_corpus(sentences)
    # 总共有词数
    num_tokens = sum(counter.values())

    def keep(token):
        return (random.uniform(0,1) < math.sqrt(1e-4/counter[token]*num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)

subsampled, counter = subsample(sentences, vocab)
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence', 'count',
                            sentences, subsampled)
d2l.plt.show()
# 如图所示，下采样通过删除高频词缩短句子

def compare_counts(token):
    return (f'"{token}"的数量: 之前={sum(l.count(token) for l in sentences)}, '
            f'之后={sum(l.count(token) for l in subsampled)}')
print(compare_counts('the'))  # 高频词，舍去很多
print(compare_counts('join'))  # 低频词，几乎完全保留
# 词映射到索引
corpus = [vocab[line] for line in subsampled]
print(subsampled[:3], corpus[:3])



def get_centers_and_contexts(corpus, max_window_size):
    """从coupus中提取所有中心词和上下文词，它随机采样1到max_window_size的整数为窗口大小"""
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line  # 将line每个词都当做中心词，然后分别求每个词的上下文词
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i-window_size), min(len(line), i+1+window_size)))
            indices.remove(i)  # 排除中心词本身
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
# 演示
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('数据集', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心词', center, '的上下文词是', context)

all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
print(f'# “中心词-上下文词对”的数量(上下文词数): {sum([len(contexts) for contexts in all_contexts])}')

# 负采样
class RandomGenerator:
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights)+1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果，供选取。如果选完了，重新采样
            # random.choices(population, weights, cum_weights, k)
            # 从population集群中选出k个数，如果设置了weights，就规定了每个数被选中的相对权重
            # cum_weights为累加权重，即一个数权重=前一个数权重+本数相对权重
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

# 在索引1，2，3中绘制10个随机变量，采样概率分别是2/9, 3/9和4/9
generator = RandomGenerator([2, 3, 4])
print([generator.draw() for _ in range(10)])


# 对于一对中心词和上下文词，抽取K个（5个）噪声词，根据论文，将噪声词的采样概率P(w)设置为其在字典中的相对概率，幂为0.75
def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样的噪声词"""
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:  # len(contexts)个中心词，抽取len()*K个噪声词
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)



def batchify(data):
    """返回带有负采样的跳远模型小批量样本，输入为中心词、上下文词、噪声词"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    # 中心词，上下文词和噪声词，掩掉填充的词，区分上下文词和噪声词
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return torch.tensor(centers).reshape((-1,1)), torch.tensor(contexts_negatives),\
        torch.tensor(masks), torch.tensor(labels)

x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)


def n_gram(sentences, l1=3, l2=6):
    assert l1 < l2
    sub_tokens = []
    for sentence in sentences:
        for token in sentence:
            token = '<' + token + '>'
            sub_tokens.append(token)
            for i in range(l1, l2+1):
                begin, end = 0, i
                if end == len(token):
                    break
                while end <= len(token):
                    sub_token = token[begin:end]
                    sub_tokens.append(sub_token)
                    begin += 1
                    end += 1
    return sub_tokens  # 去重
x = [['i', 'love', 'shanghai'], ['i', 'live', 'in', 'shandong']]
print(n_gram(x))


# 整合
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    # fasttext新内容：将词分解为n=3-6的子词
    sub_tokens = n_gram(sentences)
    one_dim_sentences = [item for sl in sentences for item in sl]
    vocab = d2l.Vocab(sub_tokens + one_dim_sentences, min_freq=10)
    # 词表很大，但后面的采样生成词还是从原始词中
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        def __getitem__(self, item):
            return (self.centers[item], self.contexts[item], self.negatives[item])
        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                            collate_fn=batchify)
    return data_iter, vocab, one_dim_sentences

data_iter, vocab, all_tokens = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break




