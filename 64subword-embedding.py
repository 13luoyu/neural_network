import collections

# 英语中一个动词有很多时态，word2vec和GloVe模型都是将其标识为不同的向量，没有考虑之间的关联
# fastText模型提出一种子词嵌入方法，一个子词是一个字符n-gram，每个中心词由子词向量之和表示
# 比如where，n=3，获得长度为3的子词：<wh, whe, her, ere, re>和特殊子词<where>，其中<和>分别
# 表示前缀和后缀
# 假设子词g的向量为zg，则作为中心词的词w的向量vw = 求和(g属于G)(zg)
# 其余部分同跳元模型

# 在fastText中，所有提取的子词都必须是指定的长度，例如3到6，因此词表大小不能预定义。
# 为了在固定大小的词表中允许可变长度的子词，我们可以应用一种称为字节对编码（Byte Pair Encoding，BPE）
# 的压缩算法来提取子词


# "_"是词尾符号，"[UNK]"未知符号
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']

# 词映射到数据集中的频率
raw_token_freqs = {"fast_":4, "faster_":3, "tall_":5, "taller_":4}
# 词中间加空格，同样是词映射到频率
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = freq
print(token_freqs)

def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()  # 分成一个个字母或字母组
        for i in range(len(symbols) - 1):  # 对每个字母（组），除了_
            pairs[symbols[i], symbols[i+1]] += freq  # 字母对频率统计
    return max(pairs, key=pairs.get)  # 返回频率最大的字母对

def merge_symbols(max_freq_pair, token_freqs, symbols):
    """将频率最大的字母（组）对合并为一个字母组"""
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        # 原来的a b，替换为ab
        new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs

num_merges = 10  # 10轮合并
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'合并# {i+1}:',max_freq_pair)
print(symbols)  # 列表symbols现在又包含10个从其他符号迭代合并而来的符号
print(token_freqs)  # 数据集中每个词被子词分隔

def segment_BPE(tokens, symbols):
    """函数尝试将单词从输入参数symbols分成可能最长的子词"""
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        while start < len(token) and start < end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append("[UNK]")
        outputs.append(' '.join(cur_output))
    return outputs

tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
# 可以看出，函数很好的提取了tallest的词根
