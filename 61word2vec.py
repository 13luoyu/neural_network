# 词嵌入，将单词映射到实向量的技术
# 独热向量不太好，独热向量假设词表大小为N，那么用一个N维向量表示，某个词只有1位为1，其余为0。独热向量不能表示
# 词之间的相似度
# word2vec将词映射到一个固定长度的向量，更好的表达词之间的相似性和类比关系。它有2种模型

# 跳元模型skip-gram，假设一个词可以用来在文本序列中生成周围的词，比如给定词loves，可以生成the man loves his son
# 生成序列的条件概率为P(the,man,his,son|loves)=P(the|loves)P(man|loves)P(his|loves)P(son|loves)
# 跳元模型中每个词用2个d维向量表示，对于索引为i的词，使用向量vi表示它作为中心词的向量，使用ui表示它作为上下文词的向量
# 因此，给定中心词wc，生成任何上下文词wo的概率为P(wo|wc)=exp(uo*vc) / Σ(i从0到V)exp(ui*vc)
# 其中V为词表索引集的大小。给定长度T的文本序列，对于上下文窗口m，跳元模型似然函数是在给定任何中心词的情况下生成所有上下文词的概率：
# P = 连乘(t = 1 to T) 连乘(-m <= j <=m, j!=0) P(wt+j | wt)
# 训练：损失函数为l = -求和(t = 1 to T) 求和(-m <= j <=m, j!=0) logP(wt+j | wt)
# 对词典中索引为i的词进行训练后，得到vi（作为中心词）和ui（作为上下文词）两个词向量。
# 在自然语言处理应用中，跳元模型的中心词向量通常用作词表示。

# 连续词袋模型CBOW，与跳元模型的区别在于，词袋模型是给出上下文词，推导中间词
# P(wc | wo1, ..., wo2m) = exp(1/2m*uc*(vo1+...+vo2m)) / Σ(i从0到V)exp(1/2m*ui*(vo1+...+vo2m))
# 训练时，损失函数为l = -求和(t = 1 to T)logP(wt | w(t-m), ..., w(t-1), w(t+1), ..., w(t+m))


# 近似训练：由于一个词表可能有成千上万个单词，因此softmax中分母过大，求和计算梯度更是困难。
# 为了解决，介绍两种近似训练方法：负采样和分层softmax

# 负采样：就是在上下文窗口外的所有节点，选一部分更新权重，而窗口内的则要全部更新
# 假设D是中心词wc，上下文词wo是否在一个上下文窗口中中，令 P(D=1|wc,wo) = σ(uo * vc)
# σ为sigmoid函数，考虑最大化联合概率P = 连乘(t=1 to T)连乘(-m<=j<=m,j!=0)P(D=1|wt,wt+j)
# 若只考虑正样本，则上概率最大时所有词向量无穷大，P=1，这显然是不行的
# 负采样意味着考虑一些不在窗口中的词，考虑K个，K为超参数，上式重写为
# P = 连乘(t=1 to T)连乘(-m<=j<=m,j!=0)P(wt+j|wt)，
# 则给定词wt生成wt+j概率为P(wt+j | wt) =
# P(D=1|wt,wt+j) * 连乘(k = 1 to K)P(D=0|wt,wk)，不仅考虑窗口内的，还要考虑窗口外的
# 使用it和hk表示词wt和噪声词wk在词表中的索引，则损失函数
# l = -logP(wt+j | wt) = ... = -logσ(uit+j * vit) - 求和(k=1 to K)logσ(-uhk*vit)

# 分层softmax
# 一棵二叉树树，叶子节点表示一个词，L(w)表示字w从根节点到叶节点路上的节点数（含两端），n(w,j)为路径上第j个点
# 其上下文字向量为un(w,j)，则条件概率近似为
# P(wo | wc) = 连乘(j=1 to L(wo)-1)σ(sgn(n(wo,j+1) == leftChild(n(wo,j))) * un(wo,j)*vc)
# 意思是，对wo从根节点开始看，如果路径上下一个节点是当前节点左孩子，连乘数为sigmoid(u当前节点 * vc)
# 否则连乘数为sigmoid(-u当前节点 * vc)
# 可以计算，基于词wc生成词表中所有其他词的概率和为1
# 树形结构，每次计算概率时时间复杂度为O(log(V))，比O(V)好



# 全局向量的词嵌入GloVe
# https://github.com/stanfordnlp/GloVe
# 跳元模型中，给定中心词wi，得到词wj的概率qij = exp(uj*vi) / 求和(k=0~V)exp(uk*vi)
# 考虑词wi可能出现多次，可以统计整个数据集中wi窗口内其他词的出现次数，次数多的在损失函数中应该体现
# 定义词wi的多重集Ci，假设wi出现2次，上下文词分别是kjmk和klkj，Ci={j,j,k,k,k,k,l,m}
# 其中元素权重依次为2,4,1,1，将多重集Ci中元素j的重数设为xij，则损失函数更新为
# -求和(i属于V)求和(j属于V)xij * log(qij)，比跳元模型多了个xij，令xi表示Ci中词的数量，pij表示生成
# 上下文词wj的条件概率xij/xi，则上损失函数重写为
# -求和(i属于V) xi 求和(j属于V)pij * log(qij)
# -求和(j属于V)pij * log(qij)表示全局语料统计的条件分布pij和模型预测的条件分布qij的交叉熵，
# 之后按xi加权。
# GloVe模型基于平方损失对上述描述做了3个优化：
# 1,为了计算，不考虑条件分布，而是直接令pij = xij，令qij = exp(uj*vi)，并取对数，则平方损失项为
# (logpij - logqij)^2 = (uj*vi - logxij)^2
# 2,为每个词wi添加两个标量模型参数：中心词偏置bi和上下文词ci
# 3,用权重函数h(xij)替换每个损失项的权重xi，h(x)值域在[0,1]，递增
# l = 求和(i属于V) 求和(j属于V) h(xij)*(uj*vi + bi + cj - logxij)^2
# i和j越接近，logxij项越大，要想l小，需要i和j更关联，即前面的式子更小
# 论文中讲述了另一种思路
# 对于词wj和wk，衡量它们和wi的关联度，若wj关联度高，则应该pij / pik > 1
# 使用函数建模该比值，为f(uj, uk, vi) = pij / pik
# 下面设计f，因为值是标量，所以要求f是标量函数，例如f(uj,uk,vi)=f((uj-uk)*vi)
# 此外，交换j和k，我们应该得到合理的结果，这要求f(x)f(-x)=1，指数函数满足这个性质
# f(uj,uk,vi) = exp(uj*vi) / exp(uk*vi) = pij / pik
# 令exp(uj*vi) = a*pij，a是常数，pij为给出多重集Ci后元素j出现的概率，即pij=xij/xi，取对数，有
# uj*vi = loga + logxij - logxi，我们使用中心偏置bi和上下文偏置cj拟合loga-logxi，则有
# uj*vi + bi + ci = logxij
# 对其进行加权平方误差度量，则得到了损失函数