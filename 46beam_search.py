
# 束搜索
# 在seq2seq模型中，预测时我们使用的是贪心算法，下一个时间步的输入是上一个时间步输出中概率最大的那个，这可能会出现问题
# 穷举法时间复杂度太大，不能实现
# 束搜索保存最好的k个候选，在每个时间步，计算k*n（n为num_hiddens）个输出，然后选择最大的k个，如此下去
# 时间复杂度：k * num_hiddens * num_steps
# 到最后一个时间步，我们选择其路径上所有条件概率积最高的序列作为输出序列。
# p = 1/L^α * logP(y1,...,yL | c)，其中L为最终序列长度，α通常为0.75，因为长序列会有更多的对数项，需要补偿来平衡
# 假设L=10，5，P = 1/e^2, 1/e，p=0.1*log1/e^2, 0.2*log1/e = -0.2, -0.2，确实会补偿
