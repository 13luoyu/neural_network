import torch
from torch import nn
from d2l import torch as d2l

# 池化层
# 池化层返回窗口中的最大值或平均值
# 池化层的主要作用是缓解卷积层对位置的敏感性，以及降低对降采样的民构型
# 卷积层对位置敏感，如检测垂直边缘，如果发生偏移，输出的边缘也会发生变化
# 然而，我们需要一定程度的平移不变性

# 二位最大池化，返回滑动窗口的最大值
# 0 1 2
# 3 4 5  *   2*2 max pooling  =  4 5
# 6 7 8                          7 8

# 垂直边缘检测          卷积输出(1,-1)       1*2最大池化
#  1 1 0 0 0          0 1 0 0             1 1 0 0
#  1 1 0 0 0          0 1 0 0             1 1 0 0
#  1 1 0 0 0          0 1 0 0             1 1 0 0
#  1 1 0 0 0          0 1 0 0             1 1 0 0

#  1 1 0 0 0          0 1 0 0             1 1 0 0
#  0 1 1 0 0          0 0 1 0             1 1 1 0           <- 偏移
#  1 1 0 0 0          0 1 0 0             1 1 0 0
#  1 1 0 0 0          0 1 0 0             1 1 0 0
# 可以发现，第二列边界还是1

# 池化窗口大小，填充和步幅是超参数
# 没有可学习的参数
# 如果有多个通道，输入通道数 = 输出通道数

def pool2d(x, pool_size, mode='max'):
    """实现池化层的正向传播"""
    p_h, p_w = pool_size
    y = torch.zeros((x.shape[0] - p_h + 1, x.shape[1] - p_w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode == 'max':
                y[i, j] = x[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                y[i, j] = x[i:i + p_h, j:j + p_w].mean()
    return y

x = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
y = pool2d(x, (2,2))
print(y)

y = pool2d(x, (2,2), 'avg')
print(y)


print("-------------------------")

x = torch.arange(32, dtype=torch.float32).reshape(1,2,4,4)
print(x)

# 默认，深度学习框架中的步幅 = 池化窗口大小
MaxPool2d = nn.MaxPool2d(3)
y = MaxPool2d(x)
print(y)

# 自己指定窗口大小，填充，步幅
MaxPool2d = nn.MaxPool2d((2,3), padding=(1,1), stride=(2,3))
y = MaxPool2d(x)
print(y)



# 池化层在每个通道上单独运算
x = torch.cat((x,x+1), 1)
print(x)

MaxPool2d = nn.MaxPool2d(3, padding=1, stride=2)
y = MaxPool2d(x)
print(y)



