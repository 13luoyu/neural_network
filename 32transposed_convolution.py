import torch
from torch import nn
from d2l import torch as d2l

# 卷积通常不会增大输入的高宽，要么不变，要么减半
# 转置卷积可以用来增大输入高宽

# Y[i:i+h, j:j+w] += X[i,j] * K
# input    kernel
#  0 1      0 1      0 0 0     0 0 1     0 0 0     0 0 0     0 0 1
#  2 3  *   2 3   =  0 0 0  +  0 2 3  +  0 2 0  +  0 0 3  =  0 4 6
#                    0 0 0     0 0 0     4 6 0     0 6 9     4 12 9

# 对于卷积 Y = X * W，可以对W构造一个V，使得卷积等价于矩阵乘法
# Y' = VX'，其中Y',X'为Y,X对应的向量版本, Y'长度n,X'长m，则V为n*m
# 转置卷积则等价于Y'=VTX'，Y'长m，X'长n，则VT为m*n
# 如果卷积将输入从(h,w)变到了(h',w')，同样超参数（核大小，步长，填充等）
# 的转置卷积则从(h',w')变到(h,w)

# 手动实现转置卷积
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
y = trans_conv(X, K)
print(y)

# 使用高级API
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
y = tconv(X)
print(y)

# 填充、步幅、多通道
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
y = tconv(X)
print(y)
# 这个结果与常规卷积不同，在转置卷积中，填充被应用于的输出
# （常规卷积将填充应用于输入）。 例如，当将高和宽两侧的填充数
# 指定为1时，转置卷积的输出中将删除第一和最后的行与列。

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
y = tconv(X)
print(y)
# 在转置卷积中，步幅被指定为中间结果（输出），而不是输入。
# 如果步幅为2，则上述运算为
#                 0 0 0 0     0 0 0 1     0 0 0 0     0 0 0 0     0 0 0 1
# 0 1  *  0 1  =  0 0 0 0  +  0 0 2 3  +  0 0 0 0  +  0 0 0 0  =  0 0 2 3
# 2 3     2 3     0 0 0 0     0 0 0 0     0 2 0 0     0 0 0 3     0 2 0 3
#                 0 0 0 0     0 0 0 0     4 6 0 0     0 0 6 9     4 6 6 9

# 通道数卷积和转置卷积没有区别
# 验证卷积后，使用转置卷积，可以变回来
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)


# 与矩阵变换的联系
X = torch.arange(9.0).reshape(3, 3)  # 3*3
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2*2
y = d2l.corr2d(X, K)  # 卷积，y=2*2
print(y)

def kernel2matrix(K):
    """前面说过Y=WX -> Y'=VX'，函数将W->V"""
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W
W = kernel2matrix(K)
print(W)
# 验证上述卷积结论
print(y == torch.matmul(W,X.reshape(-1)).reshape(2,2))

# 验证转置卷积结论
z = trans_conv(y,K)
print(z == torch.matmul(W.T,y.reshape(-1)).reshape(3,3))
