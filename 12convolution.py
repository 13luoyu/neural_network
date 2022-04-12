import torch
from torch import nn
from d2l import torch as d2l


# 从全连接层到卷积层
# 将输入和输出变换为矩阵，此时权重变形为四维张量
# h(i,j) = Σ(k,l) w(i,j,k,l) x(k,l)
# 令v(i,j,a,b) = w(i,j,i+a,j+b)
# h(i,j) = Σ(a,b) v(i,j,a,b) x(i+a,j+b)
# 1平移不变性：无论x如何平移（图像在哪里），v不应该改变（识别器不变）
# 令v(i,j,a,b) = v(a,b)，则 h(i,j) = Σ(a,b) v(a,b) x(i+a,j+b)
# 2局部性：当评估h(i,j)时，不应该使用远离x(i,j)的参数
# 当a, b > △时，v(a,b)=0
# h(i,j) = Σ(a,b=-△, △)v(a,b) x(i+a,j+b)

# 卷积层的目的是1降低模型中的训练参数数量，2提取图像的结构特征

# 卷积层
# 二维交叉相关
# input * kernel = output
# 0 1 2    0 1     19  25
# 3 4 5 *  2 3  =  37  43
# 6 7 8
# 其中 0*0 + 1*1 + 3*2 + 4*3 = 19
# 输入x: nh * nw
# 核w: kh * kw
# 偏差b
# 输出y: (nh-kh+1) * (nw-kw+1)
# y = x * w + b
# 核矩阵的大小是超参数


def corr2d(x, k):
    """计算二维互相关运算"""
    h,w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w] * k).sum()
    return y

x = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
k = torch.tensor([[0.0,1.0],[2.0,3.0]])
y = corr2d(x, k)
print(y)

class Conv2D(nn.Module):
    """二维卷积层"""
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 简单应用，检测图像中不同颜色的边缘
x = torch.ones((6,8))
x[:,2:6] = 0
print(x)
k = torch.tensor([[1.0, -1.0]])
y = corr2d(x,k)
print(y)
# 然而，它不能检测水平边缘
print(corr2d(x.T, k))

# 学习由x生成y的卷积
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)  # 输入1个通道，输出1个通道
x = x.reshape((1,1,6,8))
y = y.reshape((1,1,6,7))
for i in range(10):
    y_hat = conv2d(x)
    l = (y_hat - y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    print(conv2d.weight.grad)
    print(conv2d.weight.data)
    conv2d.weight.data -= 3e-2 * conv2d.weight.grad
    print(conv2d.weight.data)
    if (i+1)%2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')
print(conv2d.weight.data)







