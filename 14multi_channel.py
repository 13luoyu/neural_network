import torch
from d2l import torch as d2l


# 输出通道数是卷积层的超参数
# 每个输入通道有独立的二维卷积核，所有通道结果'相加'得到一个输出通道结果
# 1 2 3
# 4 5 6  *  1 2  =  37 47
# 7 8 9     3 4     67 77
#                         +   =   56 72
# 0 1 2                           104 120
# 3 4 5  *  0 1  =  19 25
# 6 7 8     2 3     37 43
# 多个输入通道，一个输出通道，输入x:ci*nh*nw，核w:ci*kh*kw，输出y:mh*mw

# 多个输出通道，每个输出通道有独立的三维卷积核，每个核输出一个输出通道
# 输入x:ci*nh*nw，核w:co*ci*kh*kw，输出y:co*mh*mw
# 每个输出通道可以识别特定模式，如绿色通道中的一个点，斜向的一个边等
# 输入通道核'识别并组合'输入中的模式，其中输入是上一层输出通道的输出，识别出是什么，然后赋予不同的权重并组合相加

# kh=kw=1的卷积核，它不识别空间模式，只是融合通道
# 这相当于是输入形状为nhnw * ci，权重为co * ci的全连接层

# 二维卷积层
# y = x ★ w + b，其中y为3位，x为3维，w为4维，b为2维(co*ci)
# 计算复杂度为O(ci co kh kw mh mw)，对每个输出通道的每个像素，都要使用核计算，计算包含每个输入通道
# 计算复杂度非常高


def corr2d_multi_in(x,k):
    """多通道输入，单通道输出"""
    return sum(d2l.corr2d(xx,kk) for xx,kk in zip(x,k))
x = torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],
                  [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])  # 3*3*2
k = torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])  # 2*2*2
y = corr2d_multi_in(x,k)
print(y)

def corr2d_multi_in_out(x,k):
    """多通道输入和输出"""
    # 对于四维核的每个维度kk，kk*x，然后将每个维度的结果再连成新的维度
    # torch.stack(inputs, dim) -> Tensor，参数dim选择生成的维度
    return torch.stack([corr2d_multi_in(x,kk) for kk in k], 0)

k = torch.stack((k,k+1,k+2), 0)  # 输出通道，输入通道，高，宽
print(k)

y = corr2d_multi_in_out(x, k)
print(y)


print('----------------------------------------')

# 1*1卷积 = 全连接
def corr2d_multi_in_out_1x1(x,k):
    """全连接层的计算法"""
    c_i, h, w = x.shape
    c_o = k.shape[0]
    x = x.reshape((c_i, h*w))
    k = k.reshape((c_o, c_i))
    y = torch.matmul(k, x)
    return y.reshape((c_o, h, w))

x = torch.normal(0,1,(3,3,3))
k = torch.normal(0,1,(2,3,1,1))

y1 = corr2d_multi_in_out(x, k)
y2 = corr2d_multi_in_out_1x1(x, k)
print(y1)
print(y2)

