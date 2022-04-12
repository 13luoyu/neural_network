import torch
from torch import nn

# 填充和步幅
# 填充和步幅是卷积层的超参数
# 填充在输入周围添加额外的行/列，来控制输出形状的减少量。
# 给定32*32图像，应用5*5的卷积核，第一层得到输出28*28，每次减少到(nh-kh+1)*(nh-kh+1)
# 在输入周围添加行列
#              0 0 0 0 0
# 0 1 2        0 0 1 2 0                     0 3 8 4
# 3 4 5   ->   0 3 4 5 0     *   0 1    =    9 19 25 10
# 6 7 8        0 6 7 8 0         2 3         21 37 43 16
#              0 0 0 0 0                     6 7 8 0
# 填充ph行和pw列，输出形状为(nh-kh+1+ph)*(nh-kh+1+pw)
# 通常取ph=kh-1, pw=kw-1，如此形状不变
# 当kh为奇数，填充均匀ph/2即可，若为偶数，一边多一行一边少一行即可

# 步幅是每次滑动核窗口时的行/列的步长，可以成倍的减少输出形状
# 如果输入大小224*224，使用5*5卷积核，需要44层才能将输入降低到4*4，计算量极大
# 每次在行列滑动更多步长即可
# 例如高度3，宽度2的步幅
#   0 0 0 0 0
#   0 0 1 2 0
#   0 3 4 5 0  *   0 1    =    0 8
#   0 6 7 8 0      2 3         6 8
#   0 0 0 0 0
# 给定高度sh和宽度sw步幅，输出形状是(nh-kh+ph)/sh+1 * (nw-kw+pw)/sw+1（都向下取整）

def comp_conv2d(conv2d,x):
    x = x.reshape((1,1) + x.shape)  # 1批量,1通道,高,宽
    y = conv2d(x)
    return y.reshape(y.shape[2:])  # 高，宽

conv2d = nn.Conv2d(1, 1, kernel_size=(3,3), padding=1)  # 所有侧边填充1个像素
x = torch.rand(size=(8,8))
y = comp_conv2d(conv2d, x)
print(y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3,3), padding=1, stride=(2,2))  # 步长2
y = comp_conv2d(conv2d, x)
print(y.shape)












