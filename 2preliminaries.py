import os
import pandas as pd
import torch

x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
x=x.reshape(3,4)
print(x)
print(torch.zeros(3,4))
print(torch.ones(3,4))
print(torch.tensor([[1,2],[3,4]]))

y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y), dim=0))
print(torch.cat((x,y), dim=1))
print(x==y)
print(x.sum())

# 广播机制
a=torch.arange(3).reshape(3,1)
b=torch.arange(2).reshape(1,2)
print(a)
print(b)
print(a+b)  # 同行/列复制

print(x[-1])
print(x[0:2])
x[1,2]=9

# 重新分配内存
x=x+y
# 原地操作
x[:]=x+y
x+=y
y=x
y=x.clone()

x=x.numpy()
print(x)


# 数据预处理
f=open("csv/2.csv", "w")
f.write("Rooms,Alley,Price\n")
f.write("NA,Pave,127500\n")
f.write("2,Sr,106000\n")
f.write("4,NA,178100\n")
f.write("NA,NA,140000\n")  # NA表示空值
f.close()

f=open("csv/2.csv", "r")
data=pd.read_csv(f)
dat=data.iloc[:,0:1]  # 取出第一列
dat=dat.fillna(0)  # NA填0
d=torch.tensor(dat.values)
print(d)

dat=data.iloc[:,0:2]  # 取出前两列
dat=pd.get_dummies(dat, dummy_na=True)  #将每个值视为一个类别，数字除外
dat=dat.fillna(0)  #数字NA置0
print(dat)
d=torch.tensor(dat.values)




# 线性代数
a=torch.arange(20).reshape(5,4)
print(a)
print(a.T)  # 转置
b=a
print(a+b)

# 对指定轴求和
a=torch.arange(40, dtype=torch.float32).reshape(2,5,4)
print(a.sum(axis=0))
print(a.sum(axis=1))  # 对5的对应轴求和，结果为2*4
print(a.sum(axis=[0,1]))
a.sum()
print(a.mean())  # 平均值

# 计算总和或均值时保持轴数不变
print(a.sum(axis=[0]))  # 仍然三维

# 一维向量相同位置按元素乘积的和
x=torch.tensor([1,2,3,4])
y=x.clone()
print(torch.dot(x,y))

# 矩阵向量积
y=torch.tensor([[1,2,3,4],[2,3,4,5]])
print(torch.mv(y,x))

# 矩阵*矩阵
x=torch.arange(20, dtype=torch.float32).reshape(4,5)
y=x.clone().T
print(torch.mm(x,y))

# L2范数（平方和的平方根）
print(torch.norm(x))

# L1范数（绝对值之和）
print(torch.abs(x).sum())


# y=2xTx，其中x为列向量
x=torch.arange(4.0)  #x是向量
x.requires_grad_(True)   # 存储梯度
y = 2*torch.dot(x,x)    # y是标量
print(y)
y.backward()    # 调用反向传播函数计算y关于x每个分量的梯度
print(x.grad)   # 查看梯度
print(x.grad == 4*x)
# 默认，pytorch会累积梯度，所以需要清除之前的值
x.grad.zero_()
print(x.grad)
y=x*x   #y是向量
u=y.detach()    #系统将y看作常数，求梯度时
z=u*x
z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2*x)

# 即使函数计算需要通过python控制流（条件、循环、函数调用）完成，也可以调用backward计算梯度




