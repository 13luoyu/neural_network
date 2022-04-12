import torch
import torchvision
from IPython import display
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch import nn

d2l.use_svg_display()  # 使用svg格式显示图片

# 分类，多个输入，多个输出，输出oi时预测为第i类的可信度
# yi^=softmax(oi)=exp(oi)/求和(exp(oi))
# 损失l(y, y^) = -Σi(yi * logyi^)
# 梯度=softmax(oi) - yi

#batch_size和lr是超参数


# 通过ToTensor实例将图片数据从PIL类型转换为32位浮点数
# 并通过除以255使得所有像素的数值在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)


def get_fashion_mnist_labels(labels):
    """返回数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """显示图片"""
    figsize = (num_cols*scale, num_rows*scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)  # 开子图
    axes=axes.flatten()  # 返回折叠成一维的数组
    for i, (ax,img) in enumerate(zip(axes, imgs)):  #zip为元组(存图位置，原图)
        if torch.is_tensor(img):  # 张量
            ax.imshow(img.numpy())
        else:  # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

x,y=next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(x.reshape(18,28,28), 2, 9, get_fashion_mnist_labels(y))
d2l.plt.show()


batch_size=256  # 一次训练选取的样本数
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)  # 4进程读
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4)

# 查看读取所有数据需要多久
# timer=d2l.Timer()
# for x,y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')



# softmax
num_inputs = 28*28
num_outputs = 10
w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

def softmax(x):  # x为批量数*特征数
    """softmax(xij)=e^xij/求和k(e^xik)"""
    x_exp=torch.exp(x)
    partition=x_exp.sum(1, keepdim=True)  # 对每行求和，结果为向量
    return x_exp/partition  # 广播机制

def net(x):
    """神经网络模型"""
    return softmax(torch.matmul(x.reshape(-1,w.shape[0]), w)+b)

def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    # l(y, y^) = y * -log(y^)， 这里除了正确标签y=1，其余y=0
    # y[[0,1], x] = [y[0][x[0]], y[1][x[1]]]
    # y_hat[range(len(y_hat)), y]表示预测为正确类别的概率
    # y_hat[1][3]图1被归类为3的概率，y[2]=3，图2的正确归类为3
    # 然后取负对数，概率越小，结果（损失）越大
    # y_hat为256*10矩阵，因为一次输入256张图，10个分类结果
    loss=-torch.log(y_hat[range(len(y_hat)), y])  # loss = 256向量
    return loss

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # y_hat为二维矩阵并且列数>1
        y_hat = y_hat.argmax(axis=1)  # 取出每行中最大的值，结果为向量
    cmp = y_hat.type(y.dtype) == y  # y_hat转换为y数据类型并比较，结果为bool向量
    return float(cmp.type(y.dtype).sum())  # bool转为y数据类型，求和

class Accumulator:
    def __init__(self, n):
        self.data=[0.0]*n
    def add(self, *args):
        self.data=[a+float(b) for a,b in zip(self.data, args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    for x,y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric=Accumulator(3)
    for x,y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y),
                       y.numel())
    # 返回训练损失和训练精度
    return metric[0]/metric[2], metric[1]/metric[2]


class Animator: #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
            ylim=None, xscale='linear', yscale='linear',
            fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
            figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator=Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],
                      legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 每次训练完，衡量效果
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr=0.1
def updater(batch_size):
    return d2l.sgd([w,b], lr, batch_size)
num_epochs=10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net,test_iter,n=6):
    """预测标签"""
    for x,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles=[true+'\n'+pred for true, pred in zip(trues, preds)]
    d2l.show_images(x[0:n].reshape(n,28,28), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
d2l.plt.show()




print('---------------------------------------------------')

batch_size = 256
# 数据迭代器
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)  # 4进程读
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4)
# nn.Flatten()展平层，用来调整输入的形状，矩阵调整为28*28=784的向量
net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))

# 初始化权重w
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss()  # 交叉熵损失函数，就是-log(预测正确的标签的概率)
trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # 优化函数
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
predict_ch3(net, test_iter)
d2l.plt.show()



