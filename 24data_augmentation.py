import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 数据增强，对已有数据集进行增强，使得有更多的多样性
# 比如在图片里加入各种不同的噪音，改变图片的颜色和形状
# 翻转：上下翻转，左右翻转
# 切割：从图片中切割一块，然后变形为原始大小
# 颜色：改变色调、饱和度、明度

d2l.set_figsize()
img = d2l.Image.open('img/cat.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

def apply(img, aug, num_rows=2, num_cols=4,
          scale=1.5):
    """图片，图片增广办法，次数，放大缩小倍数"""
    y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# 随机水平翻转
# 函数以给定的概率翻转图片，默认0.5
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 随即垂直翻转
# 函数以给定的概率翻转图片，默认0.5
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机剪裁
# 三个参数为1期望输出尺寸，
# 2随机的面积比例，裁剪出的区域面积占原图总面积的比例，默认:0.08到1.0
# 3随机的裁剪宽高比范围，默认3.0/4到4.0/3
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)

# 随机颜色抖动
# 参数为亮度，对比度，饱和度，色温，0表示不改变
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0
))
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)
apply(img, color_aug)

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    shape_aug, color_aug
])
apply(img, augs)

#  32*32的图片，总共10类，ImageNet的子集
all_images = torchvision.datasets.CIFAR10(
    train=True, root="data", download=True
)
d2l.show_images([all_images[i][0] for i in range(32)],
                4,8,scale=0.8)
d2l.plt.show()

# 只是用随机左右翻转，然后将图片转为4维张量
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=is_train, transform=augs,
        download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size,
        shuffle = is_train
    )
    return dataloader


# 当训练网络和测试网络使用不同方式时，要在训练模型前使用net.train()
# 在测试模型前使用net.eval()，当net中有batch_normalization或dropout层时使用
# 原因是，1训练时是正对每个min-batch的，但是在测试中往往是针对单张图片，即不存在min-batch的概念。由于网络训练完毕后参数都是固定的，因此每个批次的均值和方差都是不变的，因此直接结算所有batch的均值和方差。所有Batch Normalization的训练和测试时的操作不同
# 2在训练中，每个隐层的神经元先乘概率P，然后在进行激活，在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P。

def train_batch_ch13(net, x, y, loss, trainer, devices):
    if isinstance(x, list):
        x = [xx.to(devices[0]) for xx in x]
    else:
        x = x.to(devices[0])
    y = y.to(devices[0])

    net.train()   # 当网络中有dropout或batch_normalization时使用
    trainer.zero_grad()
    pred = net(x)
    l = loss(pred, y)
    l.sum().backward()  # 在反向传播算法中，只有标量才可以计算梯度（所以调用sum，为一个数）
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

# Defined in file: ./chapter_computer-vision/image-augmentation.md
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer,
                                      devices)
            metric.add(l, acc, labels.shape[0], labels.numel())  # labels.shape[0] = 1
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches,
                    (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net)
d2l.plt.show()

