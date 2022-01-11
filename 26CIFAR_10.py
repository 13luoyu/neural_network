import collections
import math
import os
import shutil  # 文件夹与文件操作
import zipfile

import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# https://www.kaggle.com/c/cifar-10

# 数据集的子集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

fname = d2l.download('cifar10_tiny', 'data')
base_dir = os.path.dirname(fname)
fp = zipfile.ZipFile(fname, 'r')
fp.extractall(base_dir)
data_dir, ext = os.path.splitext(fname)

def read_csv_labels(fname):
    """读取fname，返回字典(map)"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]  # 不读第0行
    tokens = [l.rstrip().split(',') for l in lines]  # rstrip()删除字符串尾部空字符
    return dict(((name, label) for name, label in tokens))  # dict字典

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

# 整理数据集
def copyfile(filename, target_dir):
    """将文件复制到指定目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将原始训练数据集拆分为训练集和测试集，其比率为valid_ratio
    产生3个目录，train为训练目录，valid为验证目录，train_valid为和"""
    # 训练数据集中样本最少的类别中的样本数,Counter返回字典
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集合中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n*valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)


# 图像增广解决过拟合
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    # 裁剪一个宽高比例不变，面积为0.64到1倍的正方形，最后缩放为32*32
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64,1.0),
                                             ratio=(1.0,1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914,0.4822,0.4465],
                                     [0.2023,0.1994,0.2010])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 读取数据集
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train
    ) for folder in ['train', 'train_valid']
]
valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test
    ) for folder in ['valid', 'test']
]

# drop_last控制如果最后一个batch不足batch_size，是否丢掉
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    ) for dataset in (train_ds, train_valid_ds)
]
valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True
)
test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size, shuffle=False, drop_last=False
)

def get_net():
    return d2l.resnet18(10, 3)  # 输入3通道，输出10类
loss = nn.CrossEntropyLoss(reduction="none")

def train(net, train_iter, valid_iter, num_epochs, lr, wd,
          devices, lr_period, lr_decay):
    """每隔lr_period个训练，将lr * lr_decay(降低lr)"""
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)  # new!!
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(devices)}')

# 使用train训练，valid测试
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
d2l.plt.show()



# 使用train_valid训练，test测试，并写文件
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv(os.join(data_dir,'submission.csv'), index=False)