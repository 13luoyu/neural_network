import hashlib
import os
import tarfile
import zipfile
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()  # 创建一个字典
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
def download(name, cache_dir="data"):
    """下载一个DATA_HUB的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    fname = os.path.join(cache_dir, url.split('/')[-1])  # -1最后一个元素
    if os.path.exists(fname):  # 下载过了
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data=f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream = True, verify = True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar⽂件。"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar⽂件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有⽂件。"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (
DATA_URL + 'kaggle_house_pred_train.csv',
'585e9cc93e70b39160e7921475f9bcd7d31219ce')


DATA_HUB['kaggle_house_test'] = (
DATA_URL + 'kaggle_house_pred_test.csv',
'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

# 第一列为id，去掉，最后一列为结果(房价)，去掉
all_features = pd.concat((train_data.iloc[:,1:-1],
                          test_data.iloc[:,1:]))

# 数据预处理
# 每一列为一个特征，我们将所有缺失值替换为相应特征平均值，然后，为了
# 将所有特征放在一个尺度上，将特征值缩放到0均值和单位方差来标准化数据
# x = (x-μ)/σ
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 使用离散值替换对象特征，如MSZoning包含RL和Rm，创建两个新的列MSZoning——RL和MSZoning_RM，值为0/1
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
# 格式转换，.values为numpy格式，然后转为张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),
                            dtype=torch.float32)

# 训练
loss = nn.MSELoss()  # 均方损失函数(yi-y)^2
in_features = train_features.shape[1]  # 列数

def get_net():  # 简单的线性神经网络
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# 我们关心的是相对误差，而不是绝对误差
def log_rmse(net, features, labels):
    # 为了在取对数时稳定该值，小于1的值设定为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 根号下((log(yi)-log(y))^2)
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features,
          test_labels, num_epochs, learning_rate,
          weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features,train_labels),batch_size)
    # Adam优化算法
    optimizer =torch.optim.Adam(net.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for x,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(x), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k,i,x,y):
    """获得k折样本，其中i为验证，其余为训练"""
    assert k>1
    fold_size = x.shape[0] // k  # 整数除法
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        x_part, y_part = x[idx,:],y[idx,:]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat([x_train,x_part], 0)
            y_train = torch.cat([y_train,y_part], 0)
    return x_train, y_train, x_valid, y_valid

def k_fold(k, x_train, y_train, num_epochs, learning_rate,
           weight_decay, batch_size):
    """k折交叉验证"""
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs,
                                   learning_rate, weight_decay,
                                   batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)),[train_ls, valid_ls],
                     xlabel="epoch", ylabel="rmse",xlim=[1,num_epochs],
                     legend=['train','valid'], yscale='log')
            d2l.plt.show()
        print(f'折{i+1},训练log rmse{float(train_ls[-1]):f},'
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum/k, valid_l_sum/k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels,
                          num_epochs, lr, weight_decay, batch_size)
print(f'{k}折验证：平均训练log rmse：{float(train_l):f},'
      f'平均验证log rmse：{float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels,
                   test_data, num_epochs, lr, weight_decay, batch_size):
    """训练，并用测试集测试，然后结果写入csv"""
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None,
                        None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1,num_epochs+1), [train_ls],xlabel='epoch',
             ylabel='log rmse', xlim=[1,num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],
                           axis=1)
    submission.to_csv('data/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels,
               test_data, num_epochs, lr, weight_decay, batch_size)






