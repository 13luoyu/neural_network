import torch
import torchvision
from torch import nn
from torch import functional as F
from d2l import torch as d2l
import _27objective_detaction

img = d2l.plt.imread("img/catdog.jpg")
h, w = img.shape[:2]
print(h,w)

# 这里演示SSD多尺度目标检测的原理实现，不同大小锚框检测不同大小物体
def display_anchors(fmap_w, fmap_h, s):
    """在特征图fmap上生成锚框，每个像素为锚框中心"""
    d2l.set_figsize()
    fmap = torch.zeros((1,10,fmap_h,fmap_w))
    # 下面这个函数不关心通道数，只关心h和w，所以上面的10任意设置
    # sizes为面积缩放，ratios为宽高比
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1,2,0.5])
    bbox_scale = torch.tensor((w,h,w,h))  # 真实图片宽高，上面anchors为(0,1)之间的坐标
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0]*bbox_scale)

display_anchors(4,4,s=[0.15])
d2l.plt.show()
display_anchors(2,2,s=[0.4])
d2l.plt.show()
display_anchors(1,1,s=[0.8])
d2l.plt.show()



# 类别预测层，卷积层
def cls_predicator(num_inputs, num_anchors, num_classes):
    # 对每个锚框，预测是哪个类（+1是有背景）
    return nn.Conv2d(num_inputs, num_anchors*(num_classes+1),
                     kernel_size=3, padding=1)
    # kernel_size和padding这样设置不会改变输入图片高宽
# 卷积层输入有3维，高宽通道数，输出3维，高宽通道数，每个通道标识了一个锚框对一个类的特征提取


# 边界框预测层
def bbox_predicator(num_inputs, num_achors):
    # 4指的是4个坐标值
    return nn.Conv2d(num_inputs, num_achors*4,
                     kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    return block(x)
y1 = forward(torch.zeros(2,8,20,20), cls_predicator(8,5,10))
print(y1.shape)
y2 = forward(torch.zeros(2,16,10,10), cls_predicator(16,3,10))
print(y2.shape)

# 上面y1和y2除了批量相同，通道数、宽、高可能不同，因此将其转为二维，从而连接起来
def flatten_pred(pred):
    # 先将通道维度放到最后，然后拉成2维，start_dim=1表示将1后面所有维度拉成1个维度
    return torch.flatten(pred.permute(0,2,3,1),
                         start_dim=1)
def concat_preds(preds):
    # 在宽上连接起来，列增多
    return torch.cat([flatten_pred(p) for p in preds],
                     dim=1)
print(concat_preds([y1,y2]).shape)

def down_sample_blk(in_channels, out_channels):
    """高和宽减半块"""
    blk=[]
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3,padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels=out_channels
    blk.append(nn.MaxPool2d(2))  # 这里才减半
    return nn.Sequential(*blk)

y = forward(torch.zeros((2,3,20,20)),
            down_sample_blk(3,10))
print(y.shape)

# 基本网络块，从原图到加锚框之间的部分
def base_net():  # 经过这个网络，通道数3->64，高宽/8
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

y = forward(torch.zeros((2, 3, 256, 256)), base_net())
print(y.shape)

def get_blk(i):
    if i == 0:
        blk = base_net()  # 通道数3->64，高宽/8
    elif i == 1:
        blk = down_sample_blk(64, 128)  # 通道数64->128，高宽/2
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))  # 特征压到1*1
    else:
        blk = down_sample_blk(128, 128)  # 通道数不变，高宽/2
    return blk

#  给每个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)  # 特征提取
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)  # 生成锚框
    cls_preds = cls_predictor(Y)  # 类别预测  (批量数，锚框数*(类数+1)，高，宽)
    bbox_preds = bbox_predictor(Y)  # 边界预测  (批量数，锚框数*4，高，宽)
    return (Y, anchors, cls_preds, bbox_preds)

# 超参数
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]  # 锚框逐渐变大
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    """完整模型"""
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predicator(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predicator(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)  # 三维变三维，将所有锚框合起来
        cls_preds = concat_preds(cls_preds)  # 四维变二维
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)  # 二维变三维（批量，锚框数，类别数）
        bbox_preds = concat_preds(bbox_preds)  # 四维变二维
        return anchors, cls_preds, bbox_preds
        # 返回锚框，对类的预测，对边界框的预测

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)  # 三维，(1，锚框数，坐标)，1是因为所有批量都用这些锚框
print('output class preds:', cls_preds.shape)  # 三维，（批量，锚框数，类别数(1+1=2)）
print('output bbox preds:', bbox_preds.shape)  # 二维，（批量，锚框数*4）


# 训练香蕉数据集
batch_size = 16
train_iter, _ = _27objective_detaction.load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
cls_loss = nn.CrossEntropyLoss(reduction='none')  # 不要加起来，保留每个批量的loss
bbox_loss = nn.L1Loss(reduction='none')  # 差绝对值

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),  # 批量和锚框数合并为一维
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,  # 如果是背景，就不要算偏移了，mask=0
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

# 评价
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)  # 由预测一个y，变为返回锚框，类预测，边框预测
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
d2l.plt.show()



# 测试
X = torchvision.io.read_image('data/banana-detection/bananas_val/0.png').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]  # 返回非背景的预测
output = predict(X)

# 筛选出置信度高于threshold边界框，输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
display(img, output.cpu(), threshold=0.9)