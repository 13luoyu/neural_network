import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import _31semantic_segmentation

# 全连接卷积神经网络FCN
# FCN使用转置卷积层替换CNN最后的全连接层，从而实现每个像素的预测
# img -> CNN -> 1*1Conv -> Transposed conv -> img

# 使用预训练的RestNet
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 查看最后3层
print(list(pretrained_net.children())[-3:])
# 最后三层分别为一个残差块，一个全局平均池化层和一个全连接层
# 我们的工作是替换掉池化层和全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])

x = torch.rand(size=(1,3,320,480))
print(net(x).shape)  # 经过网络可以发现，高宽都/32

num_classes = 21
# 1*1的卷积层，不改变高宽，只改变通道数
net.add_module('final_conv',
               nn.Conv2d(512, num_classes, kernel_size=1))
# 步长32，高宽*32，恢复到原来大小，剩余参数控制高宽不变
net.add_module('transpose_conv',
               nn.ConvTranspose2d(num_classes, num_classes,
                                  kernel_size=64, padding=16,
                                  stride=32))
print(net(x).shape)



# 有时我们需要放大图片，双线性插值是一种方法（上采样），它也可以被用来初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2  # 整数除法
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight  # 这个权重标识了某像素周边像素对该像素值的影响

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))  #初始化权重
img = torchvision.transforms.ToTensor()(d2l.Image.open('img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()
# 上面的转置卷积，将图像高宽*2，而不失真



# 使用上述方法初始化
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
# 读取语义分割数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = _31semantic_segmentation.load_data_voc(batch_size, crop_size)
# 训练
def loss(inputs, targets):
    # 这样写，是因为现在的比较是两个图片比较，要在高上和宽上作差，求loss均值
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
d2l.plt.show()

# 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)  # 所有通道的最大值
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]  # 颜色值
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)  # 裁剪
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
d2l.plt.show()