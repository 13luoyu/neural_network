

# 样式迁移：将样式图片的样式迁移到内容图片上，得到合成图片
# 比如将风景图与油画合成，得到油画样式的风景图
# 基于CNN的样式迁移
# 我们要训练的是合成图像。将其初始化为内容图像。该合成图像是风格迁移过程中唯一
# 需要更新的变量，即风格迁移所需迭代的模型参数。 然后，我们选择一个
# 预训练的卷积神经网络来抽取图像的特征，其中的模型参数在训练中无须更新。
# 这个深度卷积神经网络凭借多个层逐级抽取图像的特征，我们可以选择其中某些层的输
# 出作为内容特征或风格特征。

# 接下来，我们通过前向传播计算风格迁移的损失函数，并通过反向传播迭代模型参数，
# 即不断更新合成图像。
# 风格迁移常用的损失函数由3部分组成：
# （i）内容损失使合成图像与内容图像在内容特征上接近；
# （ii）风格损失使合成图像与风格图像在风格特征上接近；
# （iii）全变分损失则有助于减少合成图像中的噪点。
# 最后，当模型训练结束时，我们输出风格迁移的模型参数，即得到最终的合成图像。


import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
content_img = d2l.Image.open('img/rainier.jpg')
style_img = d2l.Image.open('img/autumn-oak.jpg')

# 预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    """标准化大小，在三个通道标准化颜色值"""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    """将输出图像像素还原为标准化之前的值"""
    img = img[0].to(rgb_std.device)
    # torch.clamp将输入夹紧到0,1之间
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]  # 匹配层，包括样式匹配（既包含局部，又包含全局）和内容匹配（全局）
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])  # 28层之后的丢掉不要
# 抽取特征
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 获得初始内容图片和内容图片对应层的特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y
# 获得初始样式图片和样式图片对应层的特征
def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# 定义损失函数
def content_loss(Y_hat, Y):
    # 内容损失，使用平方损失函数
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):  # 格拉姆矩阵X*XT
    # n为宽*高，x转换为c个长度hw的向量，每个向量代表通道i上的风格
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    # X*XT矩阵中，Xij代表通道i和j上风格特征相关性，结果为c*c矩阵，
    return torch.matmul(X, X.T) / (num_channels * n)
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
# 全变分损失，表示像素尽可能与临近像素值相似
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])  # 侧重style
    return contents_l, styles_l, tv_l, l

# 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)  # 使用内容图片初始化合成图片
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
d2l.plt.show()