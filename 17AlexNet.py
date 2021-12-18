import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data

# AlexNet是更深更大的LeNet，主要改进：丢弃法，ReLU，MaxPooling
# sigoid，当函数输出接近于0或1时，其梯度几乎0，反向传播更新困难。ReLU在正区间梯度1
# 丢弃法，平滑图片


net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96,256,kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)

x = torch.randn(1,1,224,224)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, 'Output shape:\t', x.shape)



batch_size = 64
# 将Fashion-MNIST图像分辨率扩为224*224
trans = [torchvision.transforms.ToTensor()]
trans.insert(0, torchvision.transforms.Resize(224))
trans = torchvision.transforms.Compose(trans)

mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter,num_epochs, lr, d2l.try_gpu() )
d2l.plt.show()

# d2l.load_data_fashion_mnist()


