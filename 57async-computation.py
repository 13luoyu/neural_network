import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l

# Python是单线程的
# 在诸多的深度学习框架中，MXNet和TensorFlow之类则采用了一种异步编程（asynchronous programming）模型来提高性能，
# 而PyTorch则使用了Python自己的调度器来实现不同的性能权衡。对于PyTorch来说GPU操作在默认情况下是异步的。
# 当你调用一个使用GPU的函数时，操作会排队到特定的设备上，但不一定要等到以后才执行。
# 这允许我们并行执行更多的计算，包括在CPU或其他GPU上的操作。

# 考虑矩阵相乘
device = d2l.try_gpu()
a = torch.randn(size=(1000,1000), device=device)
b = torch.mm(a,a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000,1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000,1000), device=device)
        b = torch.mm(a,a)

with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

# numpy在cpu上执行，最慢，第一个torch是在gpu上，所以快
# 2比3快的原因在于，pytorch默认异步，所以执行完成前返回，3则是同步了
