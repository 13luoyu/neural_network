# 命令式编程，一步一步执行(解释器)
def add(a, b):
    return a + b
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
print(fancy_func(1, 2, 3, 4))

# 符号式编程，代码在完全定义了之后运算，（编译器）过程为：
# 1、定义计算流程
# 2、将流程编译成可执行的程序
# 3、给定输入，调用编译好的程序执行
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1,2,3,4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

# pytorch和tensorflow都支持命令式编程和符号式编程
# Sequential混合式编程
import torch
from torch import nn
from d2l import torch as d2l


# 生产网络的工厂模式
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
print(net(x))

# torch.jit.script函数转换模型，从解释器到编译器
net = torch.jit.script(net)
print(net(x))

#@save
class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

net = get_net()
with Benchmark('无torchscript'):
    for i in range(1000): net(x)
torch.save(net, 'my_mlp1')

net = torch.jit.script(net)
print(net.code)  # 打印执行流(python语言)
with Benchmark('有torchscript'):
    for i in range(1000): net(x)

# 编译模型的好处之一是我们可以将模型及其参数序列化（保存）到磁盘。这允许这些训练好的模型部署到其他设备上
net.save('my_ml2')

# TorchScript是pytorch模型的中间表示形式，可以在高性能环境（c++等）下运行
# TorchScript是一种从PyTorch代码创建可序列化和可优化模型的方法。任何TorchScript程序都可以从Python进程中保存并在没有Python依赖项的进程中加载
# TorchScript主要在torch.jit中，有两个核心模块：tracing和scripting
# torch.jit.trace接受一个module或一个函数，以及一个样例输入，它就会调用这个模块或者函数，跟踪计算步骤，输出函数或者module的静态图。
# Tracing 可以很好的用于直接跟踪不依赖于数据的计算流程。
# 但是如果函数依赖于数据，含有控制流，那么就会出现有的控制流不能被捕捉到，对于这种情况，需要使用scripting
# 它接受一个module或一个函数，但是不需要输入样例，它会把包括控制流的内容也转成TorchScript。
# 需要注意的是scripting 只支持Python的子集，所以有时候可能需要重写代码来支持TorchScript。