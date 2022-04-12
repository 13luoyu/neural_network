import torch

# 可⽤gpu的数量
print(torch.cuda.device_count())

def try_gpu(i=0):
    """如果存在，返回gpu的第i个核心，否则返回cpu"""
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def try_all_gpus():
    """返回所有可用gpu，若没有gpu返回cpu"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


x = torch.tensor([1,2,3])
print(x.device)

x = torch.tensor([1,2,3], device=try_gpu())
print(x)

net = torch.nn.Sequential(torch.nn.Linear(3,1))
net = net.to(device=try_gpu())  # 所有模型以及参数存储在gpu上