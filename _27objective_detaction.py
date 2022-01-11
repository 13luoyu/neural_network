import zipfile

import torch
from d2l import torch as d2l
import os
import pandas as pd
import torchvision

# 物体检测，识别图片里的多个物体的类别和位置
# 边缘框，通过4个数字定义，左上x，左上y，  右下x，右下y或宽，高
# 目标检测数据集，每行表示一个物体，包含图片文件名，物体类别，边缘框
# 经典数据集：cocodataset.org 80种物体，330K图片，1.5M个物体


d2l.set_figsize()
img = d2l.plt.imread('img/catdog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 两种边框表示法转换
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx,cy,w,h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

# 验证正确性
boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
d2l.plt.show()



d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
fname = d2l.download('banana-detection', 'data')
base_dir = os.path.dirname(fname)
fp = zipfile.ZipFile(fname, 'r')
fp.extractall(base_dir)
data_dir, ext = os.path.splitext(fname)

def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256  # /256将边框归一化到[0,1]之间，下面会缩放回来
    # unsqueeze在第1维上增加一个维度，存放边框数量


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape)  # （批量大小、通道数、高度、宽度）
print(batch[1].shape)  # (批量大小， m ，5）其中 m 是数据集的任何图像中边界框可能出现的最大数量。
# 小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。通常来说，图像可能拥有不同数量个边界框(多个物体)，因此，在达到m
# 之前，边界框少于m的图像将被非法边界框填充。这样，每个边界框的标签将被长度为5的数组表示。数组中的第一个元素是边界框中对象的类别，其中-1表示用于填
# 充的非法边界框。对于香蕉数据集而言，由于每张图像上只有一个边界框，因此m=1


imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255  # permute将tensor的维度换位，并/255，都是为了满足plt的显示要求
print(imgs.shape)
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
d2l.plt.show()


