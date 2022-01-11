import torch
from d2l import torch as d2l


# 锚框，一类目标检测算法，提出多个边缘框，称为锚框，然后预测每个锚框是否有
# 关注的物体，如果有，预测从这个锚框到真实边缘框的偏移
# IoU交并比用以计算两个框之间的相似度，J(A,B) = A∩B / A∪B，0表示无重叠，1重合
# 每个锚框是一个训练样本，将每个锚框，要么标注成背景，要么关联一个真实边框
# 使用非极大值抑制（NMS）输出，每个锚框预测一个边缘框，NMS用以合并相似的预测（多合一）
# 做法：选中是非背景类的最大预测值，去掉所有其他和它的IoU值大于θ的预测


# 假设输入图像高度h，宽度w，以图像每个像素为中心生成不同形状的锚框，缩放比为∈（0,1）
# 宽高比为r>0，锚框宽度高度分别为ws根号r, hs/根号r
# 为了生成多个锚框，我们设置许多缩放比s1,s2,...,sn和许多宽高比r1,r2,...,rm
# 在实践中，为了减少锚框数，只考虑包含s1和r1的组合：
# (s1,r1),(s1,r2),...,(s1,rm),(s2,r1),(s3,r1),...,(sn,r1)
# 对于一幅图，我们有wh(n+m-1)个锚框，下面的代码实现了这一思想
# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!
def multibox_prior(data, sizes, ratios):
    """生成锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1  # 每个像素锚框数
    size_tensor = torch.tensor(sizes, device=device)  # 转化为张量
    ratio_tensor = torch.tensor(ratios, device=device)

    # 将锚点移到像素中心
    offset_h, offset_w = 0.5, 0.5  # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    steps_h = 1.0 / in_height  # 将高宽缩放到(0,1)
    steps_w = 1.0 / in_width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)  # 返回排列组合的结果，输入为2个一维张量，输出向量
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # a = torch.tensor([1, 2, 3, 4])
    # print(a)
    # b = torch.tensor([4, 5, 6])
    # print(b)
    # x, y = torch.meshgrid(a, b)
    # print(x)
    # print(y)
    #
    # 结果显示：
    # tensor([1, 2, 3, 4])
    # tensor([4, 5, 6])
    # tensor([[1, 1, 1],
    #         [2, 2, 2],
    #         [3, 3, 3],
    #         [4, 4, 4]])
    # tensor([[4, 5, 6],
    #         [4, 5, 6],
    #         [4, 5, 6],
    #         [4, 5, 6]])
    # 生成boxes_per_pixel个高和宽
    # 之后用于创建锚框的四角坐标 (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    print(size_tensor, ratio_tensor)
    print(w,h)
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w,-h,w,h)).T.repeat(
        in_height * in_width, 1) / 2  #  行重复in_h*in_w次，列重复1次
    print(anchor_manipulations.shape)
    # 每个中心点有boxes_per_pixel个锚框
    out_grid = torch.stack((shift_x, shift_y, shift_x, shift_y),dim=1)\
        .repeat_interleave(boxes_per_pixel, dim=0)
    # 这里dim=1等同于上面的.T


    # # 假设是时间步T1的输出
    # T1 = torch.tensor([[1, 2, 3],
    #                    [4, 5, 6],
    #                    [7, 8, 9]])
    # # 假设是时间步T2的输出
    # T2 = torch.tensor([[10, 20, 30],
    #                    [40, 50, 60],
    #                    [70, 80, 90]])
    # print(torch.stack((T1, T2), dim=0).shape)
    # print(torch.stack((T1, T2), dim=1).shape)
    # print(torch.stack((T1, T2), dim=2).shape)
    # # outputs:
    # torch.Size([2, 3, 3])
    # torch.Size([3, 2, 3])
    # torch.Size([3, 3, 2])

    # >> > y = torch.tensor([[1, 2], [3, 4]])
    # >> > torch.repeat_interleave(y, 2)
    # tensor([1, 1, 2, 2, 3, 3, 4, 4])
    # # 指定维度
    # >> > torch.repeat_interleave(y, 3, 0)
    # tensor([[1, 2],
    #         [1, 2],
    #         [1, 2],
    #         [3, 4],
    #         [3, 4],
    #         [3, 4]])
    # >> > torch.repeat_interleave(y, 3, dim=1)
    # tensor([[1, 1, 1, 2, 2, 2],
    #         [3, 3, 3, 4, 4, 4]])

    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread('img/catdog.jpg')
h, w = img.shape[:2]
print(h,w)
x = torch.rand(size=(1,3,h,w))
y = multibox_prior(x, sizes=[0.75,0.5,0.25],
                   ratios=[1,2,0.5])
print(y.shape)  # y的形状为(图像高度，图像宽度，同一像素为中心的锚框数，4)
boxes = y.reshape(h,w,5,4)  # 这里5是已知的，因为sizes和ratios已知
print(boxes[250,250,0,:])  # 打印某个像素第0锚框的坐标


# 显示以图像中一个像素为中心的所有锚框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """图像，锚框，...,..."""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list,tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b','g','r','m','c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i%len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color=='w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center',ha='center',fontsize=9,
                      color=text_color,
                      bbox=dict(facecolor=color,lw=0))

d2l.set_figsize()
bbox_scale = torch.tensor((w,h,w,h))
fig=d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250,250,:,:]*bbox_scale,
            ['s=0.7,r=1','s=0.5,r=1','s=0.25,r=1','s=0.75,r=2',
             'x=0.75,r=0.5'])
d2l.plt.show()

# 计算两个锚框的交并比
def box_iou(boxes1, boxes2):
    """boxes1.shape=(boxe1的数量,4)"""
    box_area = lambda boxes: ((boxes[:,2] - boxes[:,0]) *
                              (boxes[:,3] - boxes[:,1]))

    # .shape=(box数量,1)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # 下面3个变量.shape=(boxes1数量，boxes2数量，2)
    inter_upperlefts = torch.max(boxes1[:,None,:2],boxes2[:,:2])  # 这里boxes1这样写将2维转换为3维，boxes2利用广播机制
    inter_lowerrights = torch.min(boxes1[:,None,2:],boxes2[:,2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)  # clamp夹紧，将所有小于0的值设置为0

    # 下面2个变量.shape=(boxes1数量，boxes2数量)
    inter_areas = inters[:,:,0] * inters[:,:,1]  # 面积
    union_areas = areas1[:,None] + areas2 - inter_areas
    return inter_areas / union_areas


# 将最接近的真实边界框分配给锚框
# 给定图像，假设锚框是A1, A2, . . . , Ana，真实边界框是B1, B2, . . . , Bnb，其中na ≥ nb。让我们定义⼀个矩
# 阵X ∈ Rna×nb，其中ith⾏和jth列中的元素xij是锚框Ai和真实边界框Bj的IoU。该算法包含以下步骤：
# 1. 在矩阵X中找到最大的元素，并将它的行索引和列索引分别表示为i1和j1。然后将真实边界框Bj1分配给
# 锚框Ai1。这很直观，因为Ai1和Bj1是所有锚框和真实边界框配对中最相近的。在第一个分配完成后，丢
# 弃矩阵中i1th行和j1th列中的所有元素。
# 2. 在矩阵X中找到剩余元素中最大的元素，并将它的行索引和列索引分别表示为i2和j2。我们将真实边界
# 框Bj2分配给锚框Ai2，并丢弃矩阵中i2th行和j2th列中的所有元素。
# 3. 此时，矩阵X中两行和两列中的元素已被丢弃。我们继续，直到丢弃掉矩阵X中nb列中的所有元素。此
# 时，我们已经为这nb个锚框各自分配了⼀个真实边界框。
# 4. 只遍历剩下的na − nb个锚框。例如，给定任何锚框Ai，在矩阵X的第ith⾏中找到与Ai的IoU最⼤的真实
# 边界框Bj，只有当此IoU⼤于预定义的阈值时，才将Bj分配给Ai。
# @depressed 这是旧的算法，新的算法使用神经网络自动学习
def assign_anchor_to_bbox(ground_truth,anchors,device,iou_threshold=0.5):
    """真实框，锚框，设备，阈值"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算iou,jaccard[i][j]为锚框i和真实边界框j的iou
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)  # 求对于每个锚框，它和真实边界框的最大值，返回最大值和索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j  # 阈值分配法，对应算法步骤4
    col_discard = torch.full((num_anchors,),-1)
    row_discard = torch.full((num_gt_boxes,),-1)
    for _ in range(num_gt_boxes):  # 对于每个真实框（每列）
        max_idx = torch.argmax(jaccard)  # 找到整个iou最大值
        box_idx = (max_idx%num_gt_boxes).long()  # 第几列，对应哪个真实框
        anc_idx = (max_idx/num_gt_boxes).long()  # 第几行，对应哪个锚框
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard  # 设为-1，表示已找到，不再参与计算
    return anchors_bbox_map


# 现在已经找到了每个锚框的对应真实框，A锚框的类别被设置为B真实框的类别，
# 为了度量锚框A的偏移量，使用B和A中心坐标相对位置，以及这两个框大小标记
# 下面的式子是为了获得分布更均衡，更易于适应的偏移量
# 给定框A，B，中心为(xa,ya)和(xb,yb)，宽wa,wb，高ha,hb，A的偏移量为
# (  ((xb-xa)/wa - μx) / σx,
#    ((yb-ya)/ha - μy) / σy,
#    (log(wb/wa) - μw) / σw,
#    (log(hb/ha) - μh) / σh    )
# 其中常量默认值 μx=μy=μw=μh=0, σx=σy=0.1, σw=σh=0.2

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = d2l.box_corner_to_center(anchors)  # 左上x，左上y，宽，高
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:,:2] - c_anc[:,:2]) / c_anc[:,2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:,2:] / c_anc[:,2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

# 如果一个锚框没有被分配真实边界框，我们只需将锚框的类别标记为“背景”
# 此函数将背景类别的索引设置为零，然后将新类别的整数索引递增一。
# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)  # unsqueeze增加维度，squeeze减少维度，去掉维数为1的的维度
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)  # 为锚框分配边界框
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)  # >=0，代表被分配了边界框，否则为背景
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)  # 返回非0元素的索引
        bb_idx = anchors_bbox_map[indices_true]  # 非0元素的索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
# 返回的第一个元素包含了为每个锚框标记的四个偏移值。 请注意，负类锚框的偏移量被标记为零。
# 返回的第二个元素是掩码（mask）变量，形状为（批量大小，锚框数*4）。
# 掩码变量中的元素与每个锚框的4个偏移量一一对应。 由于我们不关心对背景的检测，
# 负类的偏移量不应影响目标函数。 通过元素乘法，掩码变量中的零将在计算目标函数之前过滤掉负类偏移量。
# 返回的第三个元素包含标记的输入锚框的类别。

# 通过例子说明锚框标签
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
d2l.plt.show()


labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
print(labels)


# 使用非极大值抑制预测边界框
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # 将锚框+偏移量，得到预测边界框
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

# 1从 L 中选取置信度最高的预测边界框 B1 作为基准，然后将所有与 B1 的IoU超过预定阈值 ϵ
# 的非基准预测边界框从 L 中移除。这时， L 保留了置信度最高的预测边界框，
# 去除了与其太过相似的其他预测边界框。简而言之，那些具有非极大值置信度的边界框被抑制了。
# 2从 L 中选取置信度第二高的预测边界框 B2 作为又一个基准，然后将所有与 B2 的IoU大于 ϵ 的非基准预测边界框从 L 中移除。
# 3重复上述过程，直到 L 中的所有预测边界框都曾被用作基准。此时， L 中任意一对预测边界框的IoU都小于阈值 ϵ ；因此，没有一对边界框过于相似。
# 4输出列表 L 中的所有预测边界框。
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!# !!!!!!!!!!!!!!!!使用!!!!!!!!!!!!!!!!!!!!
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框，参数为预测概率，偏移，锚框，阈值"""
    # cls_prob.shape = [批量数，预测物体种数，每种物体锚框数]
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    # 要预测3种物体，每种物体4个锚框
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):  # batch_size=1
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)  # 预测边界框
        keep = nms(predicted_bb, conf, nms_threshold)  # 置信度排序

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))  # 一维张量
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]  # 只出现一次的是背景
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1  # 标记为背景
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)  # 概率小于阈值，舍弃，标记为背景
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        # pred_info第一列为类别，其中-1为背景；第二列为概率，后四列为锚框
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
d2l.plt.show()

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
d2l.plt.show()