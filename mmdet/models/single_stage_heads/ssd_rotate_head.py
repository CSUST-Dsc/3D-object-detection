import torch.nn as nn
import numpy as np
from mmdet.models.utils import one_hot
from mmdet.ops.iou3d import iou3d_utils
from mmdet.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch
import torch
import torch.nn.functional as F
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss, weighted_cross_entropy
from mmdet.core.utils.misc import multi_apply
from mmdet.core.bbox3d.target_ops import create_target_torch
import mmdet.core.bbox3d.box_coders as boxCoders
from mmdet.core.post_processing.bbox_nms import rotate_nms_torch
from functools import partial

def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha  # 1.6

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)
    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

# 对应图4中的bbox和cls初次结果
class SSDRotateHead(nn.Module):

    def __init__(self,
                 num_class=1, # 3D 目标检测类别，一类，车类
                 num_output_filters=768, # cfg 中是 256
                 num_anchor_per_loc=2, # 单元位置中 Anchor 的数量，如果是两个，那就是横放的 Anchor 和竖放的 Anchor。
                 use_sigmoid_cls=True, # 使用 sigmoid 函数用于分类
                 encode_rad_error_by_sin=True, # 使用 sin 函数计算误差角
                 use_direction_classifier=True, # 对方向进行分类（正对相机，背对相机）
                 box_coder='GroundBox3dCoder', # 有关 3D框 的参数
                 box_code_size=7, # 用 7 个参数表述一个 3D 框，分别是 xyzhwl 以及 score
                 ):
        super(SSDRotateHead, self).__init__()
        # 如果使用 sigmoid，num_cls 意思是每个位置的 Anchor 都要判别类别
        num_anchor_per_loc *= num_class
        if use_sigmoid_cls:
            self._num_class = num_class
        else:
            self._num_class = num_class + 1
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_sigmoid_cls = use_sigmoid_cls
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._use_direction_classifier = use_direction_classifier
        self._box_coder = getattr(boxCoders, box_coder)()
        self._box_code_size = box_code_size
        self._num_output_filters = num_output_filters

        # 从通道数为 num_output_filters 的特征卷积出通道数 num_cls 的特征，作为类别预测结果；
        # 若 num_cls = 1， 可以说大于 0.5 就是目标类。
        self.conv_cls = nn.Conv2d(num_output_filters, num_anchor_per_loc * self._num_class, 1)
        # 从通道数为 num_output_filters 的特征卷积出通道数 num_anchor_per_loc * box_code_size 的特征，作为 3D框 的回归结果；
        # 每一个位置上的每一个Anchor都要回归出一个 3D框 和它的置信度 score
        self.conv_box = nn.Conv2d(
            num_output_filters, num_anchor_per_loc * box_code_size, 1)
        # 从通道数为 num_output_filters 的特征卷积出通道数 num_anchor_per_loc * 2 的特征，作为类别预测结果；
        # 每一个位置上的每一个Anchor都要回归出 2 个方向，即面向相机，还是背对相机
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                num_output_filters, num_anchor_per_loc * 2, 1)

    def add_sin_difference(self, boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
            boxes2[..., -1:])
        rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
        boxes1 = torch.cat((boxes1[..., :-1], rad_pred_encoding), dim=-1)
        boxes2 = torch.cat((boxes2[..., :-1], rad_tg_encoding), dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, anchors, reg_targets, use_one_hot=True):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, 7)
        rot_gt = reg_targets[..., -1] + anchors[..., -1]
        dir_cls_targets = (rot_gt > 0).long()
        if use_one_hot:
            dir_cls_targets = one_hot(
                dir_cls_targets, 2, dtype=anchors.dtype)
        return dir_cls_targets

    def prepare_loss_weights(self, labels,
                             pos_cls_weight=1.0,
                             neg_cls_weight=1.0,
                             loss_norm_type='NormByNumPositives',
                             dtype=torch.float32):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
        reg_weights = positives.type(dtype)
        if loss_norm_type == 'NormByNumExamples':
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
        elif loss_norm_type == 'NormByNumPositives':  # for focal loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        elif loss_norm_type == 'NormByNumPosNeg':
            pos_neg = torch.stack((positives, negatives), dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer
        else:
            raise ValueError("unknown loss norm type.")
        return cls_weights, reg_weights, cared

    def create_loss(self,
                    box_preds,
                    cls_preds,
                    cls_targets,
                    cls_weights,
                    reg_targets,
                    reg_weights,
                    num_class,
                    use_sigmoid_cls=True,
                    encode_rad_error_by_sin=True,
                    box_code_size=7):
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1, box_code_size)

        if use_sigmoid_cls:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

        one_hot_targets = one_hot(
            cls_targets, depth=num_class + 1, dtype=box_preds.dtype)

        if use_sigmoid_cls:
            one_hot_targets = one_hot_targets[..., 1:]

        if encode_rad_error_by_sin:
            # sin(a - b) = sinacosb-cosasinb
            box_preds, reg_targets = self.add_sin_difference(box_preds, reg_targets)

        loc_losses = weighted_smoothl1(box_preds, reg_targets, beta=1 / 9., \
                                       weight=reg_weights[..., None], avg_factor=1.)
        cls_losses = weighted_sigmoid_focal_loss(cls_preds, one_hot_targets, \
                                                 weight=cls_weights[..., None], avg_factor=1.)

        return loc_losses, cls_losses

    # 输出每个位置每个Anchor的3D框预测结果和置信度（合在box_preds），以及所在类别cls_preds和朝向判断dir_cls_preds。
    # 进过上述讨论，x 是 [B, C, 200, 176] 的张量
    def forward(self, x):
        # conv_box 和 conv_cls 是 1*1 的卷积
        N, _, H, W = x.shape
        box_preds = self.conv_box(x)  # 输出 [B, 14, 200, 176] 的张量
        cls_preds = self.conv_cls(x)  # 输出 [B, 2, 200, 176] 的张量
        # 为啥会出现 14？
        # 是因为 conv_box 的通道数定义为 num_anchor_per_loc * box_code_size = 2*7

        box_preds = box_preds.view(N, self._num_class, -1, H, W)
        cls_preds = cls_preds.view(N, self._num_class, -1, H, W)

        # [N, C, y(H), x(W)]
        # 对张量做置换，contiguous 是让置换后的张量内存分布连续的操作
        box_preds = box_preds.permute(0, 1, 3, 4, 2).contiguous()  # [B, 200, 176, 14]
        cls_preds = cls_preds.permute(0, 1, 3, 4, 2).contiguous()  # [B, 200, 176, 2]

        if self._use_direction_classifier:
            # conv_dir_cls 也是 1*1 的卷积
            dir_cls_preds = self.conv_dir_cls(x)   # 输出 [B, 4, 200, 176] 的张量
            dir_cls_preds = dir_cls_preds.view(N, self._num_class, -1, H, W)
            # 为什么是 4 呢？
            # 是因为 conv_dir_cls 的通道数定义为 num_anchor_per_loc * 2 = 2*2
            # 输出 [B, 200, 176, 4] 的张量
            dir_cls_preds = dir_cls_preds.permute(0, 1, 3, 4, 2).contiguous()

        return box_preds, cls_preds, dir_cls_preds

    # box_preds, cls_preds 是 RPN 网络的预测值，分别是
    # [B, 200, 176, 14] 的张量和 [B, 200, 176, 2] 的张量
    # gt_bboxes, gt_labels 是 3d 目标的框参数和类别
    # anchors, anchors_mask 的概念在上一篇博客已经介绍了
    # cfg 是配置参数
    def loss(self, box_preds, cls_preds, dir_cls_preds, gt_bboxes, gt_labels, gt_types, anchors, anchors_mask, cfg):

        batch_size = box_preds.shape[0]

        multi_labels = list()
        multi_targets = list()
        multi_anchors = list()
        # 下面几行代码的作用
        # 生成与 box_preds, cls_preds 相对应的真值 targets，cls_targets
        # 和与之对应的权值 reg_weights 和 cls_weights
        for cls_name, cls_anchor in anchors.items():
            gt_mask = [torch.BoolTensor(c == cls_name).to(cls_anchor.device) for c in gt_types]
            # 这一顿操作的目的是召唤 Ground Truth
            labels, targets, ious = multi_apply(create_target_torch,
                                                cls_anchor, anchors_mask[cls_name],
                                                gt_bboxes, gt_labels, gt_mask,
                                                similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                                box_encoding_fn=second_box_encode,
                                                matched_threshold=cfg.assigner[cls_name].pos_iou_thr,
                                                unmatched_threshold=cfg.assigner[cls_name].neg_iou_thr,
                                                box_code_size=self._box_code_size)
            multi_labels.append(torch.stack(labels))
            multi_targets.append(torch.stack(targets))
            multi_anchors.append(cls_anchor)

        labels = torch.stack(multi_labels, 1)
        targets = torch.stack(multi_targets, 1)
        anchors = torch.stack(multi_anchors, 1)

        labels = labels.view(batch_size, -1)
        targets = targets.view(batch_size, -1, self._box_code_size)
        anchors = anchors.view(batch_size, -1, self._box_code_size)

        # 计算权重
        # 计算权值，计算方式跟辅助网络中的很相似
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)
        # cared 表示 labels >= 0 的 bool 张量
        # cls_targets 就是过滤掉 labels == -1 的张量
        cls_targets = labels * cared.type_as(labels)
        # 根据预测值，真值，权重，构建误差函数
        # 为了让 3d框 的回归变得更加准确，加入 _encode_rad_error_by_sin 更细致刻画 3d 框
        # loc_loss 是 3d框 的误差
        # cls_loss 是 3d框类别 的误差
        # 权值的意义：
        # 对于 loc_loss，我只关心车这一类的3d目标框，设置其他类和背景点的权值为零，滤除它们
        # 对于 cls_loss，正样本和负样本数量差异太大，比如正样本（是车的目标）太少，
        # 需要加大它误差对应的权值，提高网络对车识别的准确率

        # 位置误差：预测值是 box_preds， 真值是 reg_targets，权值是 cls_targets，使用weighted_smoothl1
        # 类别误差：预测值是 cls_preds， 真值是 reg_weights，权值是 cls_weights，使用weighted_sigmoid_focal_loss
        loc_loss, cls_loss = self.create_loss(
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights,
            reg_targets=targets,
            reg_weights=reg_weights,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            use_sigmoid_cls=self._use_sigmoid_cls,
            box_code_size=self._box_code_size,
        )

        # 计算平均然后相加
        loc_loss_reduced = loc_loss / batch_size
        loc_loss_reduced *= 2

        cls_loss_reduced = cls_loss / batch_size
        cls_loss_reduced *= 1

        loss = loc_loss_reduced + cls_loss_reduced

        # 朝向分类是一个分类问题，用交叉熵很正常
        if self._use_direction_classifier:
            # 生成与 dir_cls_preds 对应的真值 dir_labels
            dir_labels = self.get_direction_target(anchors, targets, use_one_hot=False).view(-1)
            dir_logits = dir_cls_preds.view(-1, 2)
            # 设置权值是为了仅仅考虑 labels > 0 的目标（即车这一类）
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 使用交叉熵做朝向预测的误差损失函数
            dir_loss = weighted_cross_entropy(dir_logits, dir_labels,
                                              weight=weights.view(-1),
                                              avg_factor=1.)

            dir_loss_reduced = dir_loss / batch_size
            dir_loss_reduced *= .2
            loss += dir_loss_reduced

        return dict(rpn_loc_loss=loc_loss_reduced, rpn_cls_loss=cls_loss_reduced, rpn_dir_loss=dir_loss_reduced)

    # anchors_mask 是 （1408*1600*2，1） 的 bool 型向量
    # anchors 是 （1600*1408*2，7） 的张量
    # box_preds, cls_preds, dir_cls_preds 是 [N, H, W，C] 的张量
    # 每个变量的 C 值都不一样，分别是 7， num_class， 2
    # N 是 batch size
    def get_guided_anchors(self, box_preds, cls_preds, dir_cls_preds, anchors, anchors_mask, gt_bboxes, gt_labels, thr=.1):
        batch_size = box_preds.shape[0]

        if isinstance(anchors, dict):
            anchors = torch.cat([v for v in anchors.values()], 1)

        if isinstance(anchors_mask, dict):
            anchors_mask = torch.cat([v for v in anchors_mask.values()], 1)
        # batch_box_preds 是 [N, H*W，7] 的张量
        batch_box_preds = box_preds.view(batch_size, -1, self._box_code_size)
        # batch_anchors_mask 是 [N, 1600*1408*2] 的张量
        batch_anchors_mask = anchors_mask.view(batch_size, -1)
        batch_cls_preds = cls_preds.view(batch_size, -1, self._num_class)
        batch_box_preds = second_box_decode(batch_box_preds, anchors)

        if self._use_direction_classifier:
            batch_dir_preds = dir_cls_preds.view(batch_size, -1, 2)

        guided_anchors = []
        anchor_labels = []

        if gt_bboxes is None:
            gt_bboxes = [None] * batch_size

        if gt_labels is None:
            gt_labels = [None] * batch_size
        # zip 打包遍历，感觉是遍历 N 遍，即 batch_size 的次数
        for box_preds, cls_preds, dir_preds, a_mask, gt_boxes, gt_lbls, in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask, gt_bboxes, gt_labels
        ):
            # 从函数名上理解，这段代码是获取 Guided Anchor，
            # 这一段代码我看的不是特别懂，但是我知道这一段的意思
            # 首先，把跟网络初次预测的 3d框 跟 Anchor_mask 下的 Anchor比较
            #      把重叠度高的 Anchor 保留下来；
            # 其次，这些 Anchor 对应的网络初次预测的 3d框 所对应的cls_preds 用 sigmoid 处理一遍，
            #      把高于阈值 thr 的 Anchor 框保留下来
            # 再者，如果是训练阶段，有 3d框 的真值
            # 就对每一个 Guided Anchor 贴上一个 3d框 的真值
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]

            if self._use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self._use_sigmoid_cls:
                total_scores = torch.sigmoid(cls_preds)
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            if self._num_class == 1:
                top_scores = torch.squeeze(total_scores, -1)
                top_labels = torch.zeros(total_scores.shape[0], dtype=torch.int64)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            selected = top_scores > thr

            box_preds = box_preds[selected]
            top_labels = top_labels[selected]

            if self._use_direction_classifier:
                dir_labels = dir_labels[selected]
                #opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                box_preds[opp_labels, -1] += np.pi

            # add ground-truth
            if gt_boxes is not None:
                box_preds = torch.cat([gt_boxes, box_preds],0)
                top_labels = top_labels.to(gt_lbls.device)
                top_labels = torch.cat([gt_lbls, top_labels],0)

            anchor_labels.append(top_labels)
            guided_anchors.append(box_preds)

        return guided_anchors, anchor_labels

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(win, -1), y.view(win, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1

    return torch.nn.functional.grid_sample(image, samples)

class PSWarpHead(nn.Module):
    # 根据 cfg 文件，grid_offsets = (0., 40.)，featmap_stride = 0.4，
    # in_channels = 256， num_parts = 28， num_class = 1
    def __init__(self, grid_offsets, featmap_stride, in_channels, num_class=1, num_parts=49):
        super(PSWarpHead, self).__init__()
        self._num_class = num_class
        out_channels = num_class * num_parts

        # 应该是定义采样区域的函数
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=grid_offsets, spatial_scale=1 / featmap_stride)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=False)
        )

    # guided_anchors 来自 bbox_head，参考图 3
    # guided_anchors 大概是根据置信度做的筛选
    def forward(self, x, guided_anchors, is_test=False):
        x = self.convs(x)
        bbox_scores = list()
        # 对每一个候选 Anchor
        for i, ga in enumerate(guided_anchors):
            if len(ga) == 0:
                bbox_scores.append(torch.empty(0).type_as(x))
                continue
            # 采样出 K 个区域
            (xs, ys) = self.gen_grid_fn(ga[:, [0, 1, 3, 4, 6]])
            im = x[i]
            # 做类似 ROIAlign 操作
            out = bilinear_interpolate_torch_gridsample(im, xs, ys)
            # 计算把 K 个区域的特征的平均值
            score = torch.mean(out, 0).view(-1)
            bbox_scores.append(score)

        # 如果是推断阶段，还会把 guided_anchors 留下来，后续还会使用，参考图 3
        if is_test:
            return bbox_scores
        else:
            return torch.cat(bbox_scores, 0)


    def loss(self, cls_preds, gt_bboxes, gt_labels, anchors, cfg):

        batch_size = len(anchors)
        batch_none = (None, ) * batch_size

        # currently only support rescoring for class agnostic anchors
        # 这一顿操作的目的是召唤 Ground Truth
        labels, targets, ious = multi_apply(create_target_torch,
                                            anchors, batch_none, gt_bboxes, batch_none, batch_none,
                                            similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                            box_encoding_fn = second_box_encode,
                                            matched_threshold=cfg.assigner.pos_iou_thr,
                                            unmatched_threshold=cfg.assigner.neg_iou_thr)

        labels = torch.cat(labels,).unsqueeze_(1)

        # soft_label = torch.clamp(2 * ious - 0.5, 0, 1)
        # labels = soft_label * labels.float()

        cared = labels >= 0
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(torch.float32)
        cls_weights = negative_cls_weights + positives.type(torch.float32)

        pos_normalizer = positives.sum().type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = labels * cared.type_as(labels)
        cls_preds = cls_preds.view(-1, self._num_class)

        cls_losses = weighted_sigmoid_focal_loss(cls_preds, cls_targets.float(), \
                                                 weight=cls_weights, avg_factor=1.)

        cls_loss_reduced = cls_losses / batch_size

        return dict(loss_cls=cls_loss_reduced,)

    def get_rescore_bboxes(self, guided_anchors, cls_scores, anchor_labels, img_metas, cfg):

        det_bboxes = list()
        det_scores = list()
        det_labels = list()

        for i in range(len(img_metas)):
            bbox_pred = guided_anchors[i]
            scores = cls_scores[i]
            labels = anchor_labels[i]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            bbox_pred = bbox_pred.view(-1, 7)
            scores = torch.sigmoid(scores).view(-1)
            select = scores > cfg.score_thr

            bbox_pred = bbox_pred[select, :]
            scores = scores[select]
            labels = labels[select]

            if scores.numel() == 0:

                det_bboxes.append(None)
                det_scores.append(None)
                det_labels.append(None)

                continue

            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]
            labels = labels[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())
            det_labels.append(labels.detach().cpu().numpy())

        return det_bboxes, det_scores, det_labels
