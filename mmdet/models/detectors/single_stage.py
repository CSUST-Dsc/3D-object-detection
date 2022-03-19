import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F

# BaseDetector是所有检测器的基类，是虚基类
# RPNTestMixin 和 BBoxTestMixin 和 MaskTestMixin 用途不太明白，代码好像没有调用它们
# 总之， SingleStageDetector类继承自上述这些类
class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    # 单阶段目标检测由 Backbone， Neck， Bbox_head，Extra_head组成
    # 它们的实现需要设计者自己设计

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        # 初始化 Backbone
        self.backbone = builder.build_backbone(backbone)

        # 初始化 Neck
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        # 初始化 bbox_head
        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        # 初始化 extra_head
        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

        # 加载训练参数和测试参数（都是关于RPN参数的）
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 加载上次训练的模型
        self.init_weights(pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in ['voxels', 'num_points', ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key in ['coordinates', ]:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in ['img_meta', 'gt_labels', 'gt_bboxes', 'gt_types', ]:
                ret[key] = elems
            else:
                if isinstance(elems, dict):
                    ret[key] = {k: torch.stack(v, dim=0) for k, v in elems.items()}
                else:
                    ret[key] = torch.stack(elems, dim=0)
        return ret

    # 这是修改后的代码， 输入是点云， 不包含RGB图像
    def forward_train(self, img, img_meta, **kwargs):
        # img [1, 3, 384, 1248],第一个维度表示batch size

        # img_meta 数组：记录图像信息 数组长度是batch size 每一个数组元素是字典，包括标定参数、图像ID等
        # kwargs 字典：记录变量信息 包括预选框、体素中点、真值框、真值标签等,kwargs主要在kitti.py的prepare_train_img函数生成。
        batch_size = len(img_meta)

        # 处理多batch情况
        # 提取 Input 和 Ground Truth 3D框
        ret = self.merge_second_batch(kwargs)
        #  Neck的粗糙结构如下所示：
        #  输入点云 => Backbone Network => reshape 操作 => BEV Network => (x, conv6)
        #                    ||
        #                    || Tensor2Point (体素变点云)
        #                    ||
        #                 辅助网络层 => MLP层 => point_misc
        # 输入分析：
        # vx 可以理解为 pointclpoud_range 内的点云，包含 xyz 和雷达强度项，是 (N,4).
        # ret['coordinates'] 是 pointclpoud_range 内的点云体素化的结果，点对应的体素坐标
        # batch_size 是批处理的大小
        # 吐槽： ret['coordinates'] 才是真体素，如果我的理解有误，请大家多多指正
        #
        # 输出分析 ：
        # x, conv6 都是 BEV特征图，但是两个不同，两者中间还有一个卷积层
        # point_misc = (points_mean, point_cls, point_reg) 它是个元组
        # points_mean 是 bxyz 类型数据，xyz 是点云位置，b 是体素化后 z 轴分量， 它是（N，4）张量，为什么会有 b 这个分量，我也不太清楚，但是代码是这样写的
        # point_cls 是点云分类结果，它是（N，1）张量，用于前景分割（可不是3d目标分类呀）
        # point_reg 是点云回归结果，回归每一个3d类的中心位置，它是（N，3）张量
        #
        # 因为 SA-SSD 采用的是一个粗糙体素化处理方式，所以 vx  和 points_mean 的长度都是 N

        vx = self.backbone(ret['voxels'], ret['num_points'])
        x, conv6, point_misc = self.neck(vx, ret['coordinates'], batch_size, is_test=False)
        # 这里的x经过了三维卷积，大小变为了原来1/8，又经过reshape，减少了D维度，又经过二维卷积，最后（1， 256， 176， 200）
        # point_misc包括了(points_mean, point_cls, point_reg),原始点云数据，语义分割结果，中心点结果
        losses = dict()
        # point_misc = (points_mean, point_cls, point_reg)
        # points_mean 是 [N,4] 的张量，4 指 bxyz
        # point_cls 是 [N,1] 的张量，判别前景/后景
        # point_reg 是 [N,3] 的张量，预测3d目标中心点

        # 点前景/后景分类误差，分类问题，使用focal loss损失函数
        # 3d目标中心点回归误差，回归问题，使用smooth L1损失函数
        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)   #这里代表三个输出box_preds, cls_preds, dir_cls_preds[B, 200, 176, 14],[B, 200, 176, 2],[B, 200, 176, 4]
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['gt_types'],\
                            ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            # rpn_outs  = [box_preds, cls_preds, dir_cls_preds]
            # dir_cls_preds 是方向分类，分为面向相机，和背对相机两类
            # 记 N 是 Batch Size
            # box_preds 是一个 [N, y(H), x(W)，C]的张量，C = 7，用7个变量表示一个 box
            # cls_preds 是一个 [N, y(H), x(W)，C]的张量，C = num_class，如果只识别车的话，那就一类
            # dir_cls_preds 是一个 [N, y(H), x(W)，2]的张量
            # y(H), x(W) 是从BEV视图下 y 轴 和 x 轴的坐标分量
            # x 轴范围是 0~70.4m，y 轴范围是 -40.0~40.0
            # H 和 W 可不是什么相机成像面尺寸啥的，H=1408，W=1600，是体素化的范围
            guided_anchors, _ = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'],\
                        ret['anchors_mask'], ret['gt_bboxes'], ret['gt_labels'], thr=self.train_cfg.rpn.anchor_thr)
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            bbox_score = self.extra_head(conv6, guided_anchors)
            refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
            refine_losses = self.extra_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors, anchor_labels = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, None, thr=.1)

        bbox_score = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores, det_labels = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, anchor_labels, img_meta, self.test_cfg.extra)

        results = [kitti_bbox2results(*param, class_names=self.class_names) for param in zip(det_bboxes, det_scores, det_labels, img_meta)]

        return results



