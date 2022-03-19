import spconv
from torch import nn
from mmdet.models.utils import change_default_args, Sequential
from mmdet.ops.pointnet2 import pointnet2_utils
import torch
from mmdet.ops import pts_in_boxes3d
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss
from mmdet.core import tensor2points
import torch.nn.functional as F


class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape, # cfg中，output_shape=[40, 1600, 1408]
                 num_input_features=4,
                 num_hidden_features=128, # cfg中，num_hidden_features=64 * 5,
                 ):

        super(SpMiddleFHD, self).__init__()

        print(output_shape)
        self.sparse_shape = output_shape

        self.backbone = VxNet(num_input_features)
        self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)

        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

    def _make_layer(self, conv2d, bachnorm2d, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            conv2d(inplanes, planes, 3, stride=stride),
            bachnorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(conv2d(planes, planes, 3, padding=1))
            block.add(bachnorm2d(planes))
            block.add(nn.ReLU())
        return block, planes

    # nxyz是 [A,4] 的张量，4 指 bxyz
    # gt_boxes3d 是长度为 N 的元组
    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        # 遍历每一个 3D 目标真值框
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            # 把真值 3d 框内的点作为前景点，以及返回这个 3d 框的中心位置
            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            # 收集结果
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    # points 是 [A,4] 的张量，4 指 bxyz
    # point_cls 是 [A,1] 的张量，判别前景/后景
    # point_reg 是 [A,3] 的张量，预测3d目标中心点
    # gt_bboxes 真值 3D目标框,gt_bboxes 是长度为 N 的元组
    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)  # 这份点云中有 N 个车类目标

        # 生成点云前景/后景的真值 pts_labels [A,1] bool 型
        # 生成3d目标中心点的真值 center_targets [A,3]
        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()  # 转 float 型
        pos = (pts_labels > 0).float()  # 获取前景点索引向量
        neg = (pts_labels == 0).float()  # 获取背景点索引向量

        pos_normalizer = pos.sum()   # 前景点总数
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)  # 前景点总数必须大于等于　１

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos  # 回归中心点，肯定是在预测为前景点的点云做回归的
        reg_weights = reg_weights / pos_normalizer

        # 对于正负样本不均衡的数据中使用 Focal Loss 做分类问题的损失函数
        # 对于一个大点云来说，显然是背景点要多很多
        # 分割点云损失函数，使用加权 sigmoid_focal_loss
        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N
        # 回归问题业界用 smooth l1
        # 为什么这里要加权呢，是因为要滤去背景点的回归结果，只计算前景点的回归结果
        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )

    # voxel_features 是 pointclpoud_range 内的点云和雷达强度项 （N，4）张量
    # coors 是 pointclpoud_range 内的点云体素化的结果 [z,y,x]
    # coors 体素顺序为什么是 [z,y,x] 呢？ 可以追溯变量，一直到 KITTILiDAR 中的 prepare_train_img
    def forward(self, voxel_features, coors, batch_size, is_test=False):

        points_mean = torch.zeros_like(voxel_features)   # points_mean 记录了batch id + 3维平均坐标
        points_mean[:, 0] = coors[:, 0]   # 获得batch id
        points_mean[:, 1:] = voxel_features[:, :3]   # 获得每个体素的前三个特征：3个点云的平均坐标

        coors = coors.int()
        # voxel_features即非空体素的特征,coors 即记录了非空体素的实际位置,sparse_shape 值为  [40,1600,1408]，在配置文件car_cfg.py中设置,即 D, H, W
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)   # 真正的变成了体素!/初始化SparseConvTensor,3 layer 1/8
        x, middle = self.backbone(x)  # x: [5, 200, 176]  middle: [20, 800, 704] [10, 400, 352] [5, 200, 176]

        # 这一段对应图框图中的 Reshape
        x = x.dense()  # [1, 64, 5, 200, 176] 第一个维度是batch size
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)  # [1, 320, 200, 176],64*5=320

        # 把 Reshape 后的特征喂入 BEVNet 中,self.fcn就是BEVNet
        x, conv6 = self.fcn(x)  # x: [1, 256, 200, 176], conv6: [1, 256, 200, 176]

        if is_test:
            return x, conv6
        else:
            # auxiliary network
            # 反体素，之前有 down0 的降采样，用到的体素尺寸翻了一倍
            vx_feat, vx_nxyz = tensor2points(middle[0], (0, -40., -3.), voxel_size=(.1, .1, .2))
            # nearest_neighbor_interpolate大概是近邻点加权平均求特征的方法。它的实现依据PointNet++中的interpolation实现
            p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            # 反体素，之前有 down1 的降采样，用到的体素尺寸翻了一倍
            vx_feat, vx_nxyz = tensor2points(middle[1], (0, -40., -3.), voxel_size=(.2, .2, .4))
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            # 反体素，之前有 down2 的降采样，用到的体素尺寸翻了一倍
            vx_feat, vx_nxyz = tensor2points(middle[2], (0, -40., -3.), voxel_size=(.4, .4, .8))
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            # 辅助网络的输出，回归每个点是不是3D目标，以及利用每一个点回归3D目标中心点
            # points_misc 是 (points_mean, point_cls, point_reg) 的统称
            # point_fc，point_cls，point_reg都是简单的线性层

            # torch.cat([p1, p2, p3]) 是 [N, C1+C2+C3] 的张量
            # pointwise 是 [N, 64] 的张量
            pointwise = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
            point_cls = self.point_cls(pointwise)  # [N, 1] 的张量，预测是否是前景/背景
            point_reg = self.point_reg(pointwise)   # [N, 3] 的张量，预测3d目标的中心位置

            return x, conv6, (points_mean, point_cls, point_reg)


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, (2, 2, 2), padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
    )
# known 表示已知点的位置信息 [m,4]
# known_feats 表示已知点的特征信息 [m,C]
# unknown 表示需要插值点的位置信息 [n,4]，一般来所，n>m
# interpolated_feats 表示需要插值点的特征信息 [n,C]，这是返回结果
def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    # 获取 unknown 和 known 之间的近邻关系和距离信息
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    # 权值是距离的倒数
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    # 根据近邻关系以及距离信息，直接插值特征信息
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats


class VxNet(nn.Module):

    def __init__(self, num_input_features):
        super(VxNet, self).__init__()

        self.conv0 = double_conv(num_input_features, 16, 'subm0')

        self.down0 = stride_conv(16, 32, 'down0')
        self.conv1 = double_conv(32, 32, 'subm1')

        self.down1 = stride_conv(32, 64, 'down1')
        self.conv2 = triple_conv(64, 64, 'subm2')

        self.down2 = stride_conv(64, 64, 'down2')
        self.conv3 = triple_conv(64, 64, 'subm3')  # middle line

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        middle = list()

        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub
        middle.append(x)

        x = self.down1(x)
        x = self.conv2(x)
        middle.append(x)

        x = self.down2(x)
        x = self.conv3(x)
        middle.append(x)

        out = self.extra_conv(x)
        return out, middle

class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        BatchNorm2d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        self.conv0 = Conv2d(in_features, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.ca = ChannelAttention(num_filters)  #
        self.sa = SpatialAttention()  #

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.ca1 = ChannelAttention(num_filters)  #
        self.sa1 = SpatialAttention()  #

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)

        self.conv6 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = BatchNorm2d(num_filters)

        self.ca2 = ChannelAttention(num_filters)  #
        self.sa2 = SpatialAttention()  #

        self.conv7 = Conv2d(num_filters, num_filters, 1)
        self.bn7 = BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.conv5(x)
        x = F.relu(self.bn5(x), inplace=True)
        x = self.conv6(x)
        x = F.relu(self.bn6(x), inplace=True)
        x = self.ca2(x) * x  #
        x = self.sa2(x) * x  #
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


