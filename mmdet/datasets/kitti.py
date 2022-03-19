import os.path as osp
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
from torch.utils.data import Dataset
from mmdet.datasets.transforms import (ImageTransform, BboxTransform)
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.kitti_utils import read_label, read_lidar, \
    project_rect_to_velo, Calibration, get_lidar_in_image_fov, \
    project_rect_to_image, project_rect_to_right, load_proposals
from mmdet.core.bbox3d.geometry import rbbox2d_to_near_bbox, filter_gt_box_outside_range, \
    sparse_sum_for_anchors_mask, fused_get_anchors_area, limit_period, center_to_corner_box3d, points_in_rbbox
import os
from mmdet.core.point_cloud.voxel_generator import VoxelGenerator
from mmdet.ops.points_op import points_op_cpu

class KittiLiDAR(Dataset):
    def __init__(self, root, ann_file,  # 数据路径，配置文件路径
                 img_prefix,
                 img_norm_cfg,  # 图片归一化数据
                 img_scale=(1242, 375),  # 图片的大小
                 size_divisor=32,  # 标准化之后的图片大小都是32的倍数
                 proposal_file=None,
                 flip_ratio=0.5,  # 翻转机率
                 with_point=False,  # 使用点云
                 with_mask=False,  # 使用mask
                 with_label=True,  # 使用label
                 with_plane=False,
                 class_names = ['Car', 'Van'],  # 只关注这些种类的label
                 augmentor=None,  # 数据增强
                 generator=None,  # 点云生成体素的生成器
                 anchor_generator=None,   # 生成anchor的生成器
                 anchor_area_threshold=1,    # anchor的阈值，anchor里面有大于阈值数量的点，认为是有效anchor
                 target_encoder=None,
                 out_size_factor=2,  # 输出的anchor与体素规模的比例
                 test_mode=False):
        self.root = root
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # flip ratio
        self.flip_ratio = flip_ratio

        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        self.class_names = class_names
        self.test_mode = test_mode
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_point = with_point
        self.with_plane = with_plane
        # 获取KITTI相关各种数据的前缀路径
        self.img_prefix = osp.join(root, 'image_2')
        self.right_prefix = osp.join(root, 'image_3')   # 没有用到
        self.lidar_prefix = osp.join(root, 'velodyne_reduced')  # 注意这里用的reduce后的点云，打脸了
        self.calib_prefix = osp.join(root, 'calib')  # 校准器
        self.label_prefix = osp.join(root, 'label_2')   # label
        self.plane_prefix = osp.join(root, 'planes')

        with open(ann_file, 'r') as f:
            self.sample_ids = list(map(int, f.read().splitlines()))

        # 根据图像宽高比设置标志 宽高比大于1的图像将被设置为组1，否则组0。
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)

        self.augmentor = augmentor
        self.generator = generator
        self.target_encoder = target_encoder
        self.out_size_factor = out_size_factor
        self.anchor_area_threshold = anchor_area_threshold

        # anchor
        # anchor 生成anchor box，SSD有自己生成随机框的方法，同时也生成鸟瞰图里的anchor，二维框
        if anchor_generator is not None:
            # 由第二节讨论，grid_size是 [1408，1600，40]
            # feature_map_size  应该指 xy 平面上的空间区域，记为 [1408，1600]
            feature_map_size = self.generator.grid_size[:2] // self.out_size_factor
            # [1408，1600] => [1408，1600, 1] => [1, 1600, 1408]
            feature_map_size = [*feature_map_size, 1][::-1]
            # 喂入 [1, 1600, 1408] 生成 anchors
            # 它是 (1, 1600, 1408, 1, 2, 7) 的张量，
            # 2 表示旋转角度类别（ 0 和 90 度），7 表示 Anchor 参数，xyzwlh 以及 Yaw 旋转角
            # 在每一个体素都声成一个框
            if self.test_mode:
                # 7 个参数，分别是 xyzwlh 和 Yaw 旋转角
                # self.anchors 是 （1600*1408*2，7） 的张量
                self.anchors = np.concatenate([v(feature_map_size).reshape(-1, 7) for v in anchor_generator.values()],
                                              0)
                # 生成 BEV 视图下的 anchors_bv，仅仅使用 [0, 1, 3, 4, 6]
                # 使用了 xy wl 和 旋转角
                # rbbox2d 输出 [N, 4(xmin, ymin, xmax, ymax)] bboxes
                # self.anchors_bv 是 （1600*1408*2，4） 的张量
                self.anchors_bv = rbbox2d_to_near_bbox(self.anchors[..., [0, 1, 3, 4, 6]])
            else:
                self.anchors = {k: v(feature_map_size).reshape(-1, 7) for k, v in anchor_generator.items()}
                self.anchors_bv = {k: rbbox2d_to_near_bbox(v[:, [0, 1, 3, 4, 6]]) for k, v in self.anchors.items()}

        else:
            self.anchors = None

    def get_road_plane(self, plane_file):
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.sample_ids)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        sample_id = self.sample_ids[idx]

        # load image
        img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, 1, False)

        objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        gt_bboxes = [object.box3d for object in objects if object.type not in ["DontCare"]]
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_types = [object.type for object in objects if object.type not in ["DontCare"]]

        # transfer from cam to lidar coordinates
        if len(gt_bboxes) != 0:
            gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)

        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta = DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = {k: DC(to_tensor(v.astype(np.float32))) for k, v in self.anchors.items()}

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if self.with_plane:
            plane = self.get_road_plane(osp.join(self.plane_prefix, '%06d.txt' % sample_id))
        else:
            plane = None

        if self.augmentor is not None and self.test_mode is False:
            sampled_gt_boxes, sampled_gt_types, sampled_points = self.augmentor.sample_all(gt_bboxes, gt_types, plane, calib)
            # box的信息，box的class，box对应的点云
            ###########################################
            # 采样时是从kitti_dbinfos_train.pkl中进行采样的，同时在采样时，确保不会产生图像和点的重叠问题
            # 当gt_box中某个分类不够15个时，从文件中拿出来一些db，补进去
            ###########################################
            assert sampled_points.dtype == np.float32
            # 合并
            gt_bboxes = np.concatenate([gt_bboxes, sampled_gt_boxes])
            # 合并
            gt_types = gt_types + sampled_gt_types
            assert len(gt_types) == len(gt_bboxes)

            # to avoid overlapping point (option)
            # [num_points, num_box],表示某个点是否在某个box内部，points表示所有点，sampled_gt_boxes被抽样出来的gt_boxes
            # masks = points_op_cpu.points_in_bbox3d_np(points[:,:3], sampled_gt_boxes)
            masks = points_in_rbbox(points, sampled_gt_boxes)

            points = points[np.logical_not(masks.any(-1))]
            # 将所有不属于任意一个采样框的点拿出来，因为采样出来的框本身保证了与其他gt框没有重叠，这里避免其他非车的点与采样框对应的点的覆盖
            # paste sampled points to the scene
            points = np.concatenate([sampled_points, points], axis=0)
            # 拼接之后就是需要输入进训练模型的所有点
            # force van to have same type as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_types = np.array(gt_types)

            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = gt_types[selected]
            gt_labels = np.array([self.class_names.index(n) + 1 for n in gt_types], dtype=np.int64)
            # 使用数据增强后，输入数据和真值标签同时做变换
            # 下面是常见的四种数据增强方式：
            self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
            gt_bboxes, points = self.augmentor.random_flip(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_rotation(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_scaling(gt_bboxes, points)
            '''
            实际上augmentor包含了两个阶段，第一阶段sample，通过采样使得每一帧数据中都至少有15个敏感物体bbox，补全bbox的数量时，
            也避免了补上的bbox与原来bbox的碰撞，补上之后，又避免了补上的bbox里的点与不是敏感物体的点的碰撞；
            第二个部分就是几种典型的augmentor方式，添加噪声，翻转，旋转，裁剪。
            '''

        if isinstance(self.generator, VoxelGenerator):
            voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size    # 体素的分辨率，单位体素大小[0.05, 0.05, 0.1]
            pc_range = self.generator.point_cloud_range    # 有效点云的范围，在range内的点才会转化为体素[0, -40., -3., 70.4, 40., 1.]
            grid_size = self.generator.grid_size   # 生成后的体素的维度1408*1600*40

            #keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            #voxels = points[keep, :]
            #coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2,1,0]], dtype=np.float32)) / np.array(
            #    voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            #num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            # anchor_mask
            # 考虑到雷达点云是稀疏，尽管Anchor覆盖了整个BEV区域。显然，只有在有点云的地方，才有可能有3d目标。
            # 那些没有点云的空洞区域的Anchor是没啥用的。Anchor Mask的作用就是把覆盖点云的Anchor标记出来。
            # 在 cfg 文件中， self.anchor_area_threshold = 1
            if self.anchor_area_threshold >= 0 and self.anchors is not None:
                # coordinates 是 N*3 的张量，代表每个点云中每个点所在的体素的坐标
                # grid_size 是 [1408，1600，40]
                # tuple(grid_size[::-1][1:]） 是 [1600, 1408] 的元组
                # dense_voxel_map 是 1600*1408 的矩阵，
                # dense_voxel_map[i][j] = a，表示 (i,j) 区域内体素的个数为 a
                # dense_voxel_map 可以看作是体素分布的密度函数，只是再xy平面上的密度，没有考虑z维度的密度
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))   # 在这里舍去了z轴数据
                dense_voxel_map = dense_voxel_map.cumsum(0)  # 在第零轴上累加
                dense_voxel_map = dense_voxel_map.cumsum(1)  # 接着在第一轴上累加，得到 dense_voxel_map，还是 1600*1408 的矩阵
                # self.anchors_bv 是 BEV 视图下生成的 Anchors，是 （1600*1408*2，5） 的张量 (x y w h yaw)
                # voxel_size 是 [0.05, 0.05, 0.1]
                # pc_range 是 [0, -40., -3., 70.4, 40., 1.]
                # grid_size 是 [1408，1600，40]
                # anchors_area 是 1408*1600*2 的向量，之所以有*2是因为每个位置anchor都有0°和90°两个框，anchors_area里面放的是每个框包含点的密度
                # 累加操作可以看作是积分，两次累加，相当于在 x 轴和 y 轴做积分
                # 这时候 dense_voxel_map 是一个关于体素的分布函数
                anchors_mask = dict()
                for k, v in self.anchors_bv.items():
                    mask = fused_get_anchors_area(
                        dense_voxel_map, v, voxel_size, pc_range,
                        grid_size) > self.anchor_area_threshold
                    # mask anchor的生成过程全程没有点云的参与，就是根据步长依次生成的，覆盖了全部的区域，mask筛选出没点的框
                    # anchor_area_threshold = 1，说明只要 Anchor 里面有一个体素，就把它归入 Mask
                    # anchors_mask 是 1408*1600*2 的 bool 型向量
                    anchors_mask[k] = DC(to_tensor(mask.astype(np.bool)))
                data['anchors_mask'] = anchors_mask

            # filter gt_bbox out of range
            bv_range = self.generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(gt_bboxes, bv_range)
            gt_bboxes = gt_bboxes[mask]
            gt_types = gt_types[mask]
            gt_labels = gt_labels[mask]
            # 滤波掉超过range的gt
        else:
            NotImplementedError

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # limit rad to [-pi, pi]
        gt_bboxes[:, 6] = limit_period(
            gt_bboxes[:, 6], offset=0.5, period=2 * np.pi)

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes))
            data['gt_types'] = DC(gt_types, cpu_only=True)

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        sample_id = self.sample_ids[idx]
        # sample_id=8
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, 1, False)

        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        if self.with_label:
            objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
            gt_bboxes = [object.box3d for object in objects if object.type not in ["DontCare"]]
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_types = [object.type for object in objects if object.type not in ["DontCare"]]

            # transfer from cam to lidar coordinates
            if len(gt_bboxes) != 0:
                gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)

            # force van to have same type as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_types = np.array(gt_types)
            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = gt_types[selected]
            gt_labels = np.array([self.class_names.index(n) + 1 for n in gt_types], dtype=np.int64)


        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta=DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = DC(to_tensor(self.anchors.astype(np.float32)))

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if isinstance(self.generator, VoxelGenerator):
            voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            #keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            #voxels = points[keep, :]

            #coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2, 1, 0]], dtype=np.float32)) / np.array(
            #    voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            #num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            if self.anchor_area_threshold >= 0 and self.anchors is not None:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)

                anchors_mask = fused_get_anchors_area(
                    dense_voxel_map, self.anchors_bv, voxel_size, pc_range,
                    grid_size) > self.anchor_area_threshold

                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels), cpu_only=True)
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes), cpu_only=True)
            data['gt_types'] = DC(gt_types, cpu_only=True)
        else:
            data['gt_labels'] = DC(None, cpu_only=True)
            data['gt_bboxes'] = DC(None, cpu_only=True)
            data['gt_types'] = DC(None, cpu_only=True)

        return data

class KittiVideo(KittiLiDAR):
    ''' Load data for KITTI videos '''

    def __init__(self, img_dir, lidar_dir, calib_dir, **kwargs):
        super(KittiVideo, self).__init__(**kwargs)

        self.calib = Calibration(os.path.join(self.root, calib_dir), from_video=True)
        self.img_dir = os.path.join(self.root, img_dir)
        self.lidar_dir = os.path.join(self.root, lidar_dir)
        self.img_filenames = sorted([os.path.join(self.img_dir, filename) \
                                     for filename in os.listdir(self.img_dir)])

        self.lidar_filenames = sorted([os.path.join(self.lidar_dir, filename) \
                                       for filename in os.listdir(self.lidar_dir)])

        sample_ids = sorted([os.path.splitext(filename)[0] \
                             for filename in os.listdir(self.img_dir)])
        self.sample_ids = list(map(int, sample_ids))

    def prepare_test_img(self, idx):
        sample_id = self.sample_ids[idx]
        # load image
        img = mmcv.imread(self.img_filenames[idx])
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, 1, False)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_shape=DC(img_shape, cpu_only=True),
            sample_idx=DC(sample_id, cpu_only=True),
            calib=DC(self.calib, cpu_only=True)
        )

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(self.lidar_filenames[idx])
            points = get_lidar_in_image_fov(points, self.calib, 0, 0, img_shape[1], img_shape[0], clip_distance=0.1)

        if self.generator is not None:
            voxels, coordinates, num_points = self.generator.generate(points)
            data['voxels'] = DC(to_tensor(voxels))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))
            data['anchors'] = DC(to_tensor(self.anchors))

        return data




