import copy
from collections import Sequence

import mmcv
from mmcv.runner import obj_from_dict
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .. import datasets
from mmdet.core.point_cloud import voxel_generator
from mmdet.core.point_cloud import point_augmentor
from mmdet.core.bbox3d import bbox3d_target
from mmdet.core.anchor import anchor3d_generator
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return [to_tensor(d) for d in data]
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif data is None:
        return data
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()

# 瞧瞧函数get_dataset，它的核心操作是调用函数obj_from_dict（大概是根据输入需求写data_info，然后从datasets读出dset，添加至dsets。dsets是输出的训练数据）
def get_dataset(data_cfg):

    # 生成index文件的实例，'ann_file'是data_root + 'ImageSets/train.txt'
    # num_dset 就是训练数据总数
    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    # SA-SSD没有使用它，算法不需要图像，'img_prefix'=None
    # 按照else，生成 N 个 None
    # 如果需要RGB的话，可以在cfg中写img_prefix=data_root + 'train2017/'相应路径
    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    # obj_from_dict函数的作用是Initialize an object from dict，通俗理解是根据字典型变量info去指定初始化一个parrent类对象。
    # 说白了，就是字典型变量中储存了类的初始化变量。核心调用是getattr。总之，obj_from_dict是一种做指定初始化的功能函数。
    # 按照data_cfg['generator']的参数，初始化voxel_generator，用于预处理点云体素化
    if 'generator' in data_cfg.keys() and data_cfg['generator'] is not None:
        generator = obj_from_dict(data_cfg['generator'], voxel_generator)
    else:
        generator = None

    # 按照data_cfg['augmentor']的参数，初始化point_augmentor，用于提供3D目标真值
    if 'augmentor' in data_cfg.keys() and data_cfg['augmentor'] is not None:
        augmentor = obj_from_dict(data_cfg['augmentor'], point_augmentor)
    else:
        augmentor = None

    # 按照data_cfg['anchor_generator']的参数，初始化anchor3d_generator，用于提供3DAnchor
    if 'anchor_generator' in data_cfg.keys() and data_cfg['anchor_generator'] is not None:
        anchor_generator = {cls: obj_from_dict(cfg, anchor3d_generator) for cls, cfg in data_cfg['anchor_generator'].items()}
    else:
        anchor_generator = None

    dsets = []
    # 装填用于训练的数据
    for i in range(num_dset):
        # 定义字典型变量data_info ，用于引导训练数据的装填
        # 使用copy，复制了cfg中的所有超参数，再改了改某几个参数
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        if generator is not None:
            data_info['generator'] = generator
        if anchor_generator is not None:
            data_info['anchor_generator'] = anchor_generator
        if augmentor is not None:
            data_info['augmentor'] = augmentor
        # 使用data_info去实例化datasets，根据data_info里面的设置，生成对应的dset
        # dest就是在这里调用了kittilidar类
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        # 从上述操作中，每一个训练数据都是一个datasets类
        # 使用ConcatDataset，把所有datasets类，统一变成一类datasets类
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset
    # dset由data_info生成，data_info中包括：ann_file配置文件， proposal_file辅助文件是None，img_prefix是None，
    # generator是完成了初始化的体素生成器，augmentor是数据增强器，anchor_generator是anchor生成器
    # anchor_generator完成初始化的anchor生成器，target_encoder是None


# def example_convert_to_torch(example, device=None) -> dict:
#     example_torch = {}
#     torch_names = [
#         'img', 'voxels','coordinates',\
#         # 'anchors_mask','anchors',\
#         #'gt_labels','gt_bboxes','gt_bboxes_ignore',\
#         'num_points', 'right', 'grid'
#     ]
#     for k, v in example.items():
#         if k in torch_names:
#             example_torch[k] = to_tensor(v)
#         else:
#             example_torch[k] = v
#
#     return example_torch

# def merge_second_batch(batch_list, samples_per_gpu=1, to_torch=True):
#     example_merged = defaultdict(list)
#     for example in batch_list:
#         for k, v in example.items():
#             example_merged[k].append(v)
#     ret = {}
#
#     for key, elems in example_merged.items():
#         if key in [
#             'voxels', 'num_points',
#         ]:
#             ret[key] = np.concatenate(elems, axis=0)
#         elif key == 'coordinates':
#             coors = []
#             for i, coor in enumerate(elems):
#                 coor_pad = np.pad(
#                     coor, ((0, 0), (1, 0)),
#                     mode='constant',
#                     constant_values=i)
#                 coors.append(coor_pad)
#             ret[key] = np.concatenate(coors, axis=0)
#         elif key in [
#             'img_meta', 'img_shape', 'calib', 'sample_idx', 'gt_labels', 'gt_bboxes','gt_bboxes_ignore'
#         ]:
#             ret[key] = elems
#         else:
#             ret[key] = np.stack(elems, axis=0)
#
#     if to_torch:
#         ret = example_convert_to_torch(ret)
#     return ret