import numpy as np

def create_anchors_3d_stride(feature_size,    # 是 [1, 1600, 1408]
                             sizes=[1.6, 3.9, 1.56],  # 单个 Anchor 的大小
                             anchor_strides=[0.4, 0.4, 0.0],  # 指每个 Anchor 的间距 cfg 中是 [0.4, 0.4, 1.0]
                             anchor_offsets=[0.2, -39.8, -1.78],  # 分别是anchor中心的阈值，中心从这里开始生成
                             rotations=[0, np.pi / 2],  # 角度，0和90的弧度值
                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    # 步长和下限
    x_stride, y_stride, z_stride = anchor_strides    # 分别是 0.4，0.4，1.0
    x_offset, y_offset, z_offset = anchor_offsets    # 分别是 0.2，-39.8，-1.78
    # 基准值
    z_centers = np.arange(feature_size[0], dtype=dtype)  # 生成数组，0
    y_centers = np.arange(feature_size[1], dtype=dtype)  # 生成数组，0，1，...,1600-1
    x_centers = np.arange(feature_size[2], dtype=dtype)  # 生成数组，0，1，...,1408-1
    # 真实的zyx中心坐标
    # 这里算 center 是有问题的，y_centers 可以到 599.8m，实际上雷达测不到这么远
    z_centers = z_centers * z_stride + z_offset  # -1.78
    y_centers = y_centers * y_stride + y_offset  # -39.8，-39.4，...，599.8
    x_centers = x_centers * x_stride + x_offset  # 0.2，0.6，...，563.0
    # 单个 Anchor 的大小有多少种 N x 3---->1 x 3
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])   # 变成 1*3 张量，如果要生成 N 种大小的 Anchor，就会有 N*3 张量
    # 单个 Anchor 的角度有多少种 N
    rotations = np.array(rotations, dtype=dtype)
    # 生成网格点,用上面的数据生成了每个网格中心点的坐标，上面都是分开的
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5  # 等价于 [1,1,1,1,1]
    tile_shape[-2] = int(sizes.shape[0])  # 等价于 [1,1,1,N,1],N种anchor的size
    # 大概遍历 1408 次，下面这段代码比较难懂
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    # sizes从[N, 3]---->[1, 1, 1, N, 1, 3]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    # 输出结果是 (1, 1600, 1408, 1, 2, 7) 的张量
    # 第零维没啥说的，可能是batch
    # 第一维是 anchor 在 y 轴上的序号 0~1600-1
    # 第二维是 anchor 在 y 轴上的序号 0~1600-1
    # 第三维是 anchor 在 x 轴上的序号 0~1408-1
    # 第四维是 anchor 的类别，只生成 car，所以只有这一类
    # 第五维是 anchoe 的转角，只生成了 0 度和 90 度，这两类
    # 第六维是 anchor 的7个，第7个为 Yaw 旋转角，前六个是 xyz 和 wlh
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])  # 前5个维度表示这个框的整体信息，最后一个维度表示这个框的实际形状


def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=[1.6, 3.9, 1.56],
                            rotations=[0, np.pi / 2],
                            dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])

class AnchorGeneratorStride:
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype

    @property
    def num_anchors_per_localization(self):  # num_anchors_per_localization是计算每个位置（体素点）上有多少个anchor
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def __call__(self, feature_map_size):
        return create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._dtype)

class AnchorGeneratorRange:
    def __init__(self,
                 anchor_ranges,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def __call__(self, feature_map_size):
        return create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)
