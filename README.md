# 3D vehicle detection
### 基于注意力机制的3D点云车辆检测算法研究
### [发表的论文](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=XTYY202112027&uniplatform=NZKPT&v=gSJgB_xrLlCX7z5SdaHhvPrm18tW3zDGfduFyUc80twkh42cdD2ebf-NKLkOl-DA)

### 算法模型图

![图0](https://github.com/CSUST-Dsc/3D-object-detection/blob/main/results/result0.png)

![图-1](https://github.com/CSUST-Dsc/3D-object-detection/blob/main/results/result-1.png)

## 实验代码环境配置
### Environment
```python
   Ubuntu 16.04
   Python 3.7
   pytorch 1.4.0
   torchvision 0.5.0
   CUDA 10.0
```
### Dependencies
```python
   python3.5+
   opencv
   shapely
   mayavi
   spconv (v1.0)
```
### 实际安装命令
1.基础安装
```python
   pip install opencv-python
   pip install shapely
   pip install mayavi
   pip install scikit-image
   pip install numba
   pip install matplotlib
   pip install Cython
   pip install terminaltables
   pip install tqdm
   pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
   pip install pybind11 
```
2.spconv安装

spconv 安装过程涉及cmake安装，参考[Ubuntu安装cmake](https://blog.csdn.net/weixin_38362784/article/details/109532934)
```python
   sudo apt-get install libboost-all-dev
   git clone https://github.com/traveller59/spconv.git --recursive
   cd spconv && git checkout 7342772
   python setup.py bdist_wheel
   cd ./dist && pip install *
```
### Installation
1.mmdet/ops中，编译 C++/CUDA模块
```python
   cd mmdet/ops/points_op
   python setup.py build_ext --inplace
   cd mmdet/ops/iou3d
   python setup.py build_ext --inplace
   cd mmdet/ops/pointnet2
   python setup.py build_ext --inplace
```
2.~/.bashrc中，加环境变量
```python
   export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
   export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
   export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
   export LD_LIBRARY_PATH=/home/ch511/anaconda3/envs/sassd/lib/python3.7/site-packages/spconv;
```
3.安装mmcv
```python
   pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install mmcv -i https://pypi.tuna.tsinghua.edu.cn/simple
```
4.安装mmdet

即将本地项目中的mmdet加入site-package中，使得项目可以直接调用。源码中没有setup.py，可以用使用mmdetection的setup.py（将requirement相关注释掉）。
```python
   python setup.py develop
```
## Data Preparation
1.下载数据集

[KITTI数据集官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)下载如下数据：
- Velodyne point clouds (29 GB): input data to VoxelNet
- Training labels of object data set (5 MB): input label to VoxelNet
- Camera calibration matrices of object data set (16 MB): for visualization of predictions
- Left color images of object data set (12 GB): for visualization of predictions

2.整理KITTI数据集目录

注意需要创建空的velodyne_reduced文件夹，用来放置后续筛选得到的视锥体内的点云数据。
```python
└── KITTI
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   ├── calib 
       |   ├── velodyne
       |   └── velodyne_reduced			# empty folder
       └── testing  <--- testing data
       |   ├── image_2
       |   ├── calib
       |   ├── velodyne
       |   └── velodyne_reduced			# empty folder
```
3.下载ImageSets

下载地址：https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz

放置位置：放在KITTI目录下

4.生成数据集
```python
   python tools/create_data.py
```
## Train & Eval
Train Model with single GPU
```python
   python ./tools/train.py ./configs/car_cfg.py
```
Eval Model with single GPU
```python
   python ./tools/test.py ./configs/car_cfg.py ./work_dir/checkpoint_epoch_90.pth 
```
## 评估结果
### KITTI离线测试集训练90Epoch的结果展示

![图1](https://github.com/CSUST-Dsc/3D-object-detection/blob/main/results/result1.png)

### [提交KITTI线上测试集结果展示](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=81e546a3424a05104244c680b3274621cdb73a61)

![图2](https://github.com/CSUST-Dsc/3D-object-detection/blob/main/results/result2.png)

![图3](https://github.com/CSUST-Dsc/3D-object-detection/blob/main/results/result3.png)






