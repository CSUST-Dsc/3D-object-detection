from __future__ import division
import argparse
import sys
sys.path.append('/media/t640admin/Disk2/dsc/SA-SSD')
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataloader
from tools.env import get_root_logger, init_dist, set_random_seed
from tools.train_utils import train_model
import pathlib
from mmcv import Config
from mmdet.datasets import get_dataset
from mmdet.models import build_detector
from tools.train_utils.optimization import build_optimizer, build_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=10)

    args = parser.parse_args()

    return args



def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.work_dir)

    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # mmdetection的训练流程大致分为三步：
    # 第一步，初始化Detector，对应函数build_detector
    # cfg.model指模型的超参数
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # 分布式并行计算和单机并行计算
    if distributed:
        model = MMDistributedDataParallel(model.cuda())
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # 第二步，加载训练数据集，对应函数get_dataset
    # get_dataset，用cfg中的参数信息，通过datasets文件夹，生成包含所有数据的一个数据集
    # 再本次实例中，obj_from_dict，更有深层次的理解。根据字典型变量info去指定初始化一个parrent类对象。如果parrent类是一个虚类，
    # 它会根据info的变量自动地匹配一个Matched的子类，去指定初始化这个子类的实例。
    # 毫无疑问，肯定是生成datasets类子类中的KittiLiDAR类
    train_dataset = get_dataset(cfg.data.train)

    optimizer = build_optimizer(model, cfg.optimizer)

    train_loader = build_dataloader(
        train_dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        dist=distributed)

    start_epoch = it = 0
    last_epoch = -1

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfg.total_epochs,
        last_epoch=last_epoch, optim_cfg=cfg.optimizer, lr_cfg=cfg.lr_config
    )
    # -----------------------start training---------------------------
    logger.info('**********************Start training**********************')
    # 第三步，训练Detector，对应函数train_detector
    # 开始训练
    # mmdetection的一行代码就训练，然而这种简洁的操作多少有些惊艳
    # 后来发现，损失函数（smooth l1），优化器（SGD），学习率，训练批次等都在car_cfg.py定义好啦
    train_model(
        model,
        optimizer,
        train_loader,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.optimizer,
        start_epoch=start_epoch,
        total_epochs=cfg.total_epochs,
        start_iter=it,
        rank=args.local_rank,
        logger = logger,
        ckpt_save_dir=cfg.work_dir,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=cfg.checkpoint_config.interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        log_interval = cfg.log_config.interval
    )

    logger.info('**********************End training**********************')





if __name__ == '__main__':
    main()
