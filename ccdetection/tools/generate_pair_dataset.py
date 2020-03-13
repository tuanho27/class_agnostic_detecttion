from __future__ import division
import argparse
import os
# CUDA_LAUNCH_BLOCKING=1
import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from mmdet.datasets import DATASETS, build_dataloader
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
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
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--data_root', help='data root')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # update workdir and data_root:
    if args.work_dir:
        cfg.work_dir  = args.work_dir
    if args.data_root:
        cfg.data.train.ann_file  = cfg.data.train.ann_file.replace(cfg.data_root,args.data_root)
        cfg.data.train.img_prefix= cfg.data.train.img_prefix.replace(cfg.data_root,args.data_root)
        cfg.data.val.ann_file    = cfg.data.val.ann_file.replace(cfg.data_root,args.data_root)
        cfg.data.val.img_prefix  = cfg.data.val.img_prefix.replace(cfg.data_root,args.data_root)
        cfg.data.test.ann_file   = cfg.data.test.ann_file.replace(cfg.data_root,args.data_root)
        cfg.data.test.img_prefix = cfg.data.test.img_prefix.replace(cfg.data_root,args.data_root)
        cfg.data_root = args.data_root

    # Copy config file to work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    os.system("cp %s %s" % (args.config, cfg.work_dir))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    
    ## To Gen Pair dataset
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [ 
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            shuffle=False,
            dist=False) for ds in datasets]
    count = 0
    print(data_loaders)
    for i, data_batch in enumerate(data_loaders[0]):
        print("Generate ...")
        count +=1


if __name__ == '__main__':
    main()
