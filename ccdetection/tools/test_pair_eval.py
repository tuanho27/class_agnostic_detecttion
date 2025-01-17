import argparse
import os
import os.path as osp
import shutil
import tempfile
import mmcv
import cv2
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from tqdm import tqdm
from terminaltables import AsciiTable
from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--data_root', help='data root')
    parser.add_argument('--work_dir', help='work dir')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    gt_pairs = []
    for i, data in enumerate(data_loader):
        ## img test draw
        # size = (900, 600)
        # img0_0 = mmcv.imread(data['img_meta'][0][0].data[0][0]['filename'])
        # img1_0 = mmcv.imread(data['img_meta'][1][0].data[0][0]['filename'])
        # img = np.concatenate((mmcv.imresize(img0_0, size), mmcv.imresize(img1_0, size)), axis=1) 
        # cv2.imwrite("./pair_test.jpg", img)

        gt_positive_pairs = []
        gt_negative_pairs = []
        gt_bbox_0 = data['gt_bboxes'][0][0].squeeze(0)
        gt_bbox_1 = data['gt_bboxes'][1][0].squeeze(0)

        for i in range(data['gt_labels'][0][0].size(1)):
            for j in range(data['gt_labels'][1][0].size(1)):
                if data['gt_labels'][0][0].view(-1)[i] == data['gt_labels'][1][0].view(-1)[j]:
                    gt_positive_pairs.append(torch.stack((gt_bbox_0[i], gt_bbox_1[j])))
                else:
                    gt_negative_pairs.append(torch.stack((gt_bbox_0[i], gt_bbox_1[j])))
        try:
            gt_pairs.append(torch.stack(gt_positive_pairs))
        except:
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            result, _ = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, show=False, out_file=f'cache/{i}.png')

        batch_size = data['img'][0][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, gt_pairs


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result,_ = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)

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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.test_cfg.mode = 'eval'
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        pred_pairs, gt_pairs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        pred_pairs, gt_pairs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(pred_pairs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            num_sample = len(pred_pairs)
            true_positive = 0
            num_gt = 0
            for i in tqdm(range(num_sample)):
                for gt_box in gt_pairs[i]:
                    num_gt+=1
                    for pred_box in pred_pairs[i]:
                        iou = bbox_overlaps(gt_box.cpu().numpy(),pred_box.cpu().numpy()[:,:4])  ##
                        if iou[0][0] > 0.5 and iou[1][-1] > 0.5:
                            true_positive+=1

            table_data = [['VOCTest', 'Recall', 'Precision'],
                          ['iou:0.5', round(true_positive/num_gt,4), round(true_positive/(num_sample*100),4)]] 
            print("\n--------------------------------------")
            print(AsciiTable(table_data).table)

if __name__ == '__main__':
    main()
