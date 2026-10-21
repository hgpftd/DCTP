import cv2
import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr_dctp.data.prefetch_dataloader import PrefetchDataLoader
from basicsr_dctp.utils import get_root_logger, scandir
from basicsr_dctp.utils.dist_util import get_dist_info
from basicsr_dctp.utils.registry import DATASET_REGISTRY

# from basicsr_dctp.data.transforms import augment
# from basicsr_dctp.utils import img2tensor
# from basicsr_dctp.utils.matlab_functions import imresize
# from torchvision.transforms.functional import normalize

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'basicsr_dctp.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must constain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} ' 'is built.')
    return dataset

# def custom_collate(batch):
#     # print(batch)

#     gts = [b['gt'] for b in batch]
#     gt_path = [b['gt_path'] for b in batch]
#     in_size = [d['in_size'].unsqueeze(0) for d in batch]
#     scales = [b['scale'] for b in batch]
#     r_margin = batch[0]['r_margin']
#     v_margin = batch[0]['v_margin']
#     is_normalize = [b['is_normalize'] for b in batch]
#     hflip = batch[0]['hflip']
#     mean = batch[0]['mean']
#     std = batch[0]['std']

#     lq = []
#     gt = []

#     h, w = 1024, 1024

#     for index, img_gt in enumerate(gts):
#         h, w, _ = img_gt.shape
#         img_gt = img_gt[v_margin:h-v_margin, r_margin:w-r_margin]
#         h, w, _ = img_gt.shape
#         img_gt, status = augment(img_gt, hflip=hflip, rotation=False, return_status=True)
#         scale = scales[index]
#         img_lq = imresize(img_gt, 1/scale)
#         img_lq = cv2.resize(img_lq,(1024,1024))
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
#         img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
#         if is_normalize[index]:
#             normalize(img_lq, mean, std, inplace=True)
#             normalize(img_gt, mean, std, inplace=True)
#         lq.append(img_lq.unsqueeze(0))
#         gt.append(img_gt.unsqueeze(0))



#     # lq = [d['lq'].unsqueeze(0) for d in batch]
#     # gt = [d['gt'].unsqueeze(0) for d in batch]
#     # in_size = [d['in_size'].unsqueeze(0) for d in batch]
#     r_scale = torch.from_numpy(np.array([h/1024], dtype=np.float32))
#     v_scale = torch.from_numpy(np.array([w/1024], dtype=np.float32))


#     # inputs = pad_sequence(inputs, batch_first=True)
#     lq = torch.cat(lq, 0)
#     # print('lq:', lq.shape)
#     gt = torch.cat(gt, 0)
#     # print('gt:', gt.shape)
#     in_size = torch.cat(in_size, 0)
#     # r_scale = torch.cat(r_scale, 0)
#     # v_scale = torch.cat(v_scale, 0)

#     return {
#         'lq': lq,
#         'gt': gt,
#         'gt_path': gt_path,
#         'in_size': in_size,
#         'r_scale': r_scale,
#         'v_scale': v_scale
#     }


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            # collate_fn=custom_collate,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: ' f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
