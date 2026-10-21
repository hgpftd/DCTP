import cv2
import re
import torch
import math
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np

from basicsr_dctp.data import degradations as degradations
from basicsr_dctp.data.data_util import paths_from_lmdb
from basicsr_dctp.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr_dctp.utils.registry import DATASET_REGISTRY
from basicsr_dctp.utils.matlab_functions import imresize


@DATASET_REGISTRY.register()
class SingleImage_GT_Double_Dataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImage_GT_Double_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        self.out_size = opt['out_size']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        h, w, _ = img_lq.shape

        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        imgname = lq_path.split('/')[-1]
        deg_list = re.findall(r"[0-9]+\.*[0-9]*", imgname)[1:]
        # print(imgname, "----------------------------")
        degradation = torch.from_numpy(np.array(deg_list, dtype=np.float32))


        # degradation = torch.from_numpy(np.array([sigma_label, scale_label, noise_range_label, jpeg_range_label], dtype=np.float32))
        # degradation = torch.from_numpy(np.array([self.is_gray, self.blur_kernel_size, sigma1, sigma2, rotation, in_size, r_scale, v_scale, sigma, quality], dtype=np.float32))
        # degradation = torch.from_numpy(np.array([in_size, r_scale, v_scale]))
        # degradation = torch.from_numpy(np.array([in_size, r_scale, v_scale], dtype=np.float32))
        return {'x1': img_lq, 'lq_path': lq_path, 'degradation': degradation}
        

        # in_size = img_lq.shape[1]
        # scale = self.out_size / in_size
        # img_lq = imresize(img_lq, scale)

        # h, w, _ = img_lq.shape

        # # img_lq = img_lq[self.v_margin:h-self.v_margin, self.r_margin:w-self.r_margin]

        # h, w, _ = img_lq.shape

        # img_lq = cv2.resize(img_lq,(1024,1024))
 

        # # TODO: color space transform
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        
        # img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # in_size = scale / self.cond_norm
        # cond = torch.from_numpy(np.array([in_size], dtype=np.float32)) 
        
        # # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)

        # # r_scale = torch.from_numpy(np.array([h/1024.], dtype=np.float32))
        # # v_scale = torch.from_numpy(np.array([w/1024.], dtype=np.float32))

        # r_scale = h/1024
        # v_scale = w/1024

        # # degradation = torch.from_numpy(np.array([in_size, r_scale, v_scale, 0, 0, 0, 100], dtype=np.float32))
        # # degradation = torch.from_numpy(np.array([0, 0, 0, 0, in_size, r_scale, v_scale, 0, 100], dtype=np.float32))
        # # degradation = torch.from_numpy(np.array([in_size, r_scale, v_scale], dtype=np.float32))
        # # return {'lq': img_lq, 'lq_path': lq_path, 'in_size': cond, 'r_scale': r_scale, 'v_scale': v_scale, 'degradation': degradation}
        # return {'lq': img_lq, 'lq_path': lq_path, 'in_size': cond, 'r_scale': r_scale, 'v_scale': v_scale}

    def __len__(self):
        return len(self.paths)
