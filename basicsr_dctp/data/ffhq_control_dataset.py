import cv2
import math
import random
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr_dctp.data import degradations as degradations
from basicsr_dctp.data.data_util import paths_from_folder
from basicsr_dctp.data.transforms import augment
from basicsr_dctp.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr_dctp.utils.registry import DATASET_REGISTRY
from basicsr_dctp.utils.matlab_functions import imresize

import basicsr_dctp.data.degradations as degradations

@DATASET_REGISTRY.register()
class FFHQ_control_Dataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQ_control_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.out_size = opt['out_size']

        # self.crop_components = opt.get('crop_components', False)  # facial components
        # self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        # if self.crop_components:
        #     # load component list from a pre-process pth files
        #     self.components_list = torch.load(opt.get('component_path'))

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        # degradations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')


        # self.blur_kernel_size = 41
        # self.blur_sigma = [0.1, 10]
        # self.noise_range = [0, 20]
        # self.jpeg_range = [60, 100]
        # # self.gray_prob = 0.01
        # self.downsample_range = [0.8, 8]

        # # attribute
        # self.batch_size = opt['batch_size_per_gpu']
        # self.r_margin = 0
        # self.v_margin = 0
        # logger = get_root_logger()
        # self.cur_loop = 0
        # self.current_iter = 0

        # self.max_change = 30000*self.batch_size

        # self.is_gray = 0
        print("Use FFHQ_control_Dataset")


    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def degenerate(self, img_gt):
        h, w, _ = img_gt.shape
        # ------------------------ generate lq image ------------------------ #
        kernel, sigma_x, sigma_y = degradations.random_mixed_kernels_2_mode(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        sigma_x_label = sigma_x*10
        sigma_y_label = sigma_y*10
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        scale_label = 100 - 100/scale
        # noise
        if self.noise_range is not None:
            img_lq, noise_label = degradations.random_add_gaussian_noise(img_lq, self.noise_range, return_sigma=True)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq, quality_label = degradations.random_add_jpg_compression(img_lq, self.jpeg_range, return_q=True)
            quality_label = 100 - quality_label
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        deg_list = [sigma_x_label, sigma_y_label, scale_label, noise_label, quality_label]
        return img_lq, deg_list

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape


        img_gt = cv2.resize(img_gt, (self.out_size, self.out_size))

        # if self.crop_components:
        #     locations = self.get_component_coordinates(index, status)
        #     loc_left_eye, loc_right_eye, loc_mouth = locations

        # degenerate img_gt #
        img_lq, deg_list = self.degenerate(img_gt)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        alpha = torch.rand(1)

        degradation = torch.from_numpy(np.array(deg_list, dtype=np.float32))
        
        # if self.crop_components:
        #     return_dict = {
        #         'lq': img_lq,
        #         'gt': img_gt,
        #         'gt_path': gt_path,
        #         'degradation': degradation,
        #         'alpha': alpha,
        #         'loc_left_eye': loc_left_eye,
        #         'loc_right_eye': loc_right_eye,
        #         'loc_mouth': loc_mouth
        #     }
        #     return return_dict
        # else:
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'degradation': degradation, 'alpha': alpha}

    def __len__(self):
        return len(self.paths)
