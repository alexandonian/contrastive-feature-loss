"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from . import rand_augment


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    flip_vert = random.random() > 0.5
    rotate = random.choice([90, 180, 270])
    return {'crop_pos': (x, y), 'flip': flip, 'flip_vert': flip_vert, 'rotate': rotate}


def get_transform(
    opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, is_image=False
):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(
            transforms.Lambda(lambda img: _scale_width(img, opt.load_size, method))
        )
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(
            transforms.Lambda(lambda img: _scale_shortside(img, opt.load_size, method))
        )

    if 'crop' in opt.preprocess_mode:
        transform_list.append(
            transforms.Lambda(lambda img: _crop(img, params['crop_pos'], opt.crop_size))
        )

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(
            transforms.Lambda(lambda img: _make_power_2(img, base, method))
        )

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: _resize(img, w, h, method)))

    if opt.data_augmentation and is_image:
        transform_list.append(get_augmentation(opt))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: _flip(img, params['flip'])))

    if opt.isTrain and not opt.no_flip_vert:
        transform_list.append(
            transforms.Lambda(lambda img: _flip_vertical(img, params['flip_vert']))
        )

    if opt.isTrain and not opt.no_rotate:
        transform_list.append(
            transforms.Lambda(lambda img: _rotate(img, params['rotate']))
        )

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def _make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def _scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def _scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if ss == target_width:
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def _crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _flip_vertical(img, flip_vert):
    if flip_vert:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def _rotate(img, angle):
    if angle != 0:
        func = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}.get(
            angle
        )
        return img.transpose(func)
    return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentation(opt, n=2, m=4, blacklist=['Invert']):
    augmentation = []
    if opt.data_augmentation in ['basic_color']:
        augmentation = [
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8,  # not strengthened
            ),
        ]
    elif opt.data_augmentation.startswith('rand_color'):
        augmentation.append(
            rand_augment.rand_augment_transform(
                opt.data_augmentation, num_layers=n, magnitude=m, blacklist=blacklist
            )
        )

    elif opt.data_augmentation.startswith('rand_full'):
        augmentation.append(
            rand_augment.rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 4))
        )
    return transforms.Compose(augmentation)
