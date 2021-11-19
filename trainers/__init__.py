"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .pix2pix_trainer import Pix2PixTrainer
from .contrastive_pix2pix_trainer import ContrastivePix2PixTrainer


def get_trainer(opt):
    # if opt.model == 'contrastive_pix2pix':
    if 'contrastive_pix2pix' in opt.model:
        return ContrastivePix2PixTrainer(opt)
    elif opt.model == 'pix2pix':
        return Pix2PixTrainer(opt)
    else:
        raise ValueError('Unrecognized model/trainer')
