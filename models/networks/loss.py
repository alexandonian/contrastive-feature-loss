"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        gan_mode,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
        opt=None,
    ):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode not in ['ls', 'original', 'w', 'hinge']:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.binary_cross_entropy_with_logits(input, target_tensor)
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                return -torch.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                return -torch.mean(input)
        elif target_is_real:
            return -input.mean()
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if not isinstance(input, list):
            return self.loss(input, target_is_real, for_discriminator)
        loss = 0
        for pred_i in input:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
            bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
            loss += new_loss
        return loss / len(input)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    DEFAULT_WEIGHTS = np.array([1.0, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0])
    CRN_WEIGHTS = np.array([1.0, 1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5])

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        fw = not vars(opt).get('unfrozen_vgg', False)
        pt = vars(opt).get('pretrained_vgg', True)
        self.vgg = VGG19(opt, pretrained=pt, freeze_weights=fw).cuda()
        self.criterion = nn.L1Loss()
        if self.opt.use_crn_lambdas:
            self.weights = self.CRN_WEIGHTS
        else:
            self.weights = self.DEFAULT_WEIGHTS
        self.weights = self.weights / np.sum(self.weights)
        self.start_idx = 0 if self.opt.vgg_loss_include_input else 1

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg.insert(0, x)
        y_vgg.insert(0, y)
        return sum(
            self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            for i in range(self.start_idx, len(x_vgg))
        )


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = self._get_model(opt)

    def forward(self, *x):
        return self.model(*x)

    def _get_model(self, opt):
        if opt.perceptual_encoder == 'vgg':
            return VGGLoss(opt)
        else:
            raise ValueError(
                f'Unrecognized perceptual encoder: {opt.perceptual_encoder}'
            )
