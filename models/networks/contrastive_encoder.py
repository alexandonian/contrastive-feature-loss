"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.networks.architecture import VGG19
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.utils import FeatureHooks
from torch.nn import init


class ContrastiveEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder = self._get_encoder(opt)
        self.projection = define_H(opt)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.forward(
            torch.randn(
                1, 3, int(opt.crop_size // opt.aspect_ratio), int(opt.crop_size)
            ),
            nce_layers=self.nce_layers,
        )

    def forward(self, x, y=None, nce_layers=None, num_patches=256, patch_ids=None):
        feats = self.features(x, y=y, nce_layers=nce_layers)
        feats_pool, pids = self.projection(
            feats, num_patches=num_patches, patch_ids=patch_ids
        )
        return feats, feats_pool, pids

    def features(self, x, y=None, nce_layers=None):
        if self.opt.netF.lower() == 'vgg19':
            out = self.encoder(x)
            if self.nce_layers == [0, 1, 2, 3, 4, 5]:
                return out
            elif self.nce_layers == [1, 2, 3, 4, 5]:
                return out[1:]
            elif self.nce_layers == [2, 3, 4, 5]:
                return out[2:]
            elif self.nce_layers == [3, 4, 5]:
                return out[3:]
        elif self.opt.netF.lower() == 'pixel':
            return self.encoder(x)
        if nce_layers is not None and nce_layers != self.nce_layers:
            self.nce_layers = nce_layers
        _ = self.encoder(x, y=y)
        return self.fhooks.get_output(device=x.device)

    def _setup_hooks(self):
        self.hook_names, self.hooks, self.fhooks = self.encoder.setup_hooks(
            self.nce_layers
        )

    def _get_encoder(self, opt):
        print(f'setting encoder to {opt.netF.lower()}')
        if opt.netF.lower() == 'convencoder':
            return ConvEncoder(opt)
        elif opt.netF.lower() == 'vgg':
            return VGG(opt)
        elif 'resnet' in opt.netF.lower():
            return ResNet(opt)
        elif opt.netF.lower() == 'vgg19':
            pt = vars(opt).get('pretrained_netF', False)
            self.weights = np.array([1.0, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0])
            self.weights = self.weights / np.sum(self.weights)
            self.weights = self.weights[1:]
            return VGG19(
                opt=opt,
                pretrained=pt,
                freeze_weights=opt.freeze_netF,
                before_relu=vars(opt).get('vgg_before_relu', False),
            )
        elif opt.netF.lower() == 'pixel':
            return PixelEncoder(opt)
        else:
            raise ValueError('{} model not recognized'.format(opt.netF))

    @property
    def nce_layers(self):
        return self._nce_layers

    @nce_layers.setter
    def nce_layers(self, value):
        curr_val = getattr(self, '_nce_layers', None)
        if curr_val != value:
            self._nce_layers = value
            try:
                self._setup_hooks()
            except AttributeError as e:
                print(e)

    def compute_perceptual_loss(self, x, y):
        feats_x, feats_y = self.features(x), self.features(y)
        criterion = nn.L1Loss().cuda()
        weights = getattr(self, 'weights', None)
        if weights is None:
            weights = [1.0 / len(feats_x)] * len(feats_x)
        return sum(
            w * criterion(xf, yf.detach())
            for xf, yf, w in zip(feats_x, feats_y, weights)
        )


class ResNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = getattr(models, opt.netF)(pretrained=False)

    def forward(self, x, y=None):
        if (x.size(2) != 224 or x.size(3) != 224) and self.opt.netF_resize_input:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        return self.model(x)

    def setup_hooks(self, nce_layers):
        hook_names = []
        for nce in nce_layers:
            if nce == 0:
                hn = 'conv1'
            elif nce == 5:
                hn = 'avgpool'
            else:
                hn = 'layer{}'.format(nce)
            hook_names.append(hn)
        hooks = [{'name': name, 'type': 'forward_pre'} for name in hook_names]
        fhooks = FeatureHooks(hooks, self.model.named_modules())
        return hook_names, hooks, fhooks


class VGG(nn.Module):
    vgg_class = models.vgg19

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = self.vgg_class()

    def forward(self, x, y=None):
        if (x.size(2) != 224 or x.size(3) != 224) and self.opt.netF_resize_input:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        return self.model(x)

    def setup_hooks(self, nce_layers):
        hook_names = [
            'classifier.6' if i == -1 else 'features.{}'.format(i) for i in nce_layers
        ]
        hooks = [{'name': name, 'type': 'forward_pre'} for name in hook_names]
        fhooks = FeatureHooks(hooks, self.model.named_modules())
        return hook_names, hooks, fhooks


class ConvEncoder(BaseNetwork):
    """Same architecture as the image discriminator"""

    eff_receptive_fields = [3, 7, 15, 31, 63, 127]

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        input_nc = vars(opt).get('input_nc') or 3
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer0 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw))
        self.layer1 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer5 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw)
            )

        self.so = s0 = 4
        self.avgpool = nn.AdaptiveAvgPool2d((s0, s0))
        self.fc = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x, y=None):
        if self.opt.netF_resize_input:
            x = self.resize_input(x)
        if y is not None:
            if self.opt.netF_resize_input:
                y = self.resize_input(y)
            x = torch.cat((x, y), dim=1)

        x = self.layer0(x)
        x = self.layer1(self.actvn(x))
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer5(self.actvn(x))
        x = self.actvn(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def resize_input(self, x, size=256):
        if x.size(2) != size or x.size(3) != size:
            # x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
            x = F.interpolate(x, size=(size, size), mode='bilinear')
        return x

    def setup_hooks(self, nce_layers):
        hook_names = ['fc' if i == -1 else 'layer{}.0'.format(i) for i in nce_layers]
        hooks = [{'name': name, 'type': 'forward_pre'} for name in hook_names]
        fhooks = FeatureHooks(hooks, self.named_modules())
        return hook_names, hooks, fhooks


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type
                )
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find('BatchNorm2d') != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=None,
    debug=False,
    initialize_weights=True,
):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids is None:
        gpu_ids = []

    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_H(opt):
    opt_dict = vars(opt)
    return _define_H(opt=opt, **opt_dict)


def _define_H(
    netH,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=[],
    opt=None,
    **kwargs,
):
    if netH == 'mlp_sample':
        net = PatchSampleH(
            use_mlp=True,
            linear_mlp=False,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=opt.netH_nc,
            use_kl=opt.lambda_KL > 0.0,
            nce_norm=opt.nce_norm,
        )
    elif netH == 'linear_sample':
        net = PatchSampleH(
            use_mlp=True,
            linear_mlp=True,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=opt.netH_nc,
            use_kl=opt.lambda_KL > 0.0,
            nce_norm=opt.nce_norm,
        )
    elif netH == 'identity_sample':
        net = PatchSampleH(
            use_mlp=True,
            linear_mlp=False,
            ident_mlp=True,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=opt.netH_nc,
            use_kl=opt.lambda_KL > 0.0,
            nce_norm=opt.nce_norm,
        )

    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netH)
    return init_net(net, init_type, init_gain, gpu_ids)


class PatchSampleH(nn.Module):
    def __init__(
        self,
        use_mlp=False,
        linear_mlp=False,
        ident_mlp=False,
        init_type='normal',
        init_gain=0.02,
        nc=256,
        gpu_ids=None,
        nce_norm=2,
        use_kl=False,
    ):
        super().__init__()
        self.l2norm = Normalize(nce_norm)
        self.use_mlp = use_mlp
        self.linear_mlp = linear_mlp
        self.ident_mlp = ident_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.use_kl = use_kl
        self.mlps = nn.ModuleList()
        self.nce_norm = nce_norm
        self.gpu_ids = [] if gpu_ids is None else gpu_ids

    def create_mlp(self, feats):
        print('create mlp during runtime at device %s' % str(feats[0].device))
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            output_nc = 2 * self.nc if self.use_kl else self.nc
            if self.linear_mlp:
                mlp = nn.Linear(input_nc, output_nc)
            elif self.ident_mlp:
                mlp = nn.Identity()
            else:
                mlp = nn.Sequential(
                    *[
                        nn.Linear(input_nc, self.nc),
                        nn.ReLU(),
                        nn.Linear(self.nc, output_nc),
                    ]
                )
            self.mlps.append(mlp)
        self.mlps.to(feats[0].device)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            device = feat.device

            # [bs, H*W, num_channels]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=device)
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]

                if patch_id.ndim == 2 and patch_id.size(0) == B:
                    x_sample = torch.stack(
                        [f[p] for f, p in zip(feat_reshape, patch_id)]
                    ).flatten(0, 1)
                else:
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = self.mlps[feat_id]
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)

            if not self.use_kl:
                x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample_shape = [B, x_sample.shape[-1], H, W]
                x_sample = x_sample.permute(0, 2, 1).reshape(x_sample_shape)
            x_sample = x_sample.view(B, -1, x_sample.size(-1))
            return_feats.append(x_sample)

        return return_feats, return_ids


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        if self.power not in [1, 2]:
            self.forward = self.identity

    def forward(self, x):
        norm = x.norm(p=self.power, dim=1, keepdim=True)
        return x.div(norm + 1e-7)

    def identity(self, x):
        return x


class PixelEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def forward(self, x, y=None):
        out = []
        for size, stride in [(4, 2), (8, 4), (16, 8)]:
            o = self.to_patches(x, size=size, stride=stride)
            out.append(o)
        return out

    def to_patches(self, image, size=64, stride=8):
        b, s = image.shape[:2]
        patches = (
            image.unfold(1, s, stride).unfold(2, size, stride).unfold(3, size, stride)
        )  # [bs, 1, num_height, num_width, dim, size, size]
        h, w = patches.shape[2:4]
        patches = patches.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return patches
