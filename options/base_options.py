"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os
import pickle
import sys

import torch

import data
import evaluation
import models
from util import util


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument(
            '--name',
            type=str,
            default=None,
            help='name of the experiment. It decides where to store samples and models',
        )
        parser.add_argument(
            '--exp_id',
            type=str,
            default=None,
            help='A string prefix to allow for more name specificity with auto name generation',
        )
        parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
        )
        parser.add_argument(
            '--checkpoints_dir',
            type=str,
            default='./checkpoints',
            help='models are saved here',
        )
        parser.add_argument(
            '--model',
            type=str,
            default='contrastive_pix2pix',
            help='which model to use',
            choices=['pix2pix', 'contrastive_pix2pix'],
        )
        parser.add_argument(
            '--norm_G',
            type=str,
            default='spectralinstance',
            help='instance normalization or batch normalization',
        )
        parser.add_argument(
            '--norm_D',
            type=str,
            default='spectralinstance',
            help='instance normalization or batch normalization',
        )
        parser.add_argument(
            '--norm_E',
            type=str,
            default='spectralinstance',
            help='instance normalization or batch normalization',
        )
        parser.add_argument(
            '--phase',
            type=str,
            default='train',
            help='train, val, test, etc',
        )

        # input/output sizes
        parser.add_argument(
            '--batch_size', type=int, default=1, help='input batch size'
        )
        parser.add_argument(
            '--preprocess_mode',
            type=str,
            default='scale_width_and_crop',
            help='scaling and cropping of images at load time.',
            choices=(
                "resize_and_crop",
                "crop",
                "scale_width",
                "scale_width_and_crop",
                "scale_shortside",
                "scale_shortside_and_crop",
                "fixed",
                "none",
            ),
        )
        parser.add_argument(
            '--load_size',
            type=int,
            default=1024,
            help='Scale images to this size. The final image will be cropped to --crop_size.',
        )
        parser.add_argument(
            '--crop_size',
            type=int,
            default=512,
            help='Crop to the width of crop_size (after initially scaling the images to load_size.)',
        )
        parser.add_argument(
            '--aspect_ratio',
            type=float,
            default=1.0,
            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio',
        )
        parser.add_argument(
            '--label_nc',
            type=int,
            default=182,
            help='# of input label classes without unknown class. If you have unknown class as class label, '
            'specify --contain_dopntcare_label.',
        )
        parser.add_argument(
            '--contain_dontcare_label',
            action='store_true',
            help='if the label map contains dontcare label (dontcare=255)',
        )
        parser.add_argument(
            '--output_nc', type=int, default=3, help='# of output image channels'
        )

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--dataset_mode', type=str, default='coco')
        parser.add_argument(
            '--serial_batches',
            action='store_true',
            help='if true, takes images in order to make batches, otherwise takes them randomly',
        )
        parser.add_argument(
            '--no_flip',
            action='store_true',
            help='if specified, do not flip the images for data argumentation',
        )
        parser.add_argument(
            '--no_flip_vert',
            type=util.str2bool,
            default=True,
            help='if specified, do not randomly flip the images vertically for augmentation',
        )
        parser.add_argument(
            '--no_rotate',
            type=util.str2bool,
            default=True,
            help='If True, do not randomly rotate image for augmentation',
        )
        parser.add_argument(
            '--nThreads', default=4, type=int, help='# threads for loading data'
        )
        parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=sys.maxsize,
            help='Maximum number of samples allowed per dataset. '
            'If the dataset directory contains more than max_dataset_size, only a subset is loaded.',
        )
        parser.add_argument(
            '--load_from_opt_file',
            action='store_true',
            help='load the options from checkpoints and use that as default',
        )
        parser.add_argument(
            '--cache_filelist_write',
            action='store_true',
            help='saves the current filelist into a text file, so that it loads faster',
        )
        parser.add_argument(
            '--cache_filelist_read',
            action='store_true',
            help='reads from the file list cache',
        )

        # for displays
        parser.add_argument(
            '--display_winsize', type=int, default=400, help='display window size'
        )

        # for generator
        parser.add_argument(
            '--netG',
            type=str,
            default='spade',
            help='selects model to use for netG (pix2pixhd | spade)',
        )
        parser.add_argument(
            '--ngf', type=int, default=64, help='# of gen filters in first conv layer'
        )
        parser.add_argument(
            '--init_type',
            type=str,
            default='xavier',
            help='network initialization [normal|xavier|kaiming|orthogonal]',
        )
        parser.add_argument(
            '--init_variance',
            type=float,
            default=0.02,
            help='variance of the initialization distribution',
        )
        parser.add_argument(
            '--z_dim', type=int, default=256, help="dimension of the latent z vector"
        )

        # for instance-wise features
        parser.add_argument(
            '--no_instance',
            action='store_true',
            help='if specified, do *not* add instance map as input',
        )
        parser.add_argument(
            '--no_input_semantics',
            action='store_true',
            help='if specified, do *not* convert original label to input semantic map',
        )
        parser.add_argument(
            '--nef',
            type=int,
            default=16,
            help='# of encoder filters in the first conv layer',
        )
        parser.add_argument(
            '--use_vae',
            action='store_true',
            help='enable training with an image encoder.',
        )
        parser.add_argument(
            '--use_eval_mode',
            action='store_true',
            default=True,
            help='set model to .eval() mode during evaluation.',
        )
        parser.add_argument(
            '--evaluation_metrics',
            type=str,
            default='none',
            choices=['fid'],
            help='evaluation metrics to use.',
        )
        parser.add_argument('--inception_weights', type=str, default='fid_inception')
        parser.add_argument(
            '--perceptual_encoder',
            type=str,
            default='vgg',
            help='baseline vgg method encoder',
        )
        parser.add_argument(
            '--vgg_normalize_input',
            action='store_true',
            help='do imagenet mean/std normalization to input',
        )
        parser.add_argument(
            '--vgg_resize_input',
            action='store_true',
            help='resize input to 224 pixels if necessary',
        )
        parser.add_argument(
            '--pretrained_name',
            type=str,
            default=None,
            help='Name of pretrained model experiment to initialize networks',
        )
        parser.add_argument(
            '--pretrained_epoch',
            type=str,
            default='latest',
            help='Name of pretrained model epoch to initialize networks',
        )
        parser.add_argument(
            '--pretrained_nets',
            type=str,
            default='G,D,F_q,F_k',
            help='A comma separated list of the specific pretrained nets to load (default: %(default)s)',
        )
        parser.add_argument(
            '--data_augmentation',
            type=str,
            default=None,
            help='Indicate the type of data augmentation applied to images.',
            choices=['basic_color', 'rand_color', 'rand_color_inc'],
        )
        parser.add_argument(
            '--use_wandb',
            type=util.str2bool,
            default=True,
            help='whether to use wandb logging',
        )
        parser.add_argument(
            '--G_eval_mode',
            type=util.str2bool,
            default=False,
            help='whether to wrap model with wandb.watch',
        )
        parser.add_argument(
            '--wandb_job_type',
            type=str,
            default='training',
        )
        parser.add_argument(
            '--netF',
            type=str,
            default='vgg19',
            help='downsample the feature map: ConvEncoder | VGG | ResNet',
        )
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        # modify evaluation-related parser options
        evaluation_option_setter = evaluation.get_option_setter()
        parser = evaluation_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            try:
                parser = self.update_options_from_file(parser, opt)
            except FileNotFoundError:
                print('WARNING: Options for {} not found!.'.format(opt.name))
                print('Options not loaded from file!')

        # opt = parser.parse_args()
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = '\t[default: %s]' % str(default) if v != default else ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        return os.path.join(expr_dir, 'opt')

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                default = self.parser.get_default(k)
                comment = '\t[default: %s]' % str(default) if v != default else ''
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        return pickle.load(open(file_name + '.pkl', 'rb'))

    def update_options(self, opt):
        try:
            opt.gan_loss = not opt.no_gan_loss
            opt.ganFeat_loss = not opt.no_ganFeat_loss
            opt.vgg_loss = not opt.no_vgg_loss
            opt.L1_loss = not opt.no_L1_loss
            opt.contrastive_loss = not opt.no_contrastive_loss
        except AttributeError:
            pass
        return opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        opt = self.update_options(opt)
        opt.name = f'{opt.dataset_mode}_{opt.model}' if opt.name is None else opt.name

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = (
            opt.label_nc
            + (1 if opt.contain_dontcare_label else 0)
            + (0 if opt.no_instance else 1)
        )

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert (
            not opt.gpu_ids or opt.batch_size % len(opt.gpu_ids) == 0
        ), "Batch size %d is wrong. It must be a multiple of # GPUs %d." % (
            opt.batch_size,
            len(opt.gpu_ids),
        )

        self.opt = opt
        return self.opt
