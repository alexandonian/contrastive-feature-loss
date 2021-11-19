"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import util.util as util
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument(
            '--display_freq',
            type=int,
            default=5000,
            help='frequency of showing training results on screen',
        )
        parser.add_argument(
            '--print_freq',
            type=int,
            default=100,
            help='frequency of showing training results on console',
        )
        parser.add_argument(
            '--save_latest_freq',
            type=int,
            default=5000,
            help='frequency of saving the latest results',
        )
        parser.add_argument(
            '--save_epoch_freq',
            type=int,
            default=10,
            help='frequency of saving checkpoints at the end of epochs',
        )
        parser.add_argument(
            '--no_html',
            action='store_true',
            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/',
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='only do one epoch and displays at each iteration',
        )
        parser.add_argument(
            '--no_tf_log',
            default=True,
            type=util.str2bool,
            help='if specified, do not use tensorboard logging. Requires tensorflow installed',
        )
        parser.add_argument(
            '--no_json_log',
            action='store_true',
            help='if specified, do not use json logging',
        )
        # for training
        parser.add_argument(
            '--continue_train',
            default=False,
            type=util.str2bool,
            help='continue training: load the latest model',
        )
        parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model',
        )
        parser.add_argument(
            '--niter',
            type=int,
            default=800,
            help='# of iter at starting learning rate. This is NOT the total #epochs. '
            'Total #epochs is niter + niter_decay',
        )
        parser.add_argument(
            '--niter_decay',
            type=int,
            default=0,
            help='# of iter to linearly decay learning rate to zero',
        )
        parser.add_argument(
            '--evaluation_freq', type=int, default=5000, help='evaluation freq'
        )
        parser.add_argument(
            '--optimizer', type=str, default='adam', help='name of optimizer to use'
        )
        parser.add_argument(
            '--beta1', type=float, default=0.0, help='momentum term of adam'
        )
        parser.add_argument(
            '--beta2', type=float, default=0.9, help='momentum term of adam'
        )
        parser.add_argument(
            '--no_TTUR',
            type=util.str2bool,
            default=True,
            help='Use TTUR training scheme',
        )

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument(
            '--lr', type=float, default=0.0002, help='initial learning rate for adam'
        )
        parser.add_argument(
            '--D_steps_per_G',
            type=int,
            default=1,
            help='number of discriminator iterations per generator iterations.',
        )
        parser.add_argument(
            '--num_G_steps',
            type=int,
            default=1,
            help='number of generator iterations.',
        )
        parser.add_argument(
            '--num_F_steps_per_G',
            type=int,
            default=1,
            help='number of F iterations per G.',
        )

        # for discriminators
        parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help='# of discrim filters in first conv layer',
        )
        parser.add_argument(
            '--lambda_ganFeat',
            type=float,
            default=10.0,
            help='weight for feature matching loss',
        )
        parser.add_argument(
            '--lambda_vgg',
            type=float,
            default=10.0,
            help='weight for vgg loss',
        )
        parser.add_argument(
            '--lambda_L1',
            type=float,
            default=10.0,
            help='weight for L1 pixel loss',
        )

        parser.add_argument(
            '--lambda_gan',
            type=float,
            default=1.0,
            help='weight for gan loss',
        )
        parser.add_argument(
            '--no_ganFeat_loss',
            default=True,
            type=util.str2bool,
            help='if specified, do *not* use discriminator feature matching loss',
        )
        parser.add_argument(
            '--no_vgg_loss',
            default=True,
            type=util.str2bool,
            help='if specified, do *not* use VGG feature matching loss',
        )
        parser.add_argument(
            '--gan_loss',
            default=True,
            type=util.str2bool,
            help='whether to include gan loss',
        )
        parser.add_argument(
            '--L1_loss',
            default=True,
            type=util.str2bool,
            help='whether to include L1 loss',
        )
        parser.add_argument(
            '--vgg_loss',
            default=True,
            type=util.str2bool,
            help='whether to include vgg loss',
        )
        parser.add_argument(
            '--ganFeat_loss',
            default=True,
            type=util.str2bool,
            help='whether to include ganFeat loss',
        )
        parser.add_argument(
            '--no_gan_loss',
            default=False,
            type=util.str2bool,
            help='if specified, do *not* use GAN loss.',
        )
        parser.add_argument(
            '--no_L1_loss',
            default=True,
            type=util.str2bool,
            help='if specified, do *not* use Generator L1 pixel loss.',
        )
        parser.add_argument(
            '--use_crn_lambdas',
            action='store_true',
            help='if specified, use CRN VGG layer lambdas.',
        )
        parser.add_argument(
            '--vgg_loss_include_input',
            action='store_true',
            help='if specified, vgg loss will include input in loss.',
        )
        parser.add_argument(
            '--unfrozen_vgg',
            type=util.str2bool,
            default=False,
            help='if true, unfreeze the weights of a pretrained VGG for perceptual loss',
        )
        parser.add_argument(
            '--pretrained_vgg',
            type=util.str2bool,
            default=True,
            help='whether to use pretrained vgg for vgg loss',
        )
        parser.add_argument(
            '--pretrained_netF',
            type=util.str2bool,
            default=True,
            help='whether to use pretrained model for contrastive encoder',
        )
        parser.add_argument(
            '--gan_mode',
            type=str,
            default='hinge',
            help='(ls|original|hinge)',
        )
        parser.add_argument(
            '--netD',
            type=str,
            default='multiscale',
            help='(n_layers|multiscale|image)',
        )
        parser.add_argument('--lambda_kld', type=float, default=0.05)

        self.isTrain = True
        return parser
