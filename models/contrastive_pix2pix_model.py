"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import contextlib

import numpy as np
import torch
import util.util as util
from torchvision import models

import models.networks as networks


class ContrastivePix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument(
            '--netH',
            type=str,
            default='mlp_sample',
            help='downsample the feature map:  mlp_sample | linear_sample',
        )
        parser.add_argument(
            '--freeze_netF',
            type=util.str2bool,
            default=True,
            help='if true, freeze the weights of the encoder network F',
        )
        parser.add_argument(
            '--freeze_netH',
            type=util.str2bool,
            default=False,
            help='if true, freeze the weights of the encoder projection H',
        )
        parser.add_argument(
            '--freeze_netG',
            type=util.str2bool,
            default=False,
            help='if true, freeze the weights of the generator network G',
        )
        parser.add_argument('--netH_nc', type=int, default=256)
        parser.add_argument(
            '--nce_t', type=float, default=0.07, help='temperature for NCE loss'
        )
        parser.add_argument(
            '--num_patches', type=int, default=256, help='number of patches per layer'
        )
        parser.add_argument(
            '--no_contrastive_loss',
            default=False,
            type=util.str2bool,
            help='If specified, do *not* compute contrastive loss',
        )
        parser.add_argument(
            '--input_nc', type=int, default=3, help='number of input image channels'
        )
        parser.add_argument(
            '--lambda_KL',
            type=float,
            default=0.0,
            help='weight for VAE loss: KL(E(x), N)',
        )
        parser.add_argument(
            '--lambda_NCE',
            type=float,
            default=10.0,
            help='weight for NCE loss',
        )
        parser.add_argument(
            '--nce_use_L1',
            default=False,
            type=util.str2bool,
            help='uses L1 distance instead of Cosine distance between query and key',
        )
        parser.add_argument(
            '--lambda_NCE_perceptual',
            type=float,
            default=10.0,
            help='weight for NCE loss',
        )
        parser.add_argument(
            '--gradient_flows_to_negative_nce',
            type=util.str2bool,
            default=False,
        )
        parser.add_argument(
            '--nce_layers',
            type=str,
            default='1,2,3,4,5',
            help='compute NCE loss on which layers',
        )
        parser.add_argument(
            '--layer_weights',
            type=str,
            default=None,
            help='weights to be applied to nce_layers',
        )
        parser.add_argument(
            '--nce_mode',
            type=str,
            default='bidirectional',
            choices=['internal', 'bidirectional'],
        )
        parser.add_argument('--nce_layer_weight_multiplier', type=float, default=1.0)
        parser.add_argument('--nce_norm', type=int, default=2)
        parser.add_argument('--nce_fake_negatives', type=util.str2bool, default=True)
        parser.add_argument('--netF_resize_input', type=util.str2bool, default=True)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = (
            torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        )
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.layer_weights = None
        if self.opt.layer_weights is not None:
            self.layer_weights = np.array(
                [float(i) for i in self.opt.layer_weights.split(',')],
            )
            self.layer_weights /= self.layer_weights.sum()

        # set loss functions
        if opt.isTrain:

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
            )

            self.criterionFeat = torch.nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.PerceptualLoss(self.opt)

            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(networks.PatchNCELoss(opt))

        (
            self.netG,
            self.netD,
            self.netE,
            self.netF_q,
            self.netF_k,
        ) = self.initialize_networks(opt)
        if self.opt.pretrained_netF:
            if self.opt.netF.lower() == 'vgg19':
                self.netF_q.encoder.load_state_dict(
                    networks.VGG19(opt, True, opt.freeze_netF).state_dict()
                )
            elif 'resnet' in self.opt.netF.lower():
                # TODO: Ensure this works!
                self.netF_q.encoder.model.load_state_dict(
                    getattr(models, self.opt.netF)().state_dict()
                )

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            return self.compute_discriminator_loss(input_semantics, real_image)
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        elif mode == 'features':
            netF_feats, feat_pool, pids = self.netF_q(real_image)
            return {
                'vgg': self.criterionVGG.model.vgg(real_image),
                'netF_feats': netF_feats,
                'netF_pool': feat_pool,
            }
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = []
        F_params = []

        # Add generator params, if needed.
        if not opt.freeze_netG:
            G_params = list(self.netG.parameters())
        else:
            util.freeze_parameters(self.netG)
            self.netG.eval()

        # Add contrastive encoder params.
        if opt.lambda_NCE > 0:
            freeze_netH = vars(opt).get('freeze_netH', False)
            if not opt.freeze_netF:
                F_params += list(self.netF_q.encoder.parameters())
            if not freeze_netH:
                F_params += list(self.netF_q.projection.parameters())

        D_params = list(self.netD.parameters()) if self.needs_netD else []

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = (opt.lr, opt.lr) if opt.no_TTUR else (opt.lr / 2, opt.lr * 2)

        if opt.optimizer.lower() != 'adam':
            raise ValueError('Optimizer {} not recognized'.format(opt.optimizer))
        optimizer_G = (
            torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
            if G_params
            else None
        )
        optimizer_F = (
            torch.optim.Adam(F_params, lr=G_lr, betas=(beta1, beta2))
            if F_params
            else None
        )
        optimizer_D = (
            torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
            if D_params
            else None
        )
        return optimizer_G, optimizer_D, optimizer_F

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netF_q, 'F_q', epoch, self.opt)
        if self.needs_netD:
            util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)
        if self.opt.nce_m > 0.0 or self.opt.nce_no_share:
            util.save_network(self.netF_k, 'F_k', epoch, self.opt)
        if vars(self.opt).get('unfrozen_vgg'):
            util.save_network(self.criterionVGG, 'VGG', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netF_q = networks.define_F(opt)
        netD = networks.define_D(opt) if self.needs_netD else None
        netE = networks.define_E(opt) if opt.use_vae else None
        netF_k = None

        resume_func = (
            self.load_pretrained_networks
            if self.opt.pretrained_name is not None
            else self.resume_networks
        )
        try:
            netG, netD, netE, netF_q, netF_k = resume_func(
                opt, netG, netF_q, netF_k, netD, netE
            )
        except FileNotFoundError as e:
            print(e)
            print('WARNING: Checkpoints for {} not found!.'.format(self.opt.name))
            print('Networks initialized from scratch')
        return netG, netD, netE, netF_q, netF_k

    def resume_networks(self, opt, netG, netF_q, netF_k, netD, netE):
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            netF_q = util.load_network(
                netF_q,
                'F_q',
                opt.which_epoch,
                opt,
                use_strict=True,
            )
            if opt.isTrain and self.needs_netD:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

            if vars(opt).get('unfrozen_vgg'):
                self.criterionVGG = util.load_network(
                    self.criterionVGG, 'VGG', opt.which_epoch, opt
                )
        return netG, netD, netE, netF_q, netF_k

    def load_pretrained_networks(
        self,
        opt,
        netG,
        netF_q,
        netF_k,
        netD,
        netE,
    ):
        pretrained_nets = opt.pretrained_nets.split(',')
        epoch = opt.pretrained_epoch
        if not opt.isTrain or opt.continue_train:
            if 'G' in pretrained_nets:
                netG = util.load_pretrained_network(netG, 'G', epoch, opt)
                print('loaded pretrained G: {}'.format(opt.pretrained_name))

            with contextlib.suppress(FileNotFoundError):
                if 'F_q' in pretrained_nets:
                    netF_q = util.load_pretrained_network(netF_q, 'F_q', epoch, opt)
                    print('loaded pretrained F_q: {}'.format(opt.pretrained_name))

                if opt.isTrain and 'D' in pretrained_nets:
                    netD = util.load_pretrained_network(netD, 'D', epoch, opt)
                    print('loaded pretrained D: {}'.format(opt.pretrained_name))

                if opt.use_vae and 'E' in pretrained_nets:
                    netE = util.load_pretrained_network(netE, 'E', epoch, opt)
                    print('loaded pretrained E: {}'.format(opt.pretrained_name))

                if vars(opt).get('unfrozen_vgg'):
                    self.criterionVGG = util.load_pretrained_network(
                        self.criterionVGG, 'VGG', epoch, opt
                    )
                    print('loaded pretrained VGG: {}'.format(opt.pretrained_name))

        return netG, netD, netE, netF_q, netF_k

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        if not self.opt.no_input_semantics:
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        if self.opt.no_input_semantics:
            input = data['label']
        else:
            # create one-hot label map
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = (
                self.opt.label_nc + 1
                if self.opt.contain_dontcare_label
                else self.opt.label_nc
            )
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input = torch.cat((input, instance_edge_map), dim=1)

        return input, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics,
            real_image,
            compute_kld_loss=self.opt.use_vae,
        )

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = None, None
        if (not self.opt.no_gan_loss) or (not self.opt.no_ganFeat_loss):
            pred_fake, pred_real = self.discriminate(
                input_semantics,
                fake_image,
                real_image,
            )

        if not self.opt.no_L1_loss:
            G_losses['G_L1'] = (
                self.criterionFeat(fake_image, real_image) * self.opt.lambda_L1
            )

        if not self.opt.no_gan_loss:
            G_losses['GAN'] = (
                self.criterionGAN(pred_fake, True, for_discriminator=False)
                * self.opt.lambda_gan
            )

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    )
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_ganFeat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = (
                self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            )

        if not self.opt.no_contrastive_loss:
            nce_loss = self.compute_NCE_loss(
                input_semantics,
                fake_image,
                real_image,
                self.criterionNCE,
                lambda_NCE=self.opt.lambda_NCE,
                layer_weights=self.layer_weights,
            )
            if not isinstance(nce_loss, dict):
                nce_loss = {'NCE': nce_loss}
            G_losses.update(**nce_loss)

        return G_losses, fake_image

    def get_nce_feats(self, input_semantics, fake_image, real_image):
        netF_k = self.netF_q
        feat_k, feat_k_pool, sample_ids = netF_k(
            real_image,
            num_patches=self.opt.num_patches,
            patch_ids=None,
        )

        featq, feat_q_pool, _ = self.netF_q(
            fake_image,
            num_patches=self.opt.num_patches,
            patch_ids=sample_ids,
        )
        return feat_q_pool, feat_k_pool

    @property
    def compute_total_nce_loss(self):

        if self.opt.nce_mode in ['bidirectional']:
            return self._compute_total_nce_loss_bidirectional
        else:
            return self._compute_total_nce_loss

    def compute_NCE_loss(
        self,
        input_semantics,
        fake_image,
        real_image,
        criterionNCE,
        lambda_NCE=1.0,
        layer_weights=None,
    ):

        feat_q_pool, feat_k_pool = self.get_nce_feats(
            input_semantics, fake_image, real_image
        )

        return self.compute_total_nce_loss(
            feat_q_pool,
            feat_k_pool,
            criterionNCE,
            lambda_NCE=lambda_NCE,
            layer_weights=layer_weights,
        )

    def _compute_total_nce_loss(
        self,
        feat_q_pool,
        feat_k_pool,
        criterionNCE,
        lambda_NCE=1.0,
        layer_weights=None,
    ):
        total_nce_loss = 0.0
        num_layers = len(self.nce_layers)

        if layer_weights is None:
            layer_weights = np.array(
                [
                    (self.opt.nce_layer_weight_multiplier) ** i
                    for i in range(len(self.nce_layers))
                ]
            )
            layer_weights = layer_weights / layer_weights.sum()
        for f_q, f_k, crit, nce_layer, layer_weight in zip(
            feat_q_pool, feat_k_pool, criterionNCE, self.nce_layers, layer_weights
        ):
            loss = crit(f_q, f_k) * lambda_NCE * layer_weight
            total_nce_loss += loss.mean()
        return total_nce_loss / num_layers

    def _compute_total_nce_loss_bidirectional(
        self,
        feat_q_pool,
        feat_k_pool,
        criterionNCE,
        lambda_NCE=1.0,
        layer_weights=None,
    ):
        total_nce_loss = 0.0
        num_layers = len(self.nce_layers)

        if layer_weights is None:
            layer_weights = np.array(
                [
                    (self.opt.nce_layer_weight_multiplier) ** i
                    for i in range(len(self.nce_layers))
                ]
            )
            layer_weights = layer_weights / layer_weights.sum()
        for f_q, f_k, crit, nce_layer, layer_weight in zip(
            feat_q_pool, feat_k_pool, criterionNCE, self.nce_layers, layer_weights
        ):
            loss = crit(f_q, f_k) * lambda_NCE * layer_weight
            loss += crit(f_k, f_q) * lambda_NCE * layer_weight
            total_nce_loss += loss.mean()
        return total_nce_loss / num_layers

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        if self.opt.no_gan_loss:
            return D_losses

        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image
        )

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.G_eval_mode:
            self.netG.eval()
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (
            not compute_kld_loss
        ) or self.opt.use_vae, "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = (
            edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        )
        edge[:, :, :, :-1] = (
            edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        )
        edge[:, :, 1:, :] = (
            edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        )
        edge[:, :, :-1, :] = (
            edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        )
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            input_semantics, real_image = self.preprocess_input(data)
            self.curr_label = input_semantics
            self.curr_real_image = real_image

            netG = self.netG

            if mode == "forward":
                visuals["fake_B"] = netG(input_semantics)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    @property
    def needs_netD(self):
        opt = self.opt
        return opt.isTrain and not (opt.no_gan_loss and opt.no_ganFeat_loss)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
