"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import contextlib

import torch
import torch.optim as optim
import util.util as util

import models.networks as networks


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = (
            torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        )
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

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

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

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
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = (opt.lr, opt.lr) if opt.no_TTUR else (opt.lr / 2, opt.lr * 2)
        if opt.optimizer.lower() == 'adam':
            optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        elif opt.optimizer.lower() == 'ranger':
            optimizer_G = optim.Ranger(G_params, lr=G_lr, betas=(beta1, beta2))
            optimizer_D = optim.Ranger(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            raise ValueError('Optimizer {} not recognized'.format(opt.optimizer))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)
        if vars(self.opt).get('unfrozen_vgg'):
            util.save_network(self.criterionVGG, 'VGG', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        resume_func = (
            self.load_pretrained_networks
            if self.opt.pretrained_name is not None
            else self.resume_networks
        )
        try:
            netG, netD, netE = resume_func(opt, netG, netD, netE)
        except FileNotFoundError as e:
            print(e)
            print('WARNING: Checkpoints for {} not found!.'.format(self.opt.name))
            print('Networks initialized from scratch')
        return netG, netD, netE

    def resume_networks(self, opt, netG, netD, netE):
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            if vars(opt).get('unfrozen_vgg'):
                self.criterionVGG = util.load_network(
                    self.criterionVGG, 'VGG', opt.which_epoch, opt
                )

        return netG, netD, netE

    def load_pretrained_networks(self, opt, netG, netD, netE):
        pretrained_nets = opt.pretrained_nets.split(',')
        epoch = opt.which_epoch
        if not opt.isTrain or opt.continue_train:
            if 'G' in pretrained_nets:
                netG = util.load_pretrained_network(netG, 'G', epoch, opt)
                print('loaded pretrained G: {}'.format(opt.pretrained_name))
            with contextlib.suppress(FileNotFoundError):
                if opt.isTrain and 'D' in pretrained_nets:
                    netD = util.load_pretrained_network(netD, 'D', epoch, opt)
                    print('loaded pretrained D: {}'.format(opt.pretrained_name))
                if opt.use_vae and 'E' in pretrained_nets:
                    netE = util.load_pretrained_network(netE, 'E', epoch, opt)
                    print('loaded pretrained E: {}'.format(opt.pretrained_name))
                if vars(opt).get('unfrozen_vgg'):
                    self.criterionVGG = util.load_network(
                        self.criterionVGG, 'VGG', epoch, opt
                    )
                    print('loaded pretrained VGG: {}'.format(opt.pretrained_name))
        return netG, netD, netE

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
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae
        )

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = None, None
        if (not self.opt.no_gan_loss) or (not self.opt.no_ganFeat_loss):
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image
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

        return G_losses, fake_image

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

            if mode == "forward":
                visuals["fake_B"] = self.netG(input_semantics)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def test(self, data=None):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            input_semantics, _ = self.preprocess_input(data)
            self.generated = self.netG(input_semantics)
            self.image_paths = data['path']
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_current_visuals(self):
        return {
            # 'label': self.curr_label,
            'real': self.curr_real_image,
            'fake': self.generated,
        }

    def get_image_paths(self):
        return self.image_paths
