"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import json
import ntpath
import operator
import os
import time
from collections import defaultdict

import torch
import torch.utils.tensorboard as tb

from . import html, util

try:
    import wandb
except Exception:
    WANDB_AVAILBLE = False
else:
    WANDB_AVAILBLE = True


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and (not opt.no_tf_log)
        self.wandb_log = WANDB_AVAILBLE and opt.use_wandb
        self.use_json_log = opt.isTrain and (not opt.no_json_log)
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.wandb_run = None
        if self.tf_log:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tb.SummaryWriter(self.log_dir)
        if WANDB_AVAILBLE and opt.use_wandb:
            self.wandb_run = wandb.init(
                config=opt,
                sync_tensorboard=(opt.wandb_job_type != 'eval' and (not opt.no_tf_log)),
                job_type=opt.wandb_job_type,
                reinit=(opt.wandb_job_type == 'eval'),
            )
            if opt.wandb_job_type != 'eval':
                wandb.save(os.path.join(opt.checkpoints_dir, opt.name, '*'))

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print(f'create web directory {self.web_dir}...')
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Training Loss (%s) ================\n' % now
                )
            if self.use_json_log:
                self.json_log_name = os.path.join(
                    opt.checkpoints_dir, opt.name, 'log.json'
                )
                self.json_log = History.from_json(filename=self.json_log_name)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        # convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        for label, image_numpy in visuals.items():

            if len(image_numpy.shape) >= 4:
                image_numpy = image_numpy[0]

            if self.tf_log:  # show images in tensorboard output
                self.writer.add_image(
                    label,
                    torch.from_numpy(image_numpy).permute(2, 0, 1),
                    global_step=step,
                )
            if self.wandb_log:
                wandb.log({label: wandb.Image(image_numpy, caption=label)}, step=step)

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(
                            self.img_dir,
                            'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i),
                        )
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(
                        self.img_dir, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label)
                    )
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(
                self.web_dir, 'Experiment name = %s' % self.name, refresh=5
            )
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                                n,
                                step,
                                label,
                                i,
                            )
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(
                        ims[:num], txts[:num], links[:num], width=self.win_size
                    )
                    webpage.add_images(
                        ims[num:], txts[num:], links[num:], width=self.win_size
                    )
                step -= self.opt.display_freq
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        err = {}
        for tag, value in errors.items():
            value = value.mean().float()
            err[tag] = value
            if self.tf_log:
                self.writer.add_scalar(
                    os.path.join('errors', tag), value, global_step=step
                )
        self.json_log.add_scalars(err, int(step), main_tag='errors')
        self.json_log.to_json()
        if self.wandb_log:
            wandb.log({os.path.join('errors', k): v for k, v in err.items()}, step=step)

    def plot_current_metrics(self, metrics, step):
        if self.tf_log:
            self.json_log.add_scalars(metrics, int(step), main_tag='metrics')
            self.json_log.to_json()
            for tag, value in metrics.items():
                self.writer.add_scalar(
                    os.path.join('metrics', tag), value, global_step=step
                )
        if self.wandb_log:
            wandb.log(
                {os.path.join('metrics', k): v for k, v in metrics.items()},
                step=step,
            )

    def is_best_metric(self, metric_name, metric_val, step):
        name = metric_name.lower()
        if 'fid' in name or 'rms' in name or 'rel' in name or 'log10' in name:
            comparison_op = operator.lt
        else:
            comparison_op = operator.gt
        is_best = self.json_log.is_best(
            metric_name,
            metric_val,
            int(step),
            comparison_op,
        )
        self.json_log.to_json()
        if is_best and WANDB_AVAILBLE and self.opt.use_wandb:
            # wandb.run.summary['best_{}'.format(metric_name)] = (int(step), metric_val)
            wandb.run.summary['best_{}'.format(metric_name)] = metric_val
        return is_best

    def print_current_metrics(self, metrics, epoch, step):
        msg = 'epoch: {}, step: {}'.format(epoch, step)
        for name, value in metrics.items():
            msg += ', {}: {:.3f}'.format(name, value)
        print(msg)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batch_size > 8
            if key == 'input_label' and not self.opt.no_input_semantics:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


class History:
    def __init__(self, filename=None):
        self.filename = filename
        self.scalars = defaultdict(list)
        self.best_metrics = defaultdict(list)

    def add_scalar(self, k, v, step):
        self.scalars[k].append((step, v))

    def add_scalars(self, scalars_dict, step, main_tag=None):
        for k, v in scalars_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if main_tag is not None:
                k = os.path.join(main_tag, k)
            self.add_scalar(k, v, step)

    def get_scalar_names(self):
        return list(self.scalars.keys())

    def get_scalar(self, name):
        return list(zip(*self.scalars[name]))

    def get_total_steps(self):
        s = [v[-1][0] for v in self.scalars.values()]
        return max(s or [0])

    def update(self, vars_dict):
        self.filename = vars_dict['filename']
        self.scalars.update(vars_dict.get('scalars'))
        self.best_metrics.update(vars_dict.get('best_metrics', {}))

    def is_best(self, metric_name, metric_val, step, comparison_op=operator.gt):
        best = self.best_metrics[metric_name]
        if not best:
            self.best_metrics[metric_name].append((step, metric_val))
            is_best = True
        else:
            _, best_val = best[-1]
            is_best = comparison_op(metric_val, best_val)
            if is_best:
                self.best_metrics[metric_name].append((step, metric_val))
        return is_best

    def to_json(self, filename=None):
        fname = filename if filename is not None else self.filename
        with open(fname, 'w') as f:
            json.dump(vars(self), f)

    @classmethod
    def from_json(cls, filename):
        h = cls(filename)
        if os.path.exists(filename):
            with open(filename) as f:
                try:
                    data = json.load(f)
                except json.decoder.JSONDecodeError:
                    print('Could not load json log file. Re-initializing file...')
                else:
                    h.update(data)
        return h

    def __repr__(self) -> str:
        rep = 'History: {}\n'.format(self.filename)
        rep += 'Scalars:\n'
        for name in self.get_scalar_names():
            rep += '\t{}\n'.format(name)
        return rep


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        # image_name = '%s_%s.png' % (name, label)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)
