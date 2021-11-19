from .base_evaluator import BaseEvaluator
import torch.nn.functional as F
from collections import OrderedDict


class SingleImageEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.visualizer = opt.visualizer
        self.count = 0

    def is_target_phase(self, phase):
        return phase == "test"

    def prepare_evaluation(self, phase, dataloader, fn_model_forward, name):
        self.count += 1

    def evaluate_current_batch(self, data, fn_model_forward, name):
        AtoB = self.opt.direction == "AtoB"
        source = data["A" if AtoB else "B"]
        target = data["B" if AtoB else "A"]
        ih, iw = source.size(2), source.size(3)
        factor = 0.5
        oh = int(round(ih * factor / 4)) * 4
        ow = int(round(iw * factor / 4)) * 4
        source = F.interpolate(source, (oh, ow))
        out = fn_model_forward({"A": source, "B": source}, "forward")["fake_B"]
        target = F.interpolate(target, (source.size(2), source.size(3)))
        if self.opt.learned_minibatch_reweighting:
            target = target.cuda()
            weightmap = fn_model_forward({"A": target, "B": target}, "weightmap")["weightmap"]
            #target = weightmap
            target = (target + 1) * weightmap
            target = (target - target.min()) / (target.max() - target.min()) * 2 - 1.0
        result = OrderedDict([('source', source),
                              ('fake', out),
                              ('target', target)])
        self.visualizer.display_current_results(result, self.count, False)

    def should_stop_evaluation(self, num_cum_samples):
        return True

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        return {}
