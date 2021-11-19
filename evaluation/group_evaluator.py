import torch.nn as nn
from .base_evaluator import BaseEvaluator
import util.util as util


def find_evaluator_using_name(filename):
    target_class_name = filename
    module_name = 'evaluation.' + filename
    eval_class = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(eval_class, BaseEvaluator), (
        "Class %s should be a subclass of BaseEvaluator" % eval_class
    )

    return eval_class


def find_evaluator_classes(opt):
    if len(opt.evaluation_metrics) == 0:
        return []

    eval_metrics = opt.evaluation_metrics.split(",")

    all_classes = []
    for metric in eval_metrics:
        metric_class = find_evaluator_using_name("%s_evaluator" % metric)
        all_classes.append(metric_class)

    return all_classes


class GroupEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        opt, _ = parser.parse_known_args()
        evaluator_classes = find_evaluator_classes(opt)

        for eval_class in evaluator_classes:
            parser = eval_class.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        evaluator_classes = find_evaluator_classes(opt)
        self.all_evaluators = [cls(opt) for cls in evaluator_classes]
        self.train_evaluators = [
            ev for ev in self.all_evaluators if ev.is_target_phase("train")
        ]
        self.test_evaluators = [
            ev for ev in self.all_evaluators if ev.is_target_phase("test")
        ]

    def evaluate_each_dataset(self, phase, dataloader, fn_model_forward, name=None):
        print("Evaluating on %s set..." % phase)
        self.current_phase = phase
        self.evaluators = (
            self.train_evaluators if phase == "train" else self.test_evaluators
        )
        if name is None:
            name = str(type(self).__name__)
        self.prepare_evaluation(dataloader, fn_model_forward, name)
        num_cum_samples = 0
        for i, data_i in enumerate(dataloader):
            # real = data_i["A"]
            real = data_i["image"]
            minibatch_size = real.size(0)
            self.evaluate_current_batch(data_i, fn_model_forward, name)
            num_cum_samples += minibatch_size
            if num_cum_samples % 1000 < minibatch_size:
                print("Evaluating %d samples so far..." % num_cum_samples)
            if self.should_stop_evaluation(num_cum_samples):
                break
        metrics = self.finish_evaluation(dataloader, fn_model_forward, name)
        return metrics

    def evaluate(
        self, train_dataset=None, test_dataset=None, fn_model_forward=None, name=None
    ):
        assert fn_model_forward is not None
        model = self.find_nn_module(fn_model_forward)
        if self.opt.use_eval_mode:
            model.eval()
        train_result = self.evaluate_each_dataset(
            "train", train_dataset, fn_model_forward, name
        )
        test_result = self.evaluate_each_dataset(
            "test", test_dataset, fn_model_forward, name
        )
        train_result.update(test_result)
        model.train()
        return train_result

    def prepare_evaluation(self, dataloader, fn_model_forward, name):
        [
            child.prepare_evaluation(
                self.current_phase, dataloader, fn_model_forward, name
            )
            for child in self.evaluators
        ]

    def evaluate_current_batch(self, data, fn_model_forward, name):
        [
            child.evaluate_current_batch(data, fn_model_forward, name)
            for child in self.evaluators
        ]

    def should_stop_evaluation(self, num_cum_samples):
        for child in self.evaluators:
            if not child.should_stop_evaluation(num_cum_samples):
                return False

        return True

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        return_dict = {}
        for child in self.evaluators:
            return_dict.update(
                child.finish_evaluation(dataloader, fn_model_forward, name)
            )
        return return_dict

    def find_nn_module(self, fn):
        def _find_nn(x):
            if isinstance(x, nn.Module):
                return x
            else:
                for k, v in vars(x).items():
                    return _find_nn(v)

        obj = getattr(fn, '__self__')
        if obj is not None:
            return _find_nn(obj)

