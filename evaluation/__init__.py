from .base_evaluator import *
from .group_evaluator import GroupEvaluator


def get_option_setter():
    return GroupEvaluator.modify_commandline_options

