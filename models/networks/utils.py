import copy
from collections import OrderedDict, defaultdict
from functools import partial

import torch


class FeatureHooks:
    def __init__(self, hooks, named_modules):
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for h in hooks:
            hook_name = h['name']
            m = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            if h['type'] == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif h['type'] == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, name, *args):
        x = args[
            -1
        ]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][name] = x

    def get_output(self, device=None):
        if device is None:
            devs = list(self._feature_outputs.keys())
            assert len(devs) == 1
            device = devs[0]
        output = tuple(self._feature_outputs[device].values())
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output


class EMA(torch.nn.Module):
    """Apply EMA to a model.

    Simple wrapper that applies EMA to a model. Could be better done in 1.0 using
    the parameters() and buffers() module functions, but for now this works
    with state_dicts using .copy_

    """

    def __init__(self, source, target=None, decay=0.9999, start_itr=0):
        super().__init__()
        self.source = source
        self.target = target if target is not None else copy.deepcopy(source)
        self.decay = decay
        for param in self.target.parameters():
            param.requires_grad = False

        # Optional parameter indicating what iteration to start the decay at.
        self.start_itr = start_itr

        # Initialize target's params to be source's.
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()

        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(
                    self.target_dict[key].data * decay
                    + self.source_dict[key].data * (1 - decay)
                )

    def __repr__(self):
        return (
            f'Source: {type(self.source).__name__}\n'
            f'Target: {type(self.target).__name__}'
        )

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)
