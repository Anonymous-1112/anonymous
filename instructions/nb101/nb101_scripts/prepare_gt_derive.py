
import pickle
import os
import numpy as np

from functools import partial

import torch 

from my_nas import utils

from my_nas.btcs.nasbench_101 import *
accs = []
losses = []
gt = []

ss = NasBench101SearchSpace()
ss._init_nasbench()
nb = ss.nasbench
for d in [20, 2020, 202020]:
    archs += [[]]
    for i in [1, 2, 5, 10, 15, 20, 40, 60, 80] + list(range(100, 1001, 100)):
        pkl_name = "seed{}@{}.pkl".format(d, i)
        with open(pkl_name, "rb") as fr:
            rollouts = pickle.load(fr)
            arch = [(r.genotype, r.get_perf("acc"), r.get_perf("loss")) for r in
                    rollouts]
            for a in arch:
                a[0].matrix = a[0].matrix.astype(int)
            gt = [nb.query(a[0]) for a in arch]
            acc = [a[1] for a in arch]
            loss = [a[2] for a in arch]
        archs[-1] += [arch]

accs = np.array(accs)
losses = np.array(losses)

weights_manager_type = 'nasbench-101'
weights_manager_cfg = dict(
  # Schedulable attributes: 
  rollout_type='nasbench-101',
  gpus=[],
)
from my_nas.main import _init_component

ss = NasBench101OneShot3SearchSpace(load_nasbench=False)
cfg = {"weights_manager_type": weights_manager_type,
        "weights_manager_cfg": weights_manager_cfg}
wm = _init_component(cfg, "weights_manager", search_space=ss, device="cuda:0")

def reset_flops(self):
    self._flops_calculated = False
    self.total_flops = 0
    self.total_params = 0

def set_hook(self):
    for name, module in self.named_modules():
        module.register_forward_hook(partial(_hook_intermediate_feature, self))

def _hook_intermediate_feature(self, module, inputs, outputs):
    if not self._flops_calculated:
        if isinstance(module, nn.Conv2d):
            self.total_flops += inputs[0].size(1) * outputs.size(1) * \
                                module.kernel_size[0] * module.kernel_size[1] * \
                                inputs[0].size(2) * inputs[0].size(3) / \
                                (module.stride[0] * module.stride[1] * module.groups)
            self.total_params += module.kernel_size[0] * module.kernel_size[1] * \
                                 module.weight.shape[0] * module.weight.shape[1]
        elif isinstance(module, nn.Linear):
            self.total_flops += inputs[0].size(1) * outputs.size(1)
            self.total_params += inputs[0].size(1) * outputs.size(1)
    else:
        pass


inputs = torch.rand([1, 3, 32, 32]).to("cuda:0")

with open("seed20@100.pkl", "rb") as fr:
    rollouts = pickle.load(fr)

set_hook(wm)

flops = []
params = []
for r in rollouts:
    reset_flops(wm)
    cand = wm.assemble_candidate(r)
    cand.forward(inputs)
    flops += [wm.total_flops]
    params += [wm.total_params]


with open("stat_acc.pkl", "wb") as fw:
    pickle.dump({"derive": accs, "loss": losses, "gt": gt, "flops": np.array(flops), "params":
        np.array(params)}, fw)


