#pylint: disable-all
import numpy as np
from my_nas.btcs.nasbench_201 import NasBench201Evaluator, NasBench201SearchSpace

ss = NasBench201SearchSpace(load_nasbench=True)

from collections import defaultdict
dict_ = defaultdict(list)
for arch_info in ss.api.arch2infos_full.values():
    print(arch_info)
    for seed in [777, 888, 999]:
        if ("cifar10", seed) in arch_info.all_results:
            dict_[arch_info.arch_str].append(arch_info.all_results[('cifar10', seed)].get_eval("ori-test")["accuracy"])
    dict_[arch_info.arch_str] = float(np.mean(dict_[arch_info.arch_str]))
print(len(dict_))

import yaml
with open("/home/Anonymous/mynas/data/nasbench-201/iso_dict_new.yaml", "w") as wf:
    yaml.dump(dict(dict_), wf)
