import os
import sys
import yaml
from my_nas.utils.common_utils import _parse_derive_file
import numpy as np

oneshot_derive_file = sys.argv[1]
res_dir = sys.argv[2]

with open(oneshot_derive_file, "r") as r_f:
    parsed = _parse_derive_file(r_f)
archs, perfs = zip(*[(arch, parsed[arch]["reward"]) for arch in parsed])

os.makedirs(os.path.join(res_dir, "cfgs/"), exist_ok=True)
with open("nb201_oneshot_supernet_training.yaml", "r") as r_f:
    base_cfg = yaml.load(r_f)

ratios = [0.5, 0.25, 0.1]
sorted_inds = np.argsort(perfs)
for ratio in ratios:
    num = int(len(archs) * ratio)
    inds = sorted_inds[-num:]
    arch_list = [archs[ind] for ind in inds]
    fname = os.path.join(res_dir, "nb201_archs_oneshot_keep_{:.2f}_{:d}.yaml".format(ratio, num))
    with open(fname, "w") as w_f:
        w_f.write("\n".join(arch_list))
    cfg_fname = os.path.join(res_dir, "cfgs/nb201_archs_oneshot_keep_{:.2f}_{:d}.yaml".format(ratio, num))

    base_cfg["controller_cfg"]["text_file"] = os.path.abspath(fname)
    with open(cfg_fname, "w") as w_f:
        yaml.dump(base_cfg, w_f)
