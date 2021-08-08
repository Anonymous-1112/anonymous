#pylint: disable-all
import argparse
import pickle
from collections import OrderedDict

import yaml

from my_nas.main import _init_component
from my_nas.common import rollout_from_genotype_str

parser = argparse.ArgumentParser()
parser.add_argument("cfg_file")
parser.add_argument("--arch-file", default="/home/Anonymous/mynas/data/nasbench-201/iso.txt")
args = parser.parse_args()

with open(args.cfg_file, "r") as r_f:
    cfg = yaml.load(r_f)

with open(args.arch_file, "r") as r_f:
    genotypes = [line.strip() for line in r_f.read().strip().split("\n")]
num_rollouts = len(genotypes)
device = "cuda"
rollout_type = cfg["rollout_type"]
search_space = _init_component(cfg, "search_space", load_nasbench=False)

wm = _init_component(cfg, "weights_manager",
                     search_space=search_space,
                     device=device,
                     rollout_type=rollout_type)

def calc_param(mod, valid_from_to):
    actives = dict(mod.active_named_members("parameters"))
    # print("len actives:", len(actives))
    valid_from_to_key = ["f_{}_t_{}".format(from_, to_) for from_, to_ in valid_from_to]
    return sum([param.nelement() for name, param in actives.items() if "edge_mod" not in name or any([key in name for key in valid_from_to_key])])


rollouts = [rollout_from_genotype_str(geno, search_space) for geno in genotypes]
num_params = OrderedDict()
for i_rollout, rollout in enumerate(rollouts):
    arch = rollout.arch
    valid_input = [0]
    for to_ in range(1, 4):
        if arch[to_, valid_input].sum() > 0:
            valid_input.append(to_)
    valid_output = [3]
    for from_ in range(2, -1, -1):
        if arch[valid_output, from_].sum() > 0:
            valid_output.append(from_)
    cand_net = wm.assemble_candidate(rollout)
    valid_from_to = [(from_, to_) for  to_ in range(1, 4) for from_ in range(0, to_) if from_ in valid_input and to_ in valid_output]
    #print(valid_from_to)
    num_params[genotypes[i_rollout]] = calc_param(cand_net, valid_from_to)
    print("\r{}/{}: {}".format(i_rollout+1, num_rollouts, num_params[genotypes[i_rollout]]), end="")

with open("/home/eva_share_users/Anonymous/surgery/notebook/results/nb201_params.pkl", "wb") as w_f:
    pickle.dump(num_params, w_f)
