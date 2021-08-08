#pylint:disable-all
import pickle
import copy
import yaml
from collections import OrderedDict
from my_nas.utils import count_parameters
from my_nas.main import _init_component
from my_nas.common import rollout_from_genotype_str
from my_nas.weights_manager import BaseWeightsManager
from my_nas.btcs.nasbench_301 import NB301SearchSpace

cfg = {}
cfg["final_model_type"] = "cnn_final_model"
cfg["final_model_cfg"] = dict(
    auxiliary_cfg=None,
    auxiliary_head=False,
    dropout_path_rate=0.2,
    dropout_rate=0.1,
    init_channels=36,
    num_classes=10,
    stem_multiplier=3
)
ss = NB301SearchSpace()
wm_cfg = {
    "rollout_type": "nb301",
    "gpus": [],
    "num_classes": 10,
    "init_channels": 32,
    "stem_multiplier": 3,
    "max_grad_norm": 5.0,
    "dropout_rate": 0.1,
    "use_stem": "conv_bn_3x3",
    "stem_stride": 1,
    "stem_affine": True,
    "preprocess_op_type": None,
    "cell_use_preprocess": True,
    "cell_group_kwargs": None,
    "cell_use_shortcut": False,
    "cell_shortcut_op_type": "skip_connect",
    "candidate_member_mask": True,
    "candidate_cache_named_members": False,
    "candidate_virtual_parameter_only": False,
    "candidate_eval_no_grad": True
}
#wm_16c_cfg = copy.deepcopy(wm_cfg)
#wm_16c_cfg["init_channels"] = 16
wm = BaseWeightsManager.get_class_("supernet")(ss, "cuda:0", **wm_cfg)
#wm_16c = BaseWeightsManager.get_class_("supernet")(ss, "cuda:0", **wm_16c_cfg)

keys = ["sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3", "dil_conv_5x5"]
#op_to_arch_to_param = OrderedDict([(key, OrderedDict()) for key in keys])
#op_to_arch_to_param_16c = OrderedDict([(key, OrderedDict()) for key in keys])
op_to_arch_to_param_32c = OrderedDict([(key, OrderedDict()) for key in keys])
#op_to_arch_to_param["ori"] = []
#op_to_arch_to_param_16c["ori"] = []
op_to_arch_to_param_32c["ori"] = []
for key, op in zip(
        keys,
        ["sep3x3", "sep5x5", "dil3x3", "dil5x5"]):
    print(key)
    with open("change_non_to_{}.yaml".format(op), "r") as r_f:
        change_to_archs = yaml.load(r_f)
    for i_arch, arch in enumerate(change_to_archs):
        print("\r{}".format(i_arch), end="")
        rollout = rollout_from_genotype_str(arch, ss)
        cand_net = wm.assemble_candidate(rollout)
        op_to_arch_to_param_32c[key][arch] = count_parameters(cand_net)[0]
        #cand_net = wm_16c.assemble_candidate(rollout)
        #op_to_arch_to_param_16c[key][arch] = count_parameters(cand_net)[0]

with open("/home/eva_share_users/Anonymous/surgery/nb301/archs.yaml", "r") as r_f:
    ori_archs = yaml.load(r_f)
    key = "ori"
for i_arch, arch in enumerate(ori_archs):
    print("\r{}".format(i_arch), end="")
    rollout = rollout_from_genotype_str(arch, ss)
    cand_net = wm.assemble_candidate(rollout)
    op_to_arch_to_param_32c[key].append(count_parameters(cand_net)[0])
    #cand_net = wm_16c.assemble_candidate(rollout)
    #op_to_arch_to_param_16c[key].append(count_parameters(cand_net)[0])

# with open("nb301_increase1parm_param_16c.pkl", "wb") as w_f:
#     pickle.dump(op_to_arch_to_param_16c, w_f)


with open("nb301_increase1parm_param_32c.pkl", "wb") as w_f:
    pickle.dump(op_to_arch_to_param_32c, w_f)
