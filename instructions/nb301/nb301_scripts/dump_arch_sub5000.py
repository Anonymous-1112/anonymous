import yaml
import copy

with open("/home/eva_share_users/Anonymous/surgery/nb301/archs.yaml", "r") as r_f:
    archs = yaml.load(r_f)
ops = [
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect"
]
for op in ops:
    inds = []
    for ind, arch in enumerate(archs):
        if op not in arch:
            inds.append(ind)
    delop_archs = [archs[ind] for ind in inds]
    derive_arch_file = "/home/eva_share_users/Anonymous/surgery/nb301/results/delop_supernet_training/seed20/DEL_{}/derive_archs.yaml".format(op)
    with open(derive_arch_file, "w") as w_f:
        yaml.dump(delop_archs, w_f)
    print("Del op: {}. num: {}".format(op, len(inds)))



"""
Del op: sep_conv_3x3. num: 499
Del op: sep_conv_5x5. num: 559
Del op: dil_conv_3x3. num: 956
Del op: dil_conv_5x5. num: 927
Del op: max_pool_3x3. num: 977
Del op: avg_pool_3x3. num: 1198
Del op: skip_connect. num: 568
"""
