import yaml
import copy
import random
from my_nas.btcs.nasbench_301 import NB301SearchSpace
from my_nas.common import genotype_from_str

with open("/home/eva_share_users/Anonymous/surgery/nb301/archs.yaml", "r") as r_f:
    geno_strs = yaml.load(r_f)


ss = NB301SearchSpace()
non_parametrized = {
    "skip_connect",
    "avg_pool_3x3",
    "max_pool_3x3"
}
parametrized_ops = [
    #"sep_conv_3x3"
    # "sep_conv_5x5"
    #"dil_conv_3x3"
    "dil_conv_5x5"
]
random.seed(20)
continue_num = 0
geno_change_list = []
new_geno_list = []
for i_geno, geno_str in enumerate(geno_strs):
    geno = genotype_from_str(geno_str, ss)
    new_geno = copy.deepcopy(geno)
    non_param_subset = [
        (ind, edge) for ind, edge in enumerate(geno.normal_0 + geno.reduce_1) if edge[0] in non_parametrized]
    if not non_param_subset:
        print("Skip {}".format(geno_str))
        continue_num += 1
        continue
    change_from = random.choice(non_param_subset)
    change_to = random.choice(parametrized_ops)
    edge = (change_to, change_from[1][1], change_from[1][2])
    change_ind = change_from[0]
    change_ind, cell = change_ind % 8, (new_geno.reduce_1 if change_ind // 8 else new_geno.normal_0)
    cell[change_ind] = edge
    print("#{}: Change {} to {}: {} {}".format(i_geno+1, change_from, change_to, geno, new_geno))
    geno_change_list.append((("reduce" if change_ind // 8 else "normal", change_from[1][1:],
                              change_from[1][0], change_to), str(geno), str(new_geno)))
    new_geno_list.append(str(new_geno))
print("skip num: {}; Total pairs: {}".format(continue_num, len(new_geno_list)))

with open("change_non_to_dil5x5_details.yaml", "w") as w_f:
    yaml.dump(geno_change_list, w_f)

with open("change_non_to_dil5x5.yaml", "w") as w_f:
    yaml.dump(new_geno_list, w_f)
# ----
# with open("change_non_to_dil3x3_details.yaml", "w") as w_f:
#     yaml.dump(geno_change_list, w_f)

# with open("change_non_to_dil3x3.yaml", "w") as w_f:
#     yaml.dump(new_geno_list, w_f)
# # ----
# with open("change_non_to_sep5x5_details.yaml", "w") as w_f:
#     yaml.dump(geno_change_list, w_f)

# with open("change_non_to_sep5x5.yaml", "w") as w_f:
#     yaml.dump(new_geno_list, w_f)

# ----
# with open("change_non_to_sep3x3_details.yaml", "w") as w_f:
#     yaml.dump(geno_change_list, w_f)

# with open("change_non_to_sep3x3.yaml", "w") as w_f:
#     yaml.dump(new_geno_list, w_f)
