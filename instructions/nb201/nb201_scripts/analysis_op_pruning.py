#pylint: disable-all
import numpy as np
import os
import pickle
from scipy.stats import stats, spearmanr
from my_nas.utils.common_utils import _parse_derive_file, format_as_float

from my_nas import utils
nb201_dir = os.path.join(utils.get_mynas_dir("MYNAS_DATA", "data"), "nasbench-201")
from my_nas.btcs.nasbench_201 import NasBench201SearchSpace

def get_deiso_dict():
    ss = NasBench201SearchSpace(17, 4, load_nasbench=False)
    query_dict = {}
    iso_group = []
    with open(os.path.join(nb201_dir, "deiso_dict.txt")) as iso_f:
        lines = iso_f.readlines()[1:]
        for line in lines:
            line_split = line.strip().split(" ")
            acc_list = []
            name_list = []
            for i in range(len(line_split) // 7):
                split_ = line_split[i*7:(i+1)*7]
                arch_mat = np.zeros([ss.num_vertices,ss.num_vertices]).astype(np.int32)
                ele = [float(i) for i in split_]
                arch_mat[np.tril_indices(ss.num_vertices, k=-1)] = ele[:int(ss.num_vertices * (ss.num_vertices - 1) / 2)]
                acc_list.append(ele[-1])
                name_list.append(ss.matrix2str(arch_mat))
            for name_ in name_list:
                query_dict[name_] = np.mean(acc_list)
            iso_group.append(name_list)
    return query_dict, iso_group
query_dict, iso_group = get_deiso_dict()


with open("./nb201_oneshot_accs.pkl", "rb") as r_f:
    archs, allop_accs = pickle.load(r_f)
allop_accs_1000 = allop_accs[:, -1]
allarch_to_accs = dict(zip(archs, allop_accs_1000.T))

result_dir = ("/home/Anonymous/projects/surgery/my_nas_private/"
              "examples/research/surgery/nb201/results/ops/results_derive/seed20")
other_seed_result_dirs = [
    "/home/Anonymous/projects/surgery/my_nas_private/"
    "examples/research/surgery/nb201/results/ops/results_derive/results_supernet_training_seed2020",
    "/home/Anonymous/projects/surgery/my_nas_private/"
    "examples/research/surgery/nb201/results/ops/results_derive/results_supernet_training_seed202020"]
sub_dirs = os.listdir(result_dir)

sub_higher_kds = []
sub_higher_avg_accs = []
for sub_dir in sub_dirs:
    with open(os.path.join(result_dir, sub_dir, "1000.yaml"), "r") as r_f:
        parsed = _parse_derive_file(r_f)
    all_, sub_, gt_ = zip(*[(allarch_to_accs[key], value["reward"], query_dict[key]) for key, value in parsed.items()])
    sub_list = [sub_]
    for other_seed_dir in other_seed_result_dirs:
        with open(os.path.join(other_seed_dir, sub_dir, "1000.yaml"), "r") as r_f:
            other_seed_parsed = _parse_derive_file(r_f)
        sub_list.append([other_seed_parsed[key]["reward"] for key in parsed])
    all_kds = []
    all_avg_accs = []
    sub_kds = []
    sub_avg_accs = []
    for seed_all_ in np.array(all_).T:
        all_kd = stats.kendalltau(seed_all_, gt_).correlation
        all_kds.append(all_kd)
        all_avg_accs.append(np.mean(seed_all_))
    for seed_sub_ in sub_list:
        sub_kds.append(stats.kendalltau(seed_sub_, gt_).correlation)
        sub_avg_accs.append(np.mean(seed_sub_))
    #sub_kd = stats.kendalltau(sub_, gt_).correlation
    #sub_avg_acc = np.mean(sub_)
    print("{}: num arch {}; sub ss {} (avg acc {}); overall ss {} (avg acc {}) ... KD increase: {:.4f}, ACC decrease: {:.4f}".format(sub_dir, len(gt_), sub_kds, sub_avg_accs, format_as_float(all_kds, "{:.4f}"), format_as_float(all_avg_accs, "{:.4f}"), np.mean(all_kds)-np.mean(sub_kds), np.mean(sub_avg_accs)-np.mean(all_avg_accs)))

    gt_ = np.array(gt_)
    subss_bestacc_10 = [gt_[np.argsort(seed_sub_)[-10:]].max() for seed_sub_ in sub_list]
    subss_bestacc_1 = [gt_[np.argmax(seed_sub_)] for seed_sub_ in sub_list]
    print("sub-ss BestAcc@1: {} {:.4f}+-{:.4f}".format(format_as_float(subss_bestacc_1, "{:.4f}"), np.mean(subss_bestacc_1), np.std(subss_bestacc_1)))
    print("sub-ss BestAcc@10: {} {:.4f}+-{:.4f}".format(format_as_float(subss_bestacc_10, "{:.4f}"), np.mean(subss_bestacc_10), np.std(subss_bestacc_10)))
    # print("BestAcc@1: {:.4f}".format(gt_[np.argmax(sub_)]))
    # print("BestAcc@10: {:.4f}".format(gt_[np.argsort(sub_)[-10:]].max()))

    # in subss
    fullss_bestacc_10 = [gt_[np.argsort(seed_all_)[-10:]].max() for seed_all_ in np.array(all_).T]
    fullss_bestacc_1 = [gt_[np.argmax(seed_all_)] for seed_all_ in np.array(all_).T]
    print("full-ss BestAcc@1: {} {:.4f}+-{:.4f}".format(format_as_float(fullss_bestacc_1, "{:.4f}"), np.mean(fullss_bestacc_1), np.std(fullss_bestacc_1)))
    print("full-ss BestAcc@10: {} {:.4f}+-{:.4f}".format(format_as_float(fullss_bestacc_10, "{:.4f}"), np.mean(fullss_bestacc_10), np.std(fullss_bestacc_10)))


    if np.mean(sub_kds) > np.mean(all_kds):
        sub_higher_kds.append(sub_dir)
    if np.mean(sub_avg_accs) > np.mean(all_avg_accs):
        sub_higher_avg_accs.append(sub_dir)

print("Sub search space higher kd: {}; NOT: {}".format(sub_higher_kds, set(sub_dirs).difference(sub_higher_kds)))
print("Sub search space higher avg acc: {}; NOT: {}".format(sub_higher_avg_accs, set(sub_dirs).difference(sub_higher_avg_accs)))

# in fullss
full_all_, full_gt_ = zip(*[(allarch_to_accs[key], query_dict[key]) for key in allarch_to_accs])
full_all_ = np.array(full_all_)
full_gt_ = np.array(full_gt_)
fullss_bestacc_10 = [full_gt_[np.argsort(seed_full_all_)[-10:]].max() for seed_full_all_ in np.array(full_all_).T]
fullss_bestacc_1 = [full_gt_[np.argmax(seed_full_all_)] for seed_full_all_ in np.array(full_all_).T]
print("full-ss (in full-ss) BestAcc@1: {} {:.4f}+-{:.4f}".format(format_as_float(fullss_bestacc_1, "{:.4f}"), np.mean(fullss_bestacc_1), np.std(fullss_bestacc_1)))
print("full-ss (in full-ss) BestAcc@10: {} {:.4f}+-{:.4f}".format(format_as_float(fullss_bestacc_10, "{:.4f}"), np.mean(fullss_bestacc_10), np.std(fullss_bestacc_10)))

"""
del_avg_pool_3x3_nor_conv1x1: num arch 114; sub ss 0.5822 (avg acc 0.7596); overall ss [0.6782, 0.6944, 0.7129] (avg acc [0.6889, 0.7004, 0.6883]) ... KD increase: 0.1129, ACC decrease: 0.0670
del_nor_conv_1x1_op: num arch 1219; sub ss 0.6979 (avg acc 0.6348); overall ss [0.8249, 0.7769, 0.8281] (avg acc [0.6276, 0.6473, 0.6149]) ... KD increase: 0.1121, ACC decrease: 0.0049
del_avg_pool_3x3: num arch 1219; sub ss 0.5018 (avg acc 0.8064); overall ss [0.5609, 0.6439, 0.6169] (avg acc [0.7249, 0.7231, 0.7239]) ... KD increase: 0.1054, ACC decrease: 0.0825

del_none: num arch 3131; sub ss 0.7431 (avg acc 0.7072); overall ss [0.7871, 0.7223, 0.7875] (avg acc [0.6917, 0.7101, 0.6863]) ... KD increase: 0.0226, ACC decrease: 0.0111
del_skip_connect: num arch 2155; sub ss 0.7214 (avg acc 0.6716); overall ss [0.7555, 0.7377, 0.7359] (avg acc [0.6981, 0.7048, 0.6945]) ... KD increase: 0.0216, ACC decrease: -0.0275
del_nor_conv_3x3_op: num arch 1215; sub ss 0.6984 (avg acc 0.6223); overall ss [0.6897, 0.6284, 0.6946] (avg acc [0.5880, 0.6006, 0.5764]) ... KD increase: -0.0275, ACC decrease: 0.0340

Sub search space higher kd: ['del_nor_conv_3x3_op']; NOT: {'del_avg_pool_3x3', 'del_avg_pool_3x3_nor_conv1x1', 'del_none', 'del_skip_connect', 'del_nor_conv_1x1_op'}
Sub search space higher avg acc: ['del_avg_pool_3x3_nor_conv1x1', 'del_nor_conv_1x1_op', 'del_nor_conv_3x3_op', 'del_none', 'del_avg_pool_3x3']; NOT: {'del_skip_connect'}
"""
# add seed 2020/202020
"""
del_nor_conv_1x1_op: num arch 1219; sub ss [0.6978765576557312, 0.7484929572500585, 0.6896617638171371] (avg acc [0.6348309269893355, 0.6364229696472519, 0.6084351107465135]); overall ss [0.8249, 0.7769, 0.8281] (avg acc [0.6276, 0.6473, 0.6149]) ... KD increase: 0.0980, ACC decrease: -0.0034
del_avg_pool_3x3_nor_conv1x1: num arch 114; sub ss [0.5822117143365987, 0.7023374795336006, 0.6683266245940789] (avg acc [0.7595929824561403, 0.7957070175438596, 0.7383526315789474]); overall ss [0.6782, 0.6944, 0.7129] (avg acc [0.6889, 0.7004, 0.6883]) ... KD increase: 0.0442, ACC decrease: 0.0720
del_none: num arch 3131; sub ss [0.7430879472456602, 0.6891966549235501, 0.743655625642267] (avg acc [0.7071514532098371, 0.7399118492494411, 0.692499872245289]); overall ss [0.7871, 0.7223, 0.7875] (avg acc [0.6917, 0.7101, 0.6863]) ... KD increase: 0.0403, ACC decrease: 0.0171

del_avg_pool_3x3: num arch 1219; sub ss [0.5018247312249835, 0.6164871201168167, 0.6264425456843382] (avg acc [0.8064023789991797, 0.8146175553732568, 0.8258912223133716]); overall ss [0.5609, 0.6439, 0.6169] (avg acc [0.7249, 0.7231, 0.7239]) ... KD increase: 0.0256, ACC decrease: 0.0917
del_skip_connect: num arch 2155; sub ss [0.7214023937444554, 0.7300508518501791, 0.7217254038436358] (avg acc [0.671585614849188, 0.6962722969837587, 0.6965246403712297]); overall ss [0.7555, 0.7377, 0.7359] (avg acc [0.6981, 0.7048, 0.6945]) ... KD increase: 0.0187, ACC decrease: -0.0110
del_nor_conv_3x3_op: num arch 1215; sub ss [0.6983940007200585, 0.6452885159688463, 0.6262781244313752] (avg acc [0.6223326748971193, 0.6069339917695473, 0.6167368724279835]); overall ss [0.6897, 0.6284, 0.6946] (avg acc [0.5880, 0.6006, 0.5764]) ... KD increase: 0.0143, ACC decrease: 0.0270


"""
