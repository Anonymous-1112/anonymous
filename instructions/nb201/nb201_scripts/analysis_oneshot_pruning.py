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


with open("/home/eva_share_users/Anonymous/surgery/notebook/results/oneshot/nb201/nb201_oneshot_accs.pkl", "rb") as r_f:
    archs, allop_accs = pickle.load(r_f)
allop_accs_1000 = allop_accs[:, -1]
allarch_to_accs = dict(zip(archs, allop_accs_1000.T))

result_dir = "/home/eva_share_users/Anonymous/surgery/nb201/results_subsearchspace/oneshot_pruning/seed20/results_derive/seed20/"
other_seed_result_dirs = [
    # "/home/Anonymous/projects/surgery/my_nas_private/"
    # "examples/research/surgery/nb201/results/ops/results_derive/results_supernet_training_seed2020",
    # "/home/Anonymous/projects/surgery/my_nas_private/"
    # "examples/research/surgery/nb201/results/ops/results_derive/results_supernet_training_seed202020"
]
# sub_dir_to_arch_dict = {
#     "": "/home/eva_share_users/Anonymous/surgery/nb201/results_subsearchspace/oneshot_pruning/seed20/nb201_archs_oneshot"
# }
#sub_dirs = os.listdir(result_dir)
sub_dirs = [
    "nb201_archs_oneshot_keep_0.10_646",
    "nb201_archs_oneshot_keep_0.25_1616",
    "nb201_archs_oneshot_keep_0.50_3233"
]


ONESHOT = True
sub_higher_kds = []
sub_higher_avg_accs = []
for sub_dir in sub_dirs:
    with open(os.path.join(result_dir, sub_dir, "1000.yaml"), "r") as r_f:
        parsed = _parse_derive_file(r_f)
    if ONESHOT:
        # for oneshot
        arch_file = "/home/eva_share_users/Anonymous/surgery/nb201/results_subsearchspace/oneshot_pruning/seed20/{}.yaml".format(sub_dir)
        with open(arch_file, "r") as r_f:
            archs = r_f.read().strip().split("\n")
        all_, sub_, gt_ = zip(*[(allarch_to_accs[key], parsed[key]["reward"], query_dict[key]) for key in archs])
    else:
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
    print("{}: num arch {}; sub ss {} (avg acc {}); overall ss {} {:.4f} (avg acc {} {:.4f}) ... KD increase: {:.4f}, ACC increase: {:.4f}".format(
        sub_dir, len(gt_), sub_kds, sub_avg_accs, format_as_float(all_kds, "{:.4f}"), np.mean(all_kds),
        format_as_float(all_avg_accs, "{:.4f}"), np.mean(all_avg_accs), np.mean(sub_kds)-np.mean(all_kds), np.mean(sub_avg_accs)-np.mean(all_avg_accs)))
    gt_ = np.array(gt_)
    print("BestAcc@1: {:.4f}".format(gt_[np.argmax(sub_)])) 
    print("BestAcc@10: {:.4f}".format(gt_[np.argsort(sub_)[-10:]].max()))

    fullss_bestacc_10 = [gt_[np.argsort(seed_all_)[-10:]].max() for seed_all_ in np.array(all_).T]
    fullss_bestacc_1 = [gt_[np.argmax(seed_all_)] for seed_all_ in np.array(all_).T]
    print("full-ss BestAcc@1: {} {:.4f}".format(format_as_float(fullss_bestacc_1, "{:.4f}"), np.mean(fullss_bestacc_1)))
    print("full-ss BestAcc@10: {} {:.4f}".format(format_as_float(fullss_bestacc_10, "{:.4f}"), np.mean(fullss_bestacc_10)))
    if not ONESHOT:
        assert len(gt_) == 6466
        numarch = len(gt_)
        gt_rank = np.zeros(numarch)
        gt_rank[np.argsort(gt_)] = np.arange(numarch)[::-1]
        print("BestRank@1: {:.4f} {}".format(gt_rank[np.argmax(sub_)] / float(numarch),gt_rank[np.argmax(sub_)]))
        print("BestRank@10: {:.4f} {}".format(gt_rank[np.argsort(sub_)[-10:]].min() / float(numarch), gt_rank[np.argsort(sub_)[-10:]].min()))
        fullss_bestrank_1 = np.array([gt_rank[np.argmax(seed_all_)] for seed_all_ in np.array(all_).T])
        fullss_bestrank_10 = np.array([gt_rank[np.argsort(seed_all_)[-10:]].min() for seed_all_ in np.array(all_).T])
        print(fullss_bestrank_1)
        print(fullss_bestrank_10)
        fullss_bestrank_1 = fullss_bestrank_1 / float(numarch)
        fullss_bestrank_10 = fullss_bestrank_10 / float(numarch)
        print("full-ss BestRank@1: {} {:.4f}".format(format_as_float(list(fullss_bestrank_1), "{:.4f}"), float(np.mean(fullss_bestrank_1))))
        print("full-ss BestRank@10: {} {:.4f}".format(format_as_float(list(fullss_bestrank_10), "{:.4f}"), float(np.mean(fullss_bestrank_10))))
    if np.mean(sub_kds) > np.mean(all_kds):
        sub_higher_kds.append(sub_dir)
    if np.mean(sub_avg_accs) > np.mean(all_avg_accs):
        sub_higher_avg_accs.append(sub_dir)

print("Sub search space higher kd: {}; NOT: {}".format(sub_higher_kds, set(sub_dirs).difference(sub_higher_kds)))
print("Sub search space higher avg acc: {}; NOT: {}".format(sub_higher_avg_accs, set(sub_dirs).difference(sub_higher_avg_accs)))

# seed 20, oneshot arch subset
"""
nb201_archs_oneshot_keep_0.25_1616: num arch 1616; sub ss [0.4386733442568404] (avg acc [0.7722189356435644]); overall ss [0.3934, 0.2691, 0.3952] (avg acc [0.7582, 0.7624, 0.7584]) ... KD increase: 0.0861, ACC increase: 0.0126
nb201_archs_oneshot_keep_0.50_3233: num arch 3233; sub ss [0.5531584688323531] (avg acc [0.7563412001237241]); overall ss [0.5457, 0.4429, 0.5336] (avg acc [0.7454, 0.7512, 0.7461]) ... KD increase: 0.0458, ACC increase: 0.0088
nb201_archs_oneshot_keep_0.10_646: num arch 646; sub ss [0.4532067994475687] (avg acc [0.7889848297213622]); overall ss [0.2422, 0.2265, 0.3242] (avg acc [0.7654, 0.7696, 0.7658]) ... KD increase: 0.1889, ACC increase: 0.0221
Sub search space higher kd: ['nb201_archs_oneshot_keep_0.25_1616', 'nb201_archs_oneshot_keep_0.50_3233', 'nb201_archs_oneshot_keep_0.10_646']; NOT: set()
Sub search space higher avg acc: ['nb201_archs_oneshot_keep_0.25_1616', 'nb201_archs_oneshot_keep_0.50_3233', 'nb201_archs_oneshot_keep_0.10_646']; NOT: set()
"""
