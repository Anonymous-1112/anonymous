import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pickle
import os

import pickle
from scipy.stats import stats, spearmanr
import pandas as pd
import seaborn as sns
import copy

with open("./fig_surgery/fig4_nb301_zs_stat.pkl", "rb") as r_f:
    stat = {k: v for k, v in pickle.load(r_f).items()}

statss = [stat[f"zeroshot_301/seed{k}/batch5.pkl"][1] for k in [20, 2020, 202020]]

keys = [x for x in list(statss[0].keys()) if x != "oneshot_loss"]
indicators = list(statss[0][keys[0]].keys())

stats_dict = {i: {k: [s[k][i] for s in statss] for k in keys} for i in indicators}
res = stats_dict

def minmax_n_at_k(predict_scores, true_scores, ks=[0.01, 0.05, 0.10, 0.20]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:int(k * len(true_scores))]]
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks


def p_at_tb_k(predict_scores, true_scores, ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs - k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks


criteria = {
    "oneshot average": lambda x, y: np.mean(x),
    "linear": lambda x, y: np.corrcoef(x, y)[0][1],
    "kd": lambda x, y: stats.kendalltau(x, y).correlation,
    "spearmanr": lambda x, y: spearmanr(x, y).correlation,
    "BWR@K": lambda x, y: minmax_n_at_k(x, y),
    "P@tbK": lambda x, y: p_at_tb_k(x, y),
}

with open("./fig_surgery/fig4_nb301_gt_stat_acc.pkl", "rb") as r_f:
    derived = pickle.load(r_f)

mean_acc = derived["derive"].mean(0)
gt = derived["gt"]

mean_acc_corr = {}
for ck, cv in criteria.items():
    mean_acc_corr[ck] = [[cv(m, gt)] for m in mean_acc]

with open("./fig_surgery/fig4_nb301_os_stat.pkl", "rb") as r_f:
    oneshot_stat = {k: v for k, v in pickle.load(r_f).items()}

# print(oneshot_stat.keys())
seeds = [20, 2020, 202020]
oneshot_statss = []
for seed in seeds:
    keep = {k: v for k, v in oneshot_stat.items()}
    oneshot_statss.append(keep[f"derived_bs{seed}/1000.pkl"]["non_sparse"])

oneshot_stat_dict = {i: {k: [s[k][i] for s in oneshot_statss] for k in oneshot_statss[0].keys()} for i in indicators}


def plot_indicators_to_ax(ax, name):
    labels = ["Linear correlation", "Spearman correlation", "Kendall's tau", "P@top5%", "P@bottom5%", "BR@5%", "WR@5%"]
    colors = ["#EF8536", "#519D3E", "#C53932", "#8D6AB8", "#8D6AB8"]
    markers = {
        "P@bottom5%": "^"
    }
    linestyles = {"P@bottom5%": "--"}
    lines = []
    for i, (nums, color, label) in enumerate(zip([linear, spearmanrs, kds, p_at_top5, p_at_bot5], colors, labels)):
        if name == "loss" and i == 0:
            ax2 = ax.twinx()
            ax2.tick_params(labelsize=16)
            ax2.set_ylabel("Oneshot loss", fontsize=16)
            plot_ax = ax2
        else:
            plot_ax = ax
        avg, std = np.mean(nums, axis=-1), np.std(nums, axis=-1)
        lines.append(plot_ax.plot(epochs, avg, color=color, linewidth=2,
                                  label=label, linestyle=linestyles.get(label, "-"), mew=0.5,
                                  marker=markers.get(label, "o"))[0])
        plot_ax.fill_between(epochs, avg - std, avg + std, facecolor=color, alpha=0.2)
    return lines

keys = [x for x in list(statss[0].keys()) if x != "oneshot_loss"]
for k in res.keys():
    res[k].update(oneshot_stat_dict[k])
keys += list(oneshot_stat_dict[k].keys())

labels = []
new_res = {}
for i, vv in res.items():
    if i == "oneshot average":
        continue

    vv = {k: np.array(v) for k, v in vv.items()}
    if i == "P@tbK":
        # print(list(vv.values())[0][0,3,:])
        new_res["P@top5%"] = {k: v[:, 3, 2] for k, v in vv.items()}
        new_res["P@bottom5%"] = {k: v[:, 3, 3] for k, v in vv.items()}
    elif i == "BWR@K":
        new_res["BR@5%"] = {k: v[:, 2, 2] for k, v in vv.items()}
        new_res["WR@5%"] = {k: v[:, 2, 4] for k, v in vv.items()}
    else:
        new_res[i] = vv

with open("./fig_surgery/fig4_nb301_zeroshot.pkl", "rb") as fr:
    zeroshot = pickle.load(fr)

zeroshot = {k: np.array([np.array(z[k]).mean() for z in zeroshot]) for k in zeroshot[0].keys()}
zeroshot = {k: v if k != "plain" else -v for k, v in zeroshot.items()}

def vote_comparetor(mets1, mets2):
    # ranking mets from large to small
    diff = np.array(mets1[0]) - np.array(mets2[0])
    if (diff > 0).sum() > len(mets1[0]) // 2:
        return -1
    return 1


from functools import cmp_to_key

vote_indicators = ['grasp', 'plain', "relu_logdet"]
zeroshot['gt'] = gt
tobe_voted = np.array([zeroshot[v] for v in vote_indicators])
nb_archs = tobe_voted.shape[-1]
tobe_voted = [x for x in zip(tobe_voted.transpose(), range(nb_archs))]
vote_order = [x[1] for x in sorted(tobe_voted, key=cmp_to_key(vote_comparetor))]
vote_score = np.zeros(nb_archs)
vote_score[np.array(vote_order)] = -np.arange(0, nb_archs)
zeroshot["vote"] = vote_score
keys = list(keys) + ["vote"]

budgets = {"GT": gt, "FLOPs": derived["flops"], "Params": derived["params"], "1k": -derived["loss"][:, -1].mean(0)}
budgets.update(zeroshot)

correlation = {}
for ck, cv in criteria.items():
    corr = {k: {} for k in budgets.keys()}
    correlation[ck] = corr
    for bk1, bv1 in budgets.items():
        for bk2, bv2 in budgets.items():
            corr[bk1][bk2] = cv(bv1, bv2)

crit = ["kd", "spearmanr", "P@topK", "P@bottomK", "BWR@K"]

with open("./fig_surgery/fig4_nb201_gt_os.pkl", "rb") as fr:
    deiso_archs, deiso_gt_flops_params_acc, oneshot_accs = pickle.load(fr)

deiso_archs, deiso_gt_flops_params_acc, oneshot_accs = np.array(deiso_archs), np.array(deiso_gt_flops_params_acc), np.array(oneshot_accs)


flops = deiso_gt_flops_params_acc[:, 0]
params = deiso_gt_flops_params_acc[:, 1]
gt = deiso_gt_flops_params_acc[:, 2]

with open("./fig_surgery/fig4_nb201_zeroshot.pkl", "rb") as fr:
    zeroshot2 = pickle.load(fr)

zeroshot2 = {k: np.array([np.array(z[k]).mean() for z in zeroshot2]) for k in zeroshot2[0].keys()}
zeroshot2 = {k: v if k != "plain" else -v for k, v in zeroshot2.items()}

def vote_comparetor(mets1, mets2):
    # ranking mets from large to small
    diff = np.array(mets1[0]) - np.array(mets2[0])
    if (diff > 0).sum() > len(mets1[0]) // 2:
        return -1
    return 1

from functools import cmp_to_key
zeroshot2['gt'] = gt
zeroshot2['param'] = params
zeroshot2['jacob_cov_new'] = copy.deepcopy(zeroshot2['jacob_cov'])
zeroshot2['jacob_cov_new'][np.where(np.isnan(zeroshot2['jacob_cov']))[0]] = -104110

zeroshot2['jacob_cov'] = zeroshot2['jacob_cov_new']
zeroshot2.pop("jacob_cov_new")
#vote_indicators = ['synflow', 'jacob_cov', 'snip']
vote_indicators = ['relu_logdet', 'jacob_cov', 'synflow']
# shape: 3 x 5000
tobe_voted = np.array([zeroshot2[v] if v != 'plain' else -zeroshot2[v] for v in vote_indicators])
nb_archs = tobe_voted.shape[-1]
tobe_voted = [x for x in zip(tobe_voted.transpose(), range(nb_archs))]
vote_order = [x[1] for x in sorted(tobe_voted, key=cmp_to_key(vote_comparetor))]
vote_score = np.zeros(nb_archs)
vote_score[np.array(vote_order)] = -np.arange(0, nb_archs)
zeroshot2["vote"] = vote_score
keys = list(keys) +["vote"]

budgets = {"GT": gt, "FLOPs": flops, "Params":params, "1k": oneshot_accs[0, -1]}
budgets.update(zeroshot2)

correlation201 = {}
for ck, cv in criteria.items():
    corr = {k: {} for k in budgets.keys()}
    correlation201[ck] = corr
    for bk1, bv1 in budgets.items():
        for bk2, bv2 in budgets.items():
            corr[bk1][bk2] = cv(bv1, bv2)

crit = ["kd", "spearmanr", "P@topK", "P@bottomK", "BWR@K"]

rm = ["reward", "loss", "param", 'gt', 'synflow_bn', 'synflow_bn_eval', 'synflow_scale_1e-5']

order = ["GT", "FLOPs", "Params", "1k", "relu_logdet", "jacob_cov", "synflow", "snip", "grad_norm", "fisher", "grasp", "plain", "vote", "relu"]

kd301 = {k1: {k2: correlation["kd"][k1][k2] for k2 in order} for k1 in order}
kd201 = {k1: {k2: correlation201["kd"][k1][k2] for k2 in order} for k1 in order}
order = ["GT", "FLOPs", "Params", "1k", "jacob_cov", "synflow", "snip", "grad_norm", "plain", "vote"]

kd301 = pd.DataFrame(kd301)
kd201 = pd.DataFrame(kd201)

sns.set(font_scale=1.4)
gs = gridspec.GridSpec(1, ncols=2, width_ratios=[10,12.5])
fig = plt.figure(figsize=[30, 15])
ax = fig.add_subplot(gs[0, 0])
ax.set_title("NAS-Bench-201", fontsize=36)
sns.heatmap(kd201, annot=True, vmin=-0.2, vmax=1, square=True, cmap="Blues", cbar=False)
#cbar_ax = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("NAS-Bench-301", fontsize=36)
sns.heatmap(kd301, annot=True, vmin=-0.2, vmax=1, square=True, cmap="Blues", cbar=True, #cbar_ax=cbar_ax,
           cbar_kws={"shrink": 0.8})
plt.tight_layout()
plt.savefig("fig4.pdf")
#plt.show()
