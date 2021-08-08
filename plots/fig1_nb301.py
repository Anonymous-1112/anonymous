import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pickle
import os

from scipy.stats import stats, spearmanr

seeds = [20, 2020, 202020]
statss = []
for seed in seeds:
    with open("./fig_surgery/fig1_nb301_seed{seed}_stat.pkl".format(seed=seed), "rb") as r_f:
        stat = {k: v["non_sparse"] for k, v in pickle.load(r_f).items()}
        statss.append(stat)

stats_list = [list(zip(*sorted([(int(os.path.basename(key).split(".")[0]), value) for key, value in stat.items()], key=lambda item: item[0]))) for stat in statss]

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

with open("/home/novauto/derived_res/derive_301/stat_acc.pkl", "rb") as r_f:
    derived = pickle.load(r_f)

mean_acc = derived["derive"].mean(0)
gt = derived["gt"]

mean_acc_corr = {}
for ck, cv in criteria.items():
    mean_acc_corr[ck] = [[cv(m, gt)] for m in mean_acc]

res = {"acc": {}, "loss": {}}
epochs = stats_list[0][0]
info_names = stats_list[0][1][0]["oneshot_acc"].keys()
for info_name in info_names:
    res["acc"][info_name] = list(
        zip(*[[epoch_stat["oneshot_acc"][info_name] for epoch_stat in seed_stat[1]] for seed_stat in stats_list]))
    res["loss"][info_name] = list(
        zip(*[[epoch_stat["oneshot_loss"][info_name] for epoch_stat in seed_stat[1]] for seed_stat in stats_list]))

res["mean_acc"] = mean_acc_corr

fig = plt.figure(figsize=[16.,6.])
gs = gridspec.GridSpec(nrows=1, ncols=2)#, width_ratios=[3]*num_cols + [2])

def plot_trend_to_ax(ax, name):
    accs = np.array(res[name]["oneshot average"])
    linear = np.array(res[name]["linear"])
    spearmanrs = np.array(res[name]["spearmanr"])
    kds = np.array(res[name]["kd"])
    p_at_top5 = np.array(res[name]["P@tbK"])[:, :, 3, 2]
    p_at_bot5 = np.array(res[name]["P@tbK"])[:, :, 3, 3]
#     br_at_5 = np.array(res[name]["BWR@K"])[:, :, 1, 2]
#     wr_at_5 = np.array(res[name]["BWR@K"])[:, :, 1, 4]

    labels = ["Oneshot average", "Linear correlation", "Spearman correlation", "Kendall's tau", "P@top5%", "P@bottom5%", "BR@5%", "WR@5%"]
    colors = ["#3A76AF", "#EF8536", "#519D3E", "#C53932", "#8D6AB8", "#8D6AB8"]
    markers = {
    "P@bottom5%": "^"
    }
    linestyles={"P@bottom5%": "--"}
    lines = []
    for i, (nums, color, label) in enumerate(zip([accs, linear, spearmanrs, kds, p_at_top5, p_at_bot5], colors, labels)):
        if name == "loss" and i == 0:
            ax2 = ax.twinx()
            ax2.tick_params(labelsize = 16)
            ax2.set_ylabel("Oneshot loss", fontsize=16)
            plot_ax = ax2
        else:
            plot_ax = ax
        avg, std = np.mean(nums, axis=-1), np.std(nums, axis=-1)
        lines.append(plot_ax.plot(epochs, avg, color=color, linewidth=2,
                             label=label,linestyle=linestyles.get(label, "-"),mew=0.5,marker=markers.get(label, "o"))[0])
        plot_ax.fill_between(epochs, avg-std, avg+std, facecolor=color, alpha=0.2)
    return lines

# ---- acc ----
ax = fig.add_subplot(gs[0, 0])
ax.tick_params(labelsize = 16)
plot_trend_to_ax(ax, "acc")
ax.set_title("Oneshot accuracy", fontsize=16)
ax.legend(fontsize=16)

# ---- loss ----
ax = fig.add_subplot(gs[0, 1])
ax.tick_params(labelsize = 16)
lines = plot_trend_to_ax(ax, "loss")
ax.set_title("Oneshot loss (negative)", fontsize=16)
labels=[l.get_label() for l in lines]
# ax.legend(lines,labels,fontsize=16)

# ---- show ----
plt.tight_layout()
plt.savefig("fig1_nb301.pdf")
#plt.show()
