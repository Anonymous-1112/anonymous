import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pickle

seeds = [20, 2020, 202020]
stats = []
for seed in seeds:
    with open("./fig_surgery/fig1_nb201_seed{}_stat.pkl".format(seed), "rb") as r_f:
        stats.append(pickle.load(r_f))

stats_list = [list(zip(*sorted([(int(os.path.basename(key).split(".")[0]), value) for key, value in stat.items()], key=lambda item: item[0]))) for stat in stats]

res = {"acc": {}, "loss": {}}
epochs = stats_list[0][0]
info_names = stats_list[0][1][0]["oneshot_acc"].keys()
for info_name in info_names:
    res["acc"][info_name] = list(zip(*[[epoch_stat["oneshot_acc"][info_name] for epoch_stat in seed_stat[1]] for seed_stat in stats_list]))
    res["loss"][info_name] = list(zip(*[[epoch_stat["oneshot_loss"][info_name] for epoch_stat in seed_stat[1]] for seed_stat in stats_list]))

fig = plt.figure(figsize=[16.,6.])
gs = gridspec.GridSpec(nrows=1, ncols=2)#, width_ratios=[3]*num_cols + [2])

def plot_trend_to_ax(res, ax, name, marker="o", linestyle="-", plot_epochs=None):
    if plot_epochs is None:
        plot_epochs = epochs
    accs = np.array(res[name]["oneshot average"])
    linear = np.array(res[name]["linear"])
    spearmanrs = np.array(res[name]["spearmanr"])
    kds = np.array(res[name]["kd"])
    if np.array(res[name]["P@tbK"]).ndim == 4:
        p_at_top5 = np.array(res[name]["P@tbK"])[:, :, 3, 2]
        p_at_bot5 = np.array(res[name]["P@tbK"])[:, :, 3, 3]
    else:
        p_at_top5 = np.array(res[name]["P@tbK"])[:, 3, 2]
        p_at_bot5 = np.array(res[name]["P@tbK"])[:, 3, 3]
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
        if nums.ndim == 2:
            avg, std = np.mean(nums, axis=-1), np.std(nums, axis=-1)
            lines.append(plot_ax.plot(plot_epochs, avg, color=color, linewidth=2,
                             label=label,linestyle=linestyles.get(label, linestyle),mew=0.5,
                                  marker=markers.get(label, marker))[0])
            plot_ax.fill_between(plot_epochs, avg-std, avg+std, facecolor=color, alpha=0.2)
        else:
            lines.append(plot_ax.plot(plot_epochs, nums, color=color, linewidth=2,
                             label=label,linestyle=linestyles.get(label, linestyle),mew=0.5,
                                  marker=markers.get(label, marker))[0])
    return lines

# ---- acc ----
ax = fig.add_subplot(gs[0, 0])
ax.tick_params(labelsize = 16)
plot_trend_to_ax(res, ax, "acc")
ax.set_title("Oneshot accuracy", fontsize=16)
ax.legend(fontsize=16)

# ---- loss ----
ax = fig.add_subplot(gs[0, 1])
ax.tick_params(labelsize = 16)
lines = plot_trend_to_ax(res, ax, "loss")
ax.set_title("Oneshot loss (negative)", fontsize=16)
labels=[l.get_label() for l in lines]
# ax.legend(lines,labels,fontsize=16)

# ---- show ----
plt.tight_layout()
plt.savefig("fig1_nb201.pdf")
