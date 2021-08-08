#pylint: disable-all
import os


base_res_dir = "/home/eva_share_users/Anonymous/surgery/nb301/results/post_evo/on_evo_trained_supernet"
cmds = []
loads = [
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/new_single_evo/from0_seed20_every10/1000",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/new_single_evo/from0_seed20_every50/1000",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/new_single_evo/from0_seed20_every10/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/new_single_evo/from0_seed20_every50/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/pareto_evo/from0_seed20_every10/1000",
    # #"/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/pareto_evo/from0_seed20_every50/1000",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/pareto_evo/from0_seed20_every10/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/pareto_evo/from0_seed20_every50/200"
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/pareto_evo_every10_halfrandom/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/pareto_evo_every50_halfrandom/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every10_halfrandom/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every10_halfrandom_warmup100/200",
    # "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every50_halfrandom/200",
    "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/pareto_evo_every10_halfrandom/1000",
    "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/pareto_evo_every50_halfrandom/1000",
    "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every10_halfrandom/1000",
    "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every10_halfrandom_warmup100/1000",
    "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/results_supernet_training_correct/half_random_eval_sample/new_single_evo/single_evo_every50_halfrandom/1000"
]

common_path = os.path.commonpath(loads)
res_dirs = [os.path.join(base_res_dir,
                         os.path.relpath(load_dir, common_path))
            for load_dir in loads]

for load_dir, train_dir in zip(loads, res_dirs):
    cp_load_dir = os.path.join(os.path.dirname(load_dir), "{}_evaluatoronly".format(os.path.basename(load_dir)))
    os.makedirs(cp_load_dir, exist_ok=True)
    dest = os.path.join(cp_load_dir, "evaluator")
    print(dest, os.path.exists(dest))
    if not os.path.exists(dest):
        os.unlink(dest)
        os.symlink(os.path.join(load_dir, "evaluator"), dest)

    for save_every, dir_name, cfg_file in [
            (100, "tournament", "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/evo/post_evo/post_tournament_evo.yaml")
        ]:
        log_file = os.path.join(train_dir, "batch_call.log")
        os.makedirs(train_dir, exist_ok=True)
        cmd = "mynas search {} --gpu 0 --seed 20 --load {} --save-every 2500 --save-controller-every {} --train-dir {} >{} 2>&1".format(cfg_file, cp_load_dir, save_every, train_dir, log_file)
        cmds.append(cmd)



with open("halfrandom_post_evo_on_evo_trained_cmds_1000.txt", "w") as wf:
    wf.write("\n".join(cmds))


# seeds = [20, 2020, 202020]
# epochs = [200, 400, 600, 800, 1000]
# cmds = []
# base_res_dir = "/home/eva_share_users/Anonymous/surgery/nb301/results/post_evo"

# for seed in seeds:
#     for epoch in epochs:
#         for save_every, dir_name, cfg_file in [
#                 (1, "cars", "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/evo/post_evo/post_cars_evo.yaml"),
#                 (100, "tournament", "/home/eva_share_users/Anonymous/surgery/nb301/controller_cfgs/evo/post_evo/post_tournament_evo.yaml")
#         ]:
#             load = ("/home/eva_share_users/Anonymous/surgery/nb301/results/tobe_derived_ckpts"
#                     "/bs256_seed{}/{}").format(seed, epoch)
#             train_dir = os.path.join(base_res_dir, dir_name, "seed{}".format(seed), str(epoch))
#             log_file = os.path.join(train_dir, "batch_call.log")
#             os.makedirs(train_dir, exist_ok=True)
#             cmd = "mynas search {} --gpu 0 --seed {} --load {} --save-controller-every {} --train-dir {} >{} 2>&1".format(cfg_file, seed, load, save_every, train_dir, log_file)
#             cmds.append(cmd)

# with open("post_evo_on_oneshot_trained_cmds.txt", "w") as wf:
#     wf.write("\n".join(cmds))
