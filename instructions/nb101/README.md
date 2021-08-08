Nasbench-1shot1-Sub3
--------

### **Note**

Actually, the results reported in our paper are runned using the official codes provided by [NB101-1shot1](https://github.com/automl/nasbench-1shot1) (with a small bug fixed). Since we find that there are differences between our supernet implementation and theirs. However, you can still use our code to run NB101 OS training.


### Prepare
* To run try OS training, install [nasbench101](https://github.com/google-research/nasbench)
    * `pip install -e git+https://github.com/google-research/nasbench@b94247037ee470418a3e56dcb83814e9be83f3a8#egg=nasbench`
    * `pip install tensorflow-gpu==1.15.0`
* To reproduce the results in our paper, besides the previous two commands, one should do several extra things (can run `run_official_nb1shot.sh` for these things)
    * Install [nasbench101-1shot](https://github.com/automl/nasbench-1shot1) by: `git clone https://github.com/automl/nasbench-1shot1`, `cd nasbench-1shot1`, `export PYTHONPATH=$(pwd):$PYTHONPATH`. Some extra requirements should also be installed (See `run_official_nb1shot.sh`).
    * Use `nasbench_1shot1_official.py` file to replace `my_nas/btcs/nasbench_101.py`. (If you want to change it back, we have a backup at `nasbench_101.py` under this dir.)
    * Use the configuration file `oneshot_cfg_official.yaml` in `mynas search` and `mynas eval-arch` commands. The only difference between `oneshot_cfg_official.yaml` and `oneshot_cfg.yaml` is `weights_manager_type: nasbench-101-official` V.S. `weights_manager_type: nasbench-101`.


### Train one-shot supernets
`examples/research/surgery/nb101/oneshot_cfg.yaml` is the basic supernet-training configuration file on NB101-1shot. Run the following command:
```
mynas search --gpu 0 --seed 20 --save-every 100 --train-dir results/nb101/oneshot_supernet_training oneshot_cfg.yaml
```


### Derive architectures using one-shot supernets
Given a supernet/evaluator checkpoint, and a YAML file containing a list of architecture genotypes, one can run the following command to estimate the one-shot rewards of these architectures:
```
mynas eval-arch oneshot_cfg.yaml archs.yaml --load [supernet/evaluator checkpoint] --dump-rollouts results/nb101/oneshot_derive/eval_results.pkl --gpu 0 --seed 123
```


### Get the evaluation results

`python ../evaluation.py results/nb101/oneshot_derive/eval_results.pkl --type nb101`
