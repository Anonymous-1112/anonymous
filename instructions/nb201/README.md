Nasbench-201
--------

## Convenient Example Runs

### Oneshot Training / Evaluation
* Run `bash run_nb201_training_example.sh` to run supernet training.
* After supernet training, one can run architecture evaluation using the saved supernet checkpoints (see the second `mynas derive` command in the script).
    * We provide several nb201 supernet checkpoints under `results/nb201`. One can modify the script to load these supernet checkpoint for arch evaluation. All checkpoints will be available later.

### Zeroshot Evaluation
```
python ../run_zero_shot.py --result-dir ./results/nb201/zeroshot/ --batch-size 128 --num-batch 5 zeroshot.yaml nb201_assets/non-isom.yaml
```

* To only run several zeroshot indicators, modify the `objective_cfg.perf_names` list in `zeroshot.yaml`. (Some zeroshot indicators are slow, e.g., jacob_cov)
* To run ZS estimation using only one random seed, modify line 77 of `../run_zeroshot.py`.

### Plotting
The results of running OS/ZS evaluations are under ...

## More Detailed Documentation

### Prepare

1. Install the NB201 package (API v1.0) following instructions [here](https://github.com/D-X-Y/NAS-Bench-201).
2. If you want to load the GT performances to use `NasBench201Evaluator`, you should download the data file `NAS-Bench-201-v1_0-e61699.pth`, and put it under the path `$MYNAS_HOME/data/nasbench-201/` (without MYNAS_HOME explicitly overrode, this path is `$HOME/mynas/data/nasbench-201/`). However, if you only want to run supernet training / architecture evaluation, there is no need to download this.
3. Several other data files are used in the evaluation script and the de-isomorphism sampling, these files (`non-isom.txt`, `non-isom5.txt`, `iso_dict.yaml`, `iso_dict.txt`, `deiso_dict.txt`) are under `instructions/nb201/nb201_assets/`, should be put under the path `$MYNAS_HOME/data/nasbench-201/`.

### Train one-shot supernets

`mynas search --gpu 0 --seed [random seed] --save-every 50 --train-dir results/oneshot-example/ examples/research/surgery/nb201/deiso_plateaulr.yaml`

One can modify the configurations in `deiso_plateaulr.yaml`, for example, 1) To use `S` architecture samples in every supernet update, change `evaluator_cfg.mepa_samples` to `S`; 2) To adjust wheher or not to use de-isomorphism sampling, add `deiso: true` and the architecture list file in the controller component cfg, as follows

```
controller_type: nasbench-201-rs
controller_cfg:
  rollout_type: nasbench-201
  deiso: true
  mode: eval
```

`examples/research/surgery/nb201/run_supernet_training.sh` is a helper script to run the previous `mynas search` command, and can be run with `bash examples/research/surgery/nb201/run_supernet_training.sh <cfg_file.yaml> <seed>`.

Also, to run multiple supernet training processes using multiple different configurations in a batched manner (on multiple GPUs), check `examples/research/surgery/run_supernet_training.py`.

### Derive architectures using one-shot supernets

```
mynas derive --load results/oneshot-example/1000/ --out-file results/oneshot-example/derive_results.yaml --gpu 0 -n 6466 --test --seed [random seed] --runtime-save examples/research/surgery/nb201/deiso_derive.yaml
```

The `--runtime-save` option is optional, and it enables `mynas derive` to continue from a previously interrupted derive process.

To run multiple `derive` processes using multiple different checkpoints in a batched manner (on multiple GPUs), check `examples/research/surgery/run_derive_ckpts.py`.

### Get the evaluation results

`python examples/research/surgery/evaluation.py results/oneshot-example/derive_results.yaml --type deiso`

