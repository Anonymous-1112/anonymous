Nasbench-301
--------

### Train one-shot supernets

`examples/research/surgery/nb301/config.yaml` is the basic supernet-training configuration file on NB301. Run the following command to start a supernet-training process:

`````
mynas search --gpu 0 --seed 20 --save-every 100 -q --train-dir ./results/nb301/oneshot_supernet_training/seed20 config.yaml
`````

Also, to run multiple supernet training processes using multiple different configurations in a batched manner (on multiple GPUs), check `examples/research/surgery/run_supernet_training.py`.

### Eval-arch using one-shot supernets

Given a supernet/evaluator checkpoint, and a YAML file containing a list of architecture genotypes, one can run the following command to estimate the one-shot rewards of these architectures:

```
mynas eval-arch examples/research/surgery/nb301/config.yaml archs.yaml --load [supernet/evaluator checkpoint] --dump-rollouts results/nb301/eval_results.pkl --gpu 0 --seed 123
```

The architecture file `arch.yaml` used in our paper can be found under the dir `nb301_assets/`.

To run multiple `eval-arch` processes using multiple different checkpoints in a batched manner (on multiple GPUs), check `examples/research/surgery/run_derive_ckpts.py`.

### Get the evaluation results

One does not need to install the NB301 benchmark while running the previous commands. However, when calculating the evaluation results, the GT performances of architectures are needed, thus [the NB301 benchmark package](https://github.com/automl/nasbench301) needs to be installed.

After successful installation, run ``python examples/research/surgery/evaluation.py results/nb301/eval_results.pkl --type nb301`` to dump the evaluation results (e.g., Kendall's Tau, SpearmanR, P@top/bottomKs, B/WR@Ks) to `results/nb301/eval_results_statistics.pkl`.

### Run Zeroshot Evaluation
```
python ../run_zero_shot.py --result-dir ./results/nb301/zeroshot/ --batch-size 128 --num-batch 5 zeroshot.yaml nb301_assets/archs.yaml
```

* To only run several zeroshot indicators, modify the `objective_cfg.perf_names` list in `zeroshot.yaml`. (Some zeroshot indicators are slow, e.g., jacob_cov)
* To run ZS estimation using only one random seed, modify line 77 of `../run_zeroshot.py`.

