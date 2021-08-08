Dear reviewers, this is the cleaned code for submitting "Evaluating Efficient Performance Estimators of Neural Architectures".

First, one should make sure that the internet to PyPI/github works, since there are many libs to install (due to the nature of our work)

1. First, it is recommend to create a new virtual env: `conda create -n mynas_clean_env python==3.7.6 pip && conda activate mynas_clean_env`. If you do not want setup a new env, at least make sure you have `pip` installed. We recommend use python version 3.7.6, since a dependent lib [`pycls`](https://github.com/facebookresearch/pycls) needs 3.7.6 to work.
2. Then, run `source setup_env.sh`. After seeing "env setuped", one can run `mynas --help`, and it should work.
    * When running `mynas --help`, there might be THREE warnings saying nasbench / nas_201_api / nasbench301 libs are not installed (shown as follows). Just ignore them. Some dependencies would need to be installed later when running OS training / evaluation (See point 4)
    ```
    08/08 06:59:01 PM btc              WARNING: Cannot import module nasbench: No module named 'nasbench'
    Should install the NASBench 101 package following https://github.com/google-research/nasbench
    08/08 06:59:01 PM btc              WARNING: Cannot import module nasbench_201: No module named 'nas_201_api'
    Should install the NASBench 201 package following https://github.com/D-X-Y/NAS-Bench-201
    08/08 06:59:01 PM btc              WARNING: Cannot import module nasbench_301: No module named 'nasbench301'
    Should install the NASBench 301 package to use NB301Evaluator (gt perfs) following https://github.com/automl/nasbench301. However, if only supernet training / arch evaluation are needed, there is no need to install it.
    ```
3. To plot figures using our checkpoints and logs, check `plots/README.md`.
4. To run supernet training / oneshot evaluation / zeroshot evaluation, check `instructions/README.md`. All instructions for OS training/evaluation and ZS evaluation on nb101/nb201/nb301/nds ResNet/nds ResNeXt search spaces are provided.
    * For example, you can run `cd instructions/nb201` and then `bash run_nb201_training_example.sh` to run OS training on nb201. Run the 3-rd `mynas derive` command in `run_nb201_training_example.sh` to conduct OS evaluation using a provided supernet checkpoint.

Some supernet checkpoints and training logs are provided under `results` for the ease of verification. The full checkpoints and logs will all be released in the future in a separate web drive.
