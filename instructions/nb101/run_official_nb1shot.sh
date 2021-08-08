if [[ ! -d nasbench-1shot1/ ]]; then
    git clone https://github.com/automl/nasbench-1shot1
fi

pushd nasbench-1shot1
export PYTHONPATH=$(pwd):$PYTHONPATH
popd

# extra requirements of nasbench-1shot1
pip install matplotlib
pip install seaborn
pip install ConfigSpace
pip install networkx

cp nasbench_1shot1_official.py ../../my_nas/btcs/nasbench_101.py

mynas search oneshot_cfg_official.yaml --gpu 0 --seed 20 --train-dir ./results/nb101/oneshot_training/seed20 --save-every 100
