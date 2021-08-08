pip install -e git+https://github.com/D-X-Y/NAS-Bench-201@06e2d872bda93ed3b520e6ab7ee646aa2f7a0fcd#egg=nas_bench_201

MYNAS_HOME=${MYNAS_HOME:-$HOME/mynas/}
	  
mkdir -p $MYNAS_HOME/data/nasbench-201

if [[ ! -h "$MYNAS_HOME/data/nasbench-201/non-isom.txt" ]]; then
    ln -s `readlink -f nb201_assets/*` $MYNAS_HOME/data/nasbench-201
else
    echo "Soft links already exists. Skip linking."
fi

# 1. OS training
mynas search --gpu 0 --seed 20 --save-every 100 -q --train-dir ./results/nb201/oneshot_supernet_training/seed20 deiso_plateaulr.yaml

# 2. run arch evaluation using the 1000-epoch checkpoint
# mynas derive ./derive_config.yaml --load results/nb201/oneshot_supernet_training/seed20/1000/ --out-file results/nb201/oneshot_derive/1000.yaml --gpu 0 -n 6466 --test --seed 20 --runtime-save

# 3. run arch evaluation using the provided 1000-epoch checkpoint
mynas derive ./derive_config.yaml --load ../../results/nb201/oneshot_supernet_training/seed20/1000/ --out-file results/nb201/oneshot_derive/1000.yaml --gpu 0 -n 6466 --test --seed 20 --runtime-save
