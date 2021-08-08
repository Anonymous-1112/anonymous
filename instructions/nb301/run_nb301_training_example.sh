# mynas search --gpu 0 --seed 20 --save-every 100 -q --train-dir ./results/nb301/oneshot_supernet_training/seed20 config.yaml

# run derive using the 1500-epoch supernet ckpt
mynas eval-arch config.yaml nb301_assets/archs.yaml --load ../../results/nb301/oneshot_supernet_training/seed20/1500/ --dump-rollouts results/nb301/oneshot_eval_results.pkl --gpu 0 --seed 123
