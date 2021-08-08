#pylint: disable-all
import os
import subprocess
import multiprocessing
import argparse

GPUs = [0,1,2,3,4,5]
parser = argparse.ArgumentParser()
parser.add_argument("cmd_text_file")
parser.add_argument("--gpus", default=None)
args = parser.parse_args()
if args.gpus is not None:
    GPUs = [int(g) for g in args.gpus.split(",")]

with open(args.cmd_text_file, "r") as rf:
    cmds = rf.read().strip().split("\n")

num_processes = len(GPUs)
print("Num process: {}. Num exp: {}".format(num_processes, len(cmds)))

queue = multiprocessing.Queue(maxsize=num_processes)
def _worker(p_id, gpu_id, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        _, cmd = token
        print("Process #{}: CMD: {}".format(p_id, cmd))
        subprocess.check_call("CUDA_VISIBLE_DEVICES={} {}".format(gpu_id, cmd), shell=True)
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for i_cmd, cmd in enumerate(cmds):
    queue.put((i_cmd, cmd))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
