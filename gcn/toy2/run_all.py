import os
import sys
import subprocess
import time
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

datafiles = os.listdir("/data/gcn_toy2/")
models = ["fgl"]  # ["baseline", "fgl"]
# ics = [1, 4, 16, 32, 128]
# lrs = [0.1, 0.001, 0.0001, 0.00001]
# seeds = check_random_state(42).randint(0, 100000, size=20).tolist()

ics = [1, 4, 16, 32]  # , 128]
lrs = [0.1, 0.001]  # , 0.0001, 0.00001]
seeds = check_random_state(42).randint(0, 100000, size=4).tolist()

MIN_GPU_ID = 2
GPU_COUNT = 2
optionals = ["--cuda"]

base_cmd = ["python", "-m", "gcn.toy2.clf"]
total_count = len(datafiles) * len(models) * len(ics) * len(lrs) * len(seeds)
i = -1


def single_run(i, datafile, model_type, ic, lr, gpu_id, seed):
    env_dict = dict(os.environ)
    env_dict['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    filename = "/data/brain_/gcn/toy2/outputs/logs/{}_ic{}_lr{:.8f}_seed{}/{}".format(model_type, ic, lr, seed, datafile.replace(".npz", ".log.txt"))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("[{}/{}] Starting {} run on {}".format(i, total_count, model_type, datafile))
    cmd = base_cmd + [os.path.join("/data/gcn_toy2/", datafile), model_type] + optionals + ["-ic", str(ic)] + ["-lr", str(lr)] + ["-s", str(seed)]
    print("$$ {}".format(" ".join(cmd)))
    start_time = time.time()
    with open(filename, "w") as f:
        subprocess.run(cmd, stdout=f, env=env_dict)
    print("Completed {} in {}s".format("CUDA_VISIBLE_DEVICES={}".format(gpu_id) + " ".join(cmd), time.time() - start_time))


arguments = []
for datafile in datafiles:
    for model_type in models:
        for ic in ics:
            for lr in lrs:
                for seed in seeds:
                    i += 1
                    gpu_id = MIN_GPU_ID + (i % GPU_COUNT)
                    arguments.append((i, datafile, model_type, ic, lr, gpu_id, seed))

Parallel(n_jobs=GPU_COUNT, verbose=10)(delayed(single_run)(*args) for args in arguments)
