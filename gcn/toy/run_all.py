import os
import sys
import subprocess
import time
from joblib import Parallel, delayed

datafiles = os.listdir("/data/gcn_toy/")
models = ["baseline", "fgl"]
ics = [1, 3, 5, 10, 128]
lrs = [0.1, 0.01, 0.001]
eps = [100, 300]
MAX_GPU_ID = 4
optionals = ["--cuda"]

base_cmd = ["python", "-m", "gcn.toy.clf"]
total_count = len(datafiles) * len(models) * len(ics) * len(lrs)
i = -1


def single_run(i, datafile, model_type, ic, lr, gpu_id):
    env_dict = dict(os.environ)
    env_dict['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    filename = "/data/brain_/gcn/toy/outputs/logs/{}_ic{}_lr{}/{}".format(model_type, ic, lr, datafile.replace(".npz", ".log.txt"))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("[{}/{}] Starting {} run on {}".format(i, total_count, model_type, datafile))
    cmd = base_cmd + [os.path.join("/data/gcn_toy/", datafile), model_type] + optionals + ["-ic", str(ic)] + ["-lr", str(lr)]
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
                i += 1
                gpu_id = i % MAX_GPU_ID
                arguments.append((i, datafile, model_type, ic, lr, gpu_id))

Parallel(n_jobs=4, verbose=10)(delayed(single_run)(*args) for args in arguments)
