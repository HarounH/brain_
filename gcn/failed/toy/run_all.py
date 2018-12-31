import os
import sys
import subprocess
import time

datafiles = os.listdir("/data/gcn_toy/")
models = ["baseline", "fgl"]
optionals = ["--cuda"]

base_cmd = ["python", "-m", "gcn.toy.clf"]
total_count = len(datafiles) * len(models)
i = -1
for datafile in datafiles:
    for model_type in models:
        i += 1
        filename = "/data/brain_/gcn/toy/outputs/logs/{}/{}".format(model_type, datafile.replace(".npz", ".log.txt"))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print("[{}/{}] Starting {} run on {}".format(i, total_count, model_type, datafile))
        cmd = base_cmd + [os.path.join("/data/gcn_toy/", datafile), model_type] + optionals
        print("$$ {}".format(" ".join(cmd)))
        start_time = time.time()
        with open(filename, "w") as f:
            subprocess.run(cmd, stdout=f)
        print("Completed in {}s".format(time.time() - start_time))
