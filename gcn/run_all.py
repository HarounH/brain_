"""
Hyperparameter search for given architecture type.
"""
import os
import sys
import subprocess
import time
from joblib import Parallel, delayed
from sklearn.utils import check_random_state


MAX_GPU_ID = 4
optionals_ = ["--cuda"]
datasets_ = ["archi", "hcp", "la5c"]
base_cmd_ = ["python", "-m", "gcn.train_clf"]
base_output_dir_ = "/data/brain_/gcn/outputs"


def parse_log_file(logfile):
    train_accuracies = {}  # epoch -> acc
    train_loss = {} # epoch -> acc
    test_accuracies = {}  # epoch -> acc
    test_loss = {} # epoch -> acc
    with open(logfile, "r") as f:
        for line in f:
            if len(line) == 0 or line[0] != "[":
                continue
            # train
            if "train" in line:
                splits = line.split(" ")
                epoch = int(splits[1])
                losseqn = splits[4]
                loss = float(losseqn.split("=")[1])
                acceqn = splits[5]
                acc = float(acceqn.split("=")[1])
                train_loss[epoch] = loss
                train_accuracies[epoch] = acc
            # test
            if "val" in line or "test" in line:
                num = float(line.split(" ")[-1])
                if "contrast acc" in line:
                    test_accuracies[epoch] = num
                if "contrast ce" in line:
                    test_loss[epoch] = num
    return train_loss, train_accuracies, test_loss, test_accuracies


# Single run over multiple seeds
def single_seed(dataset, parameters_dict, optionals, seed, gpu_id):
    env_dict = dict(os.environ)
    if gpu_id == -1:
        optionals += ["-dp"]
    else:
        env_dict['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    mt = parameters_dict["arch"]  # Model type.
    batch_size = parameters_dict["batch_size"]
    lr = parameters_dict["lr"]
    epoch = parameters_dict["epoch"]
    weight_decay = parameters_dict["weight_decay"]
    parameters = ["-b", batch_size,
                  "-lr", lr,
                  "-e", epoch,
                  "--weight_decay", weight_decay,
                  ]
    subfolder = "b{}_e{}_lr{:.8f}_wd{:.8f}".format(
        batch_size,
        lr,
        epoch,
        weight_decay
    )
    if "nregions" in parameters_dict:
        parameters = parameters + ["-nr", parameters_dict["nregions"]]
        subfolder += "_nr{}".format(parameters_dict["nregions"])

    logfile = os.path.join(
        base_output_dir_,
        "logs",
        mt,
        dataset,
        subfolder,
        "{}.log.txt".format(seed),
    )
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    cmd = base_cmd_ + dataset + parameters + optionals + ["--seed", seed]
    print("> {}".format(" ".join(cmd)))
    print("std_out to {}".format(logfile))
    start_time = time.time()
    with open(logfile, "w") as f:
        subprocess.run(cmd, stdout=f, env=env_dict)
    print("Completed {} in {}s".format("CUDA_VISIBLE_DEVICES={}".format(gpu_id) + " ".join(cmd), time.time() - start_time))
    return logfile


# Multiple seeds for a single file
def single_parameters(dataset, parameters_dict, optionals, n_seeds=20):
    seeds = check_random_state(42).randint(0, 100000, size=n_seeds).tolist()
    gpu_ids = [i % MAX_GPU_ID for i in range(len(seeds))]
    logfiles = Parallel(n_jobs=MAX_GPU_ID, verbose=10)(
        delayed(single_seed)(
            dataset,
            parameters_dict,
            optionals,
            seed,
            gpu_id
        ) for (seed, gpu_id) in zip(seeds, gpu_ids)
    )
# Run everything!
model2parameters = {}
all_models = ['fgl0', 'rfgl0', 'fc', 'conv', 'cc']
all_lrs = [0.01, 0.001, 0.0001, 0.00001]
all_epochs = [1000]
all_batch_sizes = [16, 32, 64, 128]
all_weight_decays = [0.00001, 0.0001, 0.001, 0.01]
for k in all_models:
    model2parameters[k] = {
        'lr': all_lrs,
        'epoch': all_epochs,
        'batch_size': all_batch_sizes,
        'weight_decay': all_weight_decays,
    }
    if k == 'rfgl0':
        model2parameters[k]['nregions'] = [2, 4, 8, 16]  # Not sure if I can handle more.
for arch in model2parameters.keys():
    parameters_dict = {'arch': arch}
    # For each key in 
