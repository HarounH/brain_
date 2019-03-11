from collections import defaultdict
import subprocess
models = []
dataset = []
fracs = []

single_datasets = ['camcan', 'brainomics', 'archi', 'la5c', 'hcp']


tf_datasets = ['camcan', 'brainomics', 'archi']
epoch_counts = {
    # 'fc': 30,
    # 'lin': 50,
    # 'lin_': 30,
    # 'conv': 50,
    # 'conv_': 50,
    # 'cc': 50,
    # 'smallerfgl_213_sum_tree': 20,
    'eqsmallerfgl_213_sum_tree': 20,
    # 'redfc': 30,
}
models = list(epoch_counts.keys())
not_lazy = defaultdict(lambda: False)
for x in ['archi', 'camcan', 'la5c', 'brainomics']:
    not_lazy[x] = True
training_fracs = [0.1, 0.25, 0.5, 0.7]

for dataset in single_datasets:
    for model in models:
        print('Running {} {}'.format(dataset, model))
        runcode = dataset + '0'
        cmdstr = "python -m gcn.multi_run {} --cuda -dp -mine 10 -maxe {} -ct {} -ok 10 -r {}{}".format(
            dataset,
            epoch_counts[model],
            model,
            runcode,
            ' --not_lazy' if not_lazy[model] else ''
        )
        cmd = cmdstr.split(' ')
        ret = subprocess.run(cmd)

for dataset in tf_datasets:
    for frac in training_fracs:
        for model in models:
            print("Running {} {} {}".format(dataset, model, frac))
            runcode = dataset + '_tf{}'.format(frac)
            cmdstr = "python -m gcn.multi_run {} --cuda -dp -mine 10 -maxe {} -ct {} -ok 10 -tf {} -r {}{}".format(
                dataset,
                epoch_counts[model],
                model,
                frac,
                runcode,
                ' --not_lazy' if not_lazy[model] else ''
            )
            cmd = cmdstr.split(' ')
            ret = subprocess.run(cmd)
