import subprocess
# models = ["fc", "bigconv", "bigcc", "regionfgl"] # wedgefgl"] # "complexwedgefgl"]
models = ['bigcc', 'bigconv']
#fracs = #[0.001, 0.005]  # [0.01, 0.25]  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
# fracs = []
# # dataset = "complexwedges50000"
# # dataset = "gradientwedges50000"
# # dataset = "iidxclustergrid50000"
# # dataset = "iidhclustergrid50000"
# # fracs = [0.01, 0.1, 0.25, 0.5, 0.8]
# testing_frac = {
#     'complexwedges10000': 0.5,
#     'complexwedges50000': 0.2,
#     'gradientwedges50000': 0.2,
#     'clustergrid50000': 0.2,
#     'iidxclustergrid50000': 0.2,
#     'iidhclustergrid50000': 0.2,
#     'bayesian16000': 0.2,
#     'blockybayesian16000': 0.2,
# }
# nepochs = 30
#
# dataset = "bayesian16000"
# fracs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.8]
# for frac in fracs:
#     for model in models:
#         print("Running {} {}".format(model, frac))
#         cmdstr = "python -m clevr.main {} {} --cuda -dp -ok 10 -maxe {} -df {} -tf {} -r {}{}".format(dataset, model, nepochs, testing_frac[dataset], frac, dataset, int(1000 * frac))
#         cmd = cmdstr.split(' ')
#         ret = subprocess.run(cmd)
#
# dataset = "blockybayesian16000"
# fracs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.8]
# for frac in fracs:
#     for model in models:
#         print("Running {} {}".format(model, frac))
#         cmdstr = "python -m clevr.main {} {} --cuda -dp -ok 10 -maxe {} -df {} -tf {} -r {}{}".format(dataset, model, nepochs, testing_frac[dataset], frac, dataset, int(1000 * frac))
#         cmd = cmdstr.split(' ')
#         ret = subprocess.run(cmd)
#



### New Run
testing_frac = {
    'complexwedges10000': 0.5,
    'complexwedges50000': 0.2,
    'gradientwedges50000': 0.2,
    'clustergrid50000': 0.2,
    'iidxclustergrid50000': 0.2,
    'iidhclustergrid50000': 0.2,
    'bayesian16000': 0.4,
    'blockybayesian16000': 0.4,
}
nepochs = 25

dataset = "iidxclustergrid50000"
fracs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6, 0.8]
for frac in fracs:
    for model in models:
        print("Running {} {}".format(model, frac))
        cmdstr = "python -m clevr.main {} {} --cuda -dp -ok 10 -maxe {} -df {} -tf {} -r {}_df{}_{}".format(dataset, model, nepochs, testing_frac[dataset], frac, dataset, testing_frac[dataset], int(1000 * frac))
        cmd = cmdstr.split(' ')
        ret = subprocess.run(cmd)

# dataset = "bayesian16000"
# fracs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6]
# for frac in fracs:
#     for model in models:
#         print("Running {} {}".format(model, frac))
#         cmdstr = "python -m clevr.main {} {} --cuda -dp -ok 10 -maxe {} -df {} -tf {} -r {}_df{}_{}".format(dataset, model, nepochs, testing_frac[dataset], frac, dataset, testing_frac[dataset], int(1000 * frac))
#         cmd = cmdstr.split(' ')
#         ret = subprocess.run(cmd)
#
# dataset = "blockybayesian16000"
# fracs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6]
# for frac in fracs:
#     for model in models:
#         print("Running {} {}".format(model, frac))
#         cmdstr = "python -m clevr.main {} {} --cuda -dp -ok 10 -maxe {} -df {} -tf {} -r {}_df{}_{}".format(dataset, model, nepochs, testing_frac[dataset], frac, dataset, testing_frac[dataset], int(1000 * frac))
#         cmd = cmdstr.split(' ')
#         ret = subprocess.run(cmd)
