# Runs all FGL experiments.

# Archi
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct fc -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct lin -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct lin_ -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct conv -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct conv_ -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct cc -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct fgl_132_sum_tree -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct fgl_213_sum_tree -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct rfgl -ok 10 -r archi0

# La5c
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 40 -ct fc -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 40 -ct lin -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct conv -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct cc -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct fgl_132_sum_tree -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct fgl_213_sum_tree -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct rfgl -ok 10 -r la5c0

# HCP
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 40 -ct fc -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 40 -ct lin -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 25 -ct conv -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 25 -ct cc -ok 10 -r hcp0
python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct fgl_132_sum_tree -ok 10 -r hcp0
python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct fgl_213_sum_tree -ok 10 -r hcp0
python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct rfgl -ok 10 -r hcp0

# Random FGL
python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 15 -ct randomfgl -ok 10 -r archi0
python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 15 -ct randomfgl -ok 10 -r la5c0
python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct randomfgl -ok 10 -r hcp0
