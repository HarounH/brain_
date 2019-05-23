# Runs all FGL experiments.

# Camcan
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 30 -ct fc -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 30 -ct lin -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 30 -ct lin_ -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 50 -ct conv -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 50 -ct conv_ -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 50 -ct cc -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 20 -ct smallerfgl_213_sum_tree -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 50 -ct redfc -ok 10 -r camcan0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct redfc -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 50 -ct redfc -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 50 -ct redfc -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 50 -ct redfc -ok 10 -r hcp0

## Channel size ablation
# python -m gcn.multi_run camcan --cuda -dp -mine 2 -maxe 10 -ct c1smallerfgl_213_sum_tree -ok 4 -r camcan0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 2 -maxe 10 -ct c1smallerfgl_213_sum_tree -ok 4 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 2 -maxe 10 -ct c1smallerfgl_213_sum_tree -ok 4 -r la5c0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 2 -maxe 10 -ct c1smallerfgl_213_sum_tree -ok 4 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 2 -maxe 10 -ct c1smallerfgl_213_sum_tree -ok 4 -r hcp0

# python -m gcn.multi_run camcan --cuda -dp -mine 2 -maxe 10 -ct c2smallerfgl_213_sum_tree -ok 4 -r camcan0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 2 -maxe 10 -ct c2smallerfgl_213_sum_tree -ok 4 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 2 -maxe 10 -ct c2smallerfgl_213_sum_tree -ok 4 -r la5c0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 2 -maxe 10 -ct c2smallerfgl_213_sum_tree -ok 4 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 2 -maxe 10 -ct c2smallerfgl_213_sum_tree -ok 4 -r hcp0

# python -m gcn.multi_run camcan --cuda -dp -mine 2 -maxe 10 -ct c4smallerfgl_213_sum_tree -ok 4 -r camcan0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 2 -maxe 10 -ct c4smallerfgl_213_sum_tree -ok 4 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 2 -maxe 10 -ct c4smallerfgl_213_sum_tree -ok 4 -r la5c0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 2 -maxe 10 -ct c4smallerfgl_213_sum_tree -ok 4 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 2 -maxe 10 -ct c4smallerfgl_213_sum_tree -ok 4 -r hcp0

python -m gcn.multi_run camcan --cuda -dp -mine 2 -maxe 10 -ct c8smallerfgl_213_sum_tree -ok 4 -r camcan0 --not_lazy
python -m gcn.multi_run archi --cuda -dp -mine 2 -maxe 10 -ct c8smallerfgl_213_sum_tree -ok 4 -r archi0 --not_lazy
python -m gcn.multi_run la5c --cuda -dp -mine 2 -maxe 10 -ct c8smallerfgl_213_sum_tree -ok 4 -r la5c0 --not_lazy
python -m gcn.multi_run brainomics --cuda -dp -mine 2 -maxe 10 -ct c8smallerfgl_213_sum_tree -ok 4 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 2 -maxe 10 -ct c8smallerfgl_213_sum_tree -ok 4 -r hcp0

# python -m gcn.multi_run camcan --cuda -dp -mine 2 -maxe 10 -ct c16smallerfgl_213_sum_tree -ok 4 -r camcan0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 2 -maxe 10 -ct c16smallerfgl_213_sum_tree -ok 4 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 2 -maxe 10 -ct c16smallerfgl_213_sum_tree -ok 4 -r la5c0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 2 -maxe 10 -ct c16smallerfgl_213_sum_tree -ok 4 -r brainomics0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 2 -maxe 10 -ct c16smallerfgl_213_sum_tree -ok 4 -r hcp0

# Ablation start
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 20 -ct residualsmallerfgl_213_sum_tree -ok 10 -r camcan1 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct residualsmallerfgl_213_sum_tree -ok 10 -r archi1 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct residualsmallerfgl_213_sum_tree -ok 10 -r la5c1 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 20 -ct residualsmallerfgl_213_sum_tree -ok 10 -r brainomics1 --not_lazy
#
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 20 -ct 1smallerfgl_213_sum_tree -ok 10 -r camcan1 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct 1smallerfgl_213_sum_tree -ok 10 -r archi1 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct 1smallerfgl_213_sum_tree -ok 10 -r la5c1 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 20 -ct 1smallerfgl_213_sum_tree -ok 10 -r brainomics1 --not_lazy
#
# python -m gcn.multi_run camcan --cuda -dp -mine 10 -maxe 20 -ct 2smallerfgl_213_sum_tree -ok 10 -r camcan1 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct 2smallerfgl_213_sum_tree -ok 10 -r archi1 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct 2smallerfgl_213_sum_tree -ok 10 -r la5c1 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 20 -ct 2smallerfgl_213_sum_tree -ok 10 -r brainomics1 --not_lazy
#
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 20 -ct 1smallerfgl_213_sum_tree -ok 10 -r hcp1
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 20 -ct 2smallerfgl_213_sum_tree -ok 10 -r hcp1
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 20 -ct residualsmallerfgl_213_sum_tree -ok 10 -r hcp1
# Ablation end

# brainomics
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 30 -ct fc -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 30 -ct lin -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 30 -ct lin_ -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 50 -ct conv -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 50 -ct conv_ -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 50 -ct cc -ok 10 -r brainomics0 --not_lazy
# python -m gcn.multi_run brainomics --cuda -dp -mine 10 -maxe 20 -ct smallerfgl_213_sum_tree -ok 10 -r brainomics0 --not_lazy

# Archi
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct fc -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct lin -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct lin_ -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct conv -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct conv_ -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 50 -ct cc -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct fgl_132_sum_tree -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct fgl_213_sum_tree -ok 10 -r archi0
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct rfgl -ok 10 -r archi0

# La5c
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 40 -ct fc -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 40 -ct lin -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 40 -ct lin_ -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct conv -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct cc -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct fgl_132_sum_tree -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct fgl_213_sum_tree -ok 10 -r la5c0
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct rfgl -ok 10 -r la5c0

# HCP
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 40 -ct fc -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 40 -ct lin -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 40 -ct lin_ -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 100 -ct conv -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 100 -ct cc -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct fgl_132_sum_tree -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct fgl_213_sum_tree -ok 10 -r hcp0
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct rfgl -ok 10 -r hcp0

# Residual FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 20 -ct rfgl_213_sum_tree -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct rfgl_213_sum_tree -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct rfgl_213_sum_tree -ok 10 -r hcp0

# Small FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 15 -ct smallfgl_213_sum_tree -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct smallfgl_213_sum_tree -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 20 -ct smallfgl_213_sum_tree -ok 10 -r hcp0

# Smaller FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 15 -ct smallerfgl_213_sum_tree -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct smallerfgl_213_sum_tree -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct smallerfgl_213_sum_tree -ok 10 -r hcp0

# EqSmaller FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 30 -ct eqsmallerfgl_213_sum_tree -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 30 -ct eqsmallerfgl_213_sum_tree -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 30 -ct eqsmallerfgl_213_sum_tree -ok 10 -r hcp0


# Smaller2FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 15 -ct smaller2fgl_213_sum_tree -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct smaller2fgl_213_sum_tree -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct smaller2fgl_213_sum_tree -ok 10 -r hcp0

# Random FGL
# python -m gcn.multi_run archi --cuda -dp -mine 10 -maxe 15 -ct randomfgl_213_sum_packed1.0 -ok 10 -r archi0 --not_lazy
# python -m gcn.multi_run la5c --cuda -dp -mine 10 -maxe 20 -ct randomfgl_213_sum_packed1.0 -ok 10 -r la5c0 --not_lazy
# python -m gcn.multi_run hcp --cuda -dp -mine 10 -maxe 15 -ct randomfgl_213_sum_packed1.0 -ok 10 -r hcp0
