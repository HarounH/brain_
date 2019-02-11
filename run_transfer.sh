# Everything.
# With dropout
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct lin -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 10000 -ct conv -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 10000 -ct cc -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct lin_ -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 10000 -ct conv_ -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 10000 -ct cc_ -ok 10 -r run0_drp -drp
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct smallerfgl_213_sum_tree -ok 10 -r run0_drp -drp

python -m transfer.multi_run archi camcan la5c brainomics hcp --cuda -dp -maxb 7500 -ct smallerfgl_213_sum_tree -ok 10 -drp -df 0.5 -r arthur

# Without dropout
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 5000 -ct lin -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct conv -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct cc -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 5000 -ct lin_ -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct conv_ -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 7500 -ct cc_ -ok 10 -r run0
# python -m transfer.multi_run archi la5c hcp --cuda -dp -maxb 5000 -ct smallerfgl_213_sum_tree -ok 10 -r run0
