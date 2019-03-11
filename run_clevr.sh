#!/bin/bash
# ComplexWedges with large test
declare -a models=("fc" "conv" "cc" "complexwedgefgl")
declare -a fracs=(0.1 0.2 0.3 0.4 0.5)
for model in "${models[@]}"
do
  for frac in "${fracs[@]}"
  do
    echo "complexwedges10000$frac"
  done
done
python -m clevr.main complexwedges10000 conv --cuda -dp -ok 10 -maxe 40 -df 0.5 -tf 0.2 -r complexwedges20
python -m clevr.main complexwedges10000 conv --cuda -dp -ok 10 -maxe 40 -df 0.5 -tf 0.2 -r complexwedges20
python -m clevr.main complexwedges10000 conv --cuda -dp -ok 10 -maxe 40 -df 0.5 -tf 0.2 -r complexwedges20
python -m clevr.main complexwedges10000 conv --cuda -dp -ok 10 -maxe 40 -df 0.5 -tf 0.2 -r complexwedges20

# NotSoClevr
# python -m clevr.main notso fc --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.2 -r notso20
# python -m clevr.main notso fc --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.4 -r notso40
# python -m clevr.main notso fc --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.5 -r notso50
# python -m clevr.main notso fc --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.75 -r notso75

# python -m clevr.main notso conv --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.2 -r notso20
# python -m clevr.main notso conv --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.4 -r notso40
# python -m clevr.main notso conv --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.5 -r notso50
# python -m clevr.main notso conv --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.75 -r notso75

# python -m clevr.main notso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r notso20
# python -m clevr.main notso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r notso40
# python -m clevr.main notso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r notso50
# python -m clevr.main notso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r notso75

# python -m clevr.main notso quadfgl --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.2 -r notso20
# python -m clevr.main notso quadfgl --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.4 -r notso40
# python -m clevr.main notso quadfgl --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.5 -r notso50
# python -m clevr.main notso quadfgl --cuda -dp -ok 10 -maxe 10 -df 0.25 -tf 0.75 -r notso75

# wedges10000
# python -m clevr.main wedges10000 fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.2 -r wedges1000020
# python -m clevr.main wedges10000 fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.4 -r wedges1000040
# python -m clevr.main wedges10000 fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.5 -r wedges1000050
# python -m clevr.main wedges10000 fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.75 -r wedges1000075
#
#
# python -m clevr.main wedges10000 conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r wedges1000020
# python -m clevr.main wedges10000 conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r wedges1000040
# python -m clevr.main wedges10000 conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r wedges1000050
# python -m clevr.main wedges10000 conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r wedges1000075
#
#
# python -m clevr.main wedges10000 cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r wedges1000020
# python -m clevr.main wedges10000 cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r wedges1000040
# python -m clevr.main wedges10000 cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r wedges1000050
# python -m clevr.main wedges10000 cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r wedges1000075
#
#
# python -m clevr.main wedges10000 wedgefgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r wedges1000020
# python -m clevr.main wedges10000 wedgefgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r wedges1000040
# python -m clevr.main wedges10000 wedgefgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r wedges1000050
# python -m clevr.main wedges10000 wedgefgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r wedges1000075


# # Somewhat
# python -m clevr.main somewhat fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.2 -r somewhat20
# python -m clevr.main somewhat fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.4 -r somewhat40
# python -m clevr.main somewhat fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.5 -r somewhat50
# python -m clevr.main somewhat fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.75 -r somewhat75
#
#
# python -m clevr.main somewhat conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r somewhat20
# python -m clevr.main somewhat conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r somewhat40
# python -m clevr.main somewhat conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r somewhat50
# python -m clevr.main somewhat conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r somewhat75
#
#
# python -m clevr.main somewhat cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r somewhat20
# python -m clevr.main somewhat cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r somewhat40
# python -m clevr.main somewhat cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r somewhat50
# python -m clevr.main somewhat cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r somewhat75
#
#
# python -m clevr.main somewhat quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r somewhat20
# python -m clevr.main somewhat quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r somewhat40
# python -m clevr.main somewhat quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r somewhat50
# python -m clevr.main somewhat quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r somewhat75
#
#
#
# # BigNotSo
# python -m clevr.main bignotso fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.2 -r bignotso20
# python -m clevr.main bignotso fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.4 -r bignotso40
# python -m clevr.main bignotso fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.5 -r bignotso50
# python -m clevr.main bignotso fc --cuda -dp -ok 10 -maxe 50 -df 0.25 -tf 0.75 -r bignotso75
#
#
# python -m clevr.main bignotso conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r bignotso20
# python -m clevr.main bignotso conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r bignotso40
# python -m clevr.main bignotso conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r bignotso50
# python -m clevr.main bignotso conv --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r bignotso75
#
#
# python -m clevr.main bignotso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r bignotso20
# python -m clevr.main bignotso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r bignotso40
# python -m clevr.main bignotso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r bignotso50
# python -m clevr.main bignotso cc --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r bignotso75
#
#
# python -m clevr.main bignotso quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.2 -r bignotso20
# python -m clevr.main bignotso quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.4 -r bignotso40
# python -m clevr.main bignotso quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.5 -r bignotso50
# python -m clevr.main bignotso quadfgl --cuda -dp -ok 10 -maxe 30 -df 0.25 -tf 0.75 -r bignotso75
