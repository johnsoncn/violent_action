#!/usr/bin/env bash
# START VIOLENT ACTION ENV

set +v

#title violent_action

echo Hello! This is a batch to start violent_action jupyter.


# ACTIVATE ENV ********
#eval "$(conda shell.bash hook)"
source /home/administrator/miniconda3/bin/activate violent_action

# change dir to start pyff *****
cd "/home/administrator/Z/Algorithms/MOLA/NC/violent_action/"

# Start jupyter lab *******
jupyter lab

