#!/bin/bash

# It will create a logs directory if it doesn't exist and save the output of each run to a separate log file.
mkdir -p logs

seeds=(0 1 6 10)
for seed in "${seeds[@]}"; do
    python -u 09_lgbm.py --learning_rate 0.03 --k 5 --split_seed $seed | tee logs/09_lgbm_learning_rate${lr}_k5_seed${seed}.log
done
