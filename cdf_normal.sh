#!/bin/bash
conda activate defi
cd /home/danielemdn/Documents/repositories/generative_AMM

num_epochs=50000
max_patience=2000
n_scoring_points=1000
network=simple
n_layers=2
batch_size=2048
optimizer=adam


echo "Running with n_layers=$n_layers, batch_size=$batch_size, dropout=$dropout, batch_norm=$batch_norm"
python normal_cdf_estimation.py --num_epochs $num_epochs --max_patience $max_patience --n_scoring_points $n_scoring_points --network $network --n_layers $n_layers --batch_size $batch_size --optimizer $optimizer $dropout $batch_norm

