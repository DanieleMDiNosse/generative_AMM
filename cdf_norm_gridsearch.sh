#!/bin/bash
conda activate defi
cd /home/danielemdn/Documents/repositories/generative_AMM

num_epochs=50000
max_patience=1000
n_scoring_points=1000
optimizer=adam

for network in simple reverse_autoencoder
do
    for n_layers in 2 3 4 5
    do
        for batch_size in 1024 2048
        do
            for dropout in 0.0 0.1 0.2
            do
                for learning_rate in 0.0001 0.00001
                do
                    echo "Running with  network=$network n_layers=$n_layers, batch_size=$batch_size, dropout=$dropout"
                    python normal_cdf_estimation.py --num_epochs $num_epochs --max_patience $max_patience --n_scoring_points $n_scoring_points --network $network --n_layers $n_layers --batch_size $batch_size --optimizer $optimizer --dropout $dropout
                done
            done
        done
    done
done
