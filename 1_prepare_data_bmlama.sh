#!/bin/bash

# Step1: generate training data
# BMLAMA-17 supported languages:
# en fr nl es ru ja zh ko vi el hu he tr ca ar uk fa

all_langs=(en fr nl es ru ja zh ko vi el hu he tr ca ar uk fa)

cd data
# for SEED in 0 1 2; do
for SEED in 0; do
	# for ((i=0; i<${#all_langs[@]}-1; i++)); do
	for ((i=0; i<1; i++)); do
		for ((j=i+1; j<${#all_langs[@]}; j++)); do
			echo "Seed" $SEED "Generating training split with random completion for: " ${all_langs[$i]} ${all_langs[$j]}
			python sampling.py --dataset bmlama --langs ${all_langs[$i]} ${all_langs[$j]} --seed $SEED --train_instance_num 5000 --generate_instance_num 5000
		done
	done
done
cd ..