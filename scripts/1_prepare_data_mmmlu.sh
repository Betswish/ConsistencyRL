#!/bin/bash

# Step1: generate training data
# MMMLU supported languages:
# en ar bn de es fr hi id it ja ko pt sw yo zh

all_langs=(en ar bn de es fr hi id it ja ko pt sw yo zh)

cd data
# for SEED in 0 1 2; do
for SEED in 0; do
	# for ((i=0; i<${#all_langs[@]}-1; i++)); do
	for ((i=0; i<1; i++)); do
		for ((j=i+1; j<${#all_langs[@]}; j++)); do
			echo "Seed" $SEED "Generating training split with random completion for: " ${all_langs[$i]} ${all_langs[$j]}
			python sampling.py --dataset mmmlu --langs ${all_langs[$i]} ${all_langs[$j]} --seed $SEED --train_instance_num 5000 --generate_instance_num 5000
		done
	done
done
cd ..