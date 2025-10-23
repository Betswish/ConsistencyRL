#!/bin/bash

# Step1: generate training data
# XCSQA supported languages:
# en zh de es fr it ja nl pl pt ru ar vi hi sw ur

all_langs=(en zh de es fr it ja nl pl pt ru ar vi hi sw ur)

cd data
# for SEED in 0 1 2; do
for SEED in 0; do
	# for ((i=0; i<${#all_langs[@]}-1; i++)); do
	for ((i=0; i<1; i++)); do
		for ((j=i+1; j<${#all_langs[@]}; j++)); do
			echo "Seed" $SEED "Generating training split with random completion for: " ${all_langs[$i]} ${all_langs[$j]}
			python sampling.py --dataset xcsqa --langs ${all_langs[$i]} ${all_langs[$j]} --seed $SEED --train_instance_num 800 --generate_instance_num 5600
		done
	done
done
cd ..