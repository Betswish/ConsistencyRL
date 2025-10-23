#!/bin/bash

# Step2: post-train the LM with RL
# XCSQA supported languages:
# en zh de es fr it ja nl pl pt ru ar vi hi sw ur

all_langs=(en zh de es fr it ja nl pl pt ru ar vi hi sw ur)

models=(
	"Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"

    "google/gemma-3-4b-pt"
	"google/gemma-3-12b-pt"

	"Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"

	"CohereLabs/aya-expanse-8b"

	"meta-llama/Llama-3.1-8B"
	"meta-llama/Llama-3.2-3B"
)

seeds=(
	0
	# 1
	# 2
)

for SEED in "${seeds[@]}"; do
	for pair in "${pairs[@]}"; do
		for mname in "${models[@]}"; do
			# for ((i=0; i<${#all_langs[@]}-1; i++))
			for ((i=0; i<1; i++)); do
				for ((j=i+1; j<${#all_langs[@]}; j++)); do
					echo "Post-training " $mname " with RL on split: " ${all_langs[$i]} ${all_langs[$j]}
					python train_direction.py --seed $SEED --dataset xcsqa --instance_num 5600 --mname $mname --languages ${all_langs[$i]} ${all_langs[$j]}
				done
			done
		done
	done
done

