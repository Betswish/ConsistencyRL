#!/bin/bash

# Step 4: compute acc
# BMLAMA-17 supported languages:
# en fr nl es ru ja zh ko vi el hu he tr ca ar uk fa

all_langs=(en fr nl es ru ja zh ko vi el hu he tr ca ar uk fa)

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
    for mname in "${models[@]}"; do
		python accuracy.py --seed $SEED --dataset bmlama --instance_num 5000 --mname $mname
	done
done

