#!/bin/bash

# Step 3: probing the post-trained LM with all languages
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
		# for ((i=0; i<${#all_langs[@]}-1; i++))
		for ((i=0; i<1; i++)); do
			for ((j=i+1; j<${#all_langs[@]}; j++)); do
				VLLM_CONFIGURE_LOGGING=0 python probe.py --seed $SEED --dataset bmlama --train_instance_num 5000 --instance_num 5000 --mname $mname --languages ${all_langs[$i]} ${all_langs[$j]}
			done
		done
	done
done

