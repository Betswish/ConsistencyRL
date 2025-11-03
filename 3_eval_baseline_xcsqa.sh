#!/bin/bash

# Step 3: probing the post-trained LM with all languages
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
	for mname in "${models[@]}"; do
		VLLM_CONFIGURE_LOGGING=0 python probe_baseline.py --seed $SEED --train_instance_num 800 --instance_num 5600 --dataset xcsqa --mname $mname --languages "${all_langs[@]}"
	done
done
