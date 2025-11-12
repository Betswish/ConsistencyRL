#!/bin/bash

# Step 3: probing the post-trained LM with all languages
# MMMLU supported languages:
# en ar bn de es fr hi id it ja ko pt sw yo zh

all_langs=(en ar bn de es fr hi id it ja ko pt sw yo zh)

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
		VLLM_CONFIGURE_LOGGING=0 python probe_baseline.py --seed $SEED --dataset mmmlu --mname $mname --languages "${all_langs[@]}"
	done
done
