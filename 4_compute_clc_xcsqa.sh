#!/bin/bash

# Step 4: compute acc
# XCSQA supported languages:
# en zh de es fr it ja nl pl pt ru ar vi hi sw ur

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

metric="rankc"
# metric="top1"
# metric="onlycorrect"

for SEED in "${seeds[@]}"; do
    for mname in "${models[@]}"; do
        python consistency.py --seed $SEED --dataset xcsqa --instance_num 5600 --mname $mname --metric $metric
	done
done

