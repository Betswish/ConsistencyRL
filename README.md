# Aligning Factual Knowledge across Languages in Multilingual Language Models using Preference Optimization

# Load huggingface models

```bash
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("JRQi/seed0_sample10000_bmlama_google-gemma-3-1b-pt_en-fa_0.5-0.5") # For private model, include `token="hf_xxxxxx"`
model = AutoModelForCausalLM.from_pretrained("JRQi/seed0_sample10000_bmlama_google-gemma-3-1b-pt_en-fa_0.5-0.5") # For private model, include `token="hf_xxxxxx"`
```

# Environment Setup
## Install Dependencies from environment.yml
```bash
conda env create -f environment.yml
conda activate CRL
```

## Install trl
```bash
cd trl
pip install -e .
cd ..
```

# Data Preparation
## BMLAMA
For instance, to generate 5,000 instances for English and French languages, run the following command:
```bash
bash 1_prepare_data.sh
```

If you want to generate instances only with negative answers
```bash
bash 1_prepare_data_false.sh
```
