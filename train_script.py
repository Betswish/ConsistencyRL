# train_direction_experimental.py
from datasets import Dataset
from huggingface_hub.utils import disable_progress_bars
import json
import os, subprocess
import torch
from trl.trainer import SPOConfig, SPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from huggingface_hub import login, HfApi
from transformers.integrations import MLflowCallback
from dotenv import load_dotenv
import shutil

# Disable progress bars for cleaner output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["GIT_LFS_PROGRESS"] = "false"
os.environ["TQDM_DISABLE"] = "1"  # tqdm disable
disable_progress_bars()

cache_dir = os.getenv("CACHE_DIR", "~/.cache")
cache_dir = os.path.expanduser(cache_dir)

load_dotenv(dotenv_path='./.env')
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

api = HfApi()
hf_username = api.whoami(token=hf_token)["name"]
 
def train_direction(
    seed: int = 0,
    dataset: str | None = None,
    beta: float = 1.0,
    learning_rate: float = 1e-5,
    instance_num: int = 5000,
    mname: str | None = None,
    languages: list[str] | None = None,
    num_train_epochs: int = 1,
    precompute_ref_log_probs: bool = False,
    push_to_hub: bool = False,
    use_false_examples: bool = False,
    do_logging: bool = False,
) -> None:

    model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(mname, cache_dir=cache_dir)
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # print(f"Using {tokenizer.pad_token} for padding")

    print(languages)
    print(type(languages))

    if not use_false_examples:
        # Load the training dataset
        train_dataset = Dataset.from_list(json.load(open(f"data/seed{seed}_sample{instance_num}_{dataset}/{'-'.join(languages)}.json")))
        hub_model_id = f"seed{seed}_sample{instance_num}_{dataset}_{mname.replace('/', '-')}_{'-'.join(languages)}_{lang1_learning_strength}-{lang2_learning_strength}_{beta}"
    else:
        # Load the training dataset with false examples
        train_dataset = Dataset.from_list(json.load(open(f"data/seed{seed}_sample{instance_num}_{dataset}_false/{'-'.join(languages)}.json")))
        hub_model_id = f"seed{seed}_sample{instance_num}_{dataset}_false_{mname.replace('/', '-')}_{'-'.join(languages)}_{lang1_learning_strength}-{lang2_learning_strength}_{beta}"
    
    if dataset == 'bmlama':
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 1
        learning_rate = 1e-5
        num_train_epochs = 1
    elif dataset == 'mmmlu':
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 1
        learning_rate = 1e-5
        num_train_epochs = 1
    elif dataset == 'xcsqa':
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 1
        num_train_epochs = 1
        if "gemma-3-4b-pt" in mname.lower() or "gemma-3-12b-pt" in mname.lower() or "llama-3.1-8b" in mname.lower():
            learning_rate = 1e-6
        else:
            learning_rate = 1e-5
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    training_args = SPOConfig(
        learning_rate=learning_rate,
        beta=beta,
        output_dir=f"checkpoints/{hub_model_id}",
        bf16=True,
        precompute_ref_log_probs=precompute_ref_log_probs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        save_strategy='no',
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=not do_logging,
        optim="adamw_torch_fused",
        push_to_hub=push_to_hub,
        report_to=[],
        hub_model_id=f"{hf_username}/{hub_model_id}",
        hub_strategy="end",
        hub_token=hf_token,
    )

    if precompute_ref_log_probs:
        ref_model = None
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)

    trainer = SPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    
    # Remove the MLflow integration entirely
    if not do_logging:
        print("Disabling MLflow logging")
        trainer.remove_callback(MLflowCallback)  # cuts out MLflow logging

    trainer.train()
    trainer.save_model()

    print(f"Finished training {hub_model_id} on {languages} with beta {beta}")
    print('====')
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed for data generation')
    parser.add_argument('--dataset', type=str, default='bmlama', help='dataset name')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--mname', type=str, default='Qwen/Qwen2.5-7B', help='model name')
    parser.add_argument('--languages', nargs='+', default=['en', 'fr'], help='languages')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--precompute_ref_log_probs', action='store_true', help='whether to precompute reference log probabilities')
    parser.add_argument('--push_to_hub', action='store_true', help='whether to push the model to the Hugging Face Hub')
    parser.add_argument('--use_false_examples', action='store_true', help='whether to use false examples')
    parser.add_argument('--do_logging', action='store_true', help='whether to do MLflow logging')

    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    beta = args.beta
    learning_rate = args.learning_rate
    instance_num = args.instance_num
    mname = args.mname
    languages = args.languages
    num_train_epochs = args.num_train_epochs
    precompute_ref_log_probs = args.precompute_ref_log_probs
    push_to_hub = args.push_to_hub
    use_false_examples = args.use_false_examples
    do_logging = args.do_logging

    train_direction(
        seed=seed,
        dataset=dataset,
        beta=beta,
        learning_rate=learning_rate,
        instance_num=instance_num,
        mname=mname,
        languages=languages,
        num_train_epochs=num_train_epochs,
        precompute_ref_log_probs=precompute_ref_log_probs,
        push_to_hub=push_to_hub,
        use_false_examples=use_false_examples,
        do_logging=do_logging
    )
