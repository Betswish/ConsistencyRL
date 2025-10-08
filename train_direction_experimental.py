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

load_dotenv(dotenv_path='./.env')
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

api = HfApi()


def success_repo(hub_model_id):
    try:    
        files = api.list_repo_files(repo_id=hub_model_id, repo_type="model")
        if 'tokenizer.json' in files: return True
        else: return False
    except Exception as e:
        return False
    
def train_direction(
    seed: int = 0,
    dataset: str | None = None,
    instance_num: int = 5000,
    learning_rate: float = 1e-5,
    beta: float = 1.0,
    sft_baseline: bool = False,
    constrained_logp: bool = False,
    num_train_epochs: int = 1,
    mname: str | None = None,
    languages: list[str] | None = None,
    lang1_learning_strength: float | None = None,
    lang2_learning_strength: float | None = None,
    use_false_examples: bool = False,
    precompute_ref_log_probs: bool = False,
    do_logging: bool = False,
) -> None:
    
    # Load the model and tokenizer
    if "gemma" in mname.lower():
        model = AutoModelForCausalLM.from_pretrained(
            mname,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mname,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(mname)
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        print(f"Using {tokenizer.eos_token} for padding")
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Using {tokenizer.pad_token} for padding")

    print(languages)
    print(type(languages))

    if not use_false_examples:
        # Load the training dataset
        # if precompute_ref_log_probs:
        #     train_dataset = Dataset.from_list(json.load(open(f"data/seed{seed}_sample{instance_num}_{dataset}/{mname.replace('/', '-')}/{'-'.join(languages)}.json")))
        # else:
        train_dataset = Dataset.from_list(json.load(open(f"data/seed{seed}_sample{instance_num}_{dataset}/{'-'.join(languages)}.json")))
        hub_model_id = f"seed{seed}_sample{instance_num}_{dataset}_{mname.replace('/', '-')}_{'-'.join(languages)}_{lang1_learning_strength}-{lang2_learning_strength}_{beta}"
    else:
        # Load the training dataset with false examples
        train_dataset = Dataset.from_list(json.load(open(f"data/seed{seed}_sample{instance_num}_{dataset}_false/{'-'.join(languages)}.json")))
        hub_model_id = f"seed{seed}_sample{instance_num}_{dataset}_false_{mname.replace('/', '-')}_{'-'.join(languages)}_{lang1_learning_strength}-{lang2_learning_strength}_{beta}"
    if sft_baseline:
        hub_model_id = "_".join(hub_model_id.split("_")[:-1]) + "_sft"
    print(f"Training {hub_model_id} on {languages} with beta {beta}")

    training_args = SPOConfig(
        learning_rate=learning_rate,
        beta=beta,
        constrained_logp=constrained_logp,
        output_dir=f"/cluster/work/cotterell/tianyu/spo-align-ckpt/{hub_model_id}",
        bf16=True,
        precompute_ref_log_probs=precompute_ref_log_probs,
        sft_baseline=sft_baseline,
        lang1_learning_strength=lang1_learning_strength,
        lang2_learning_strength=lang2_learning_strength,
        per_device_train_batch_size=4,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=1,
        save_strategy='no',
        # save_steps=10000,
        # logging_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        disable_tqdm=not do_logging,
        optim="adamw_torch_fused",
        push_to_hub=False,
        report_to=[],
        # report_to="wandb",
        # hub_private_repo=True, # a private repo
        # hub_model_id=f"JRQi/{hub_model_id}",
        # hub_strategy="end",
        # hub_token=hf_token,
    )
    if precompute_ref_log_probs or sft_baseline:
        ref_model = None
    else:
        if "gemma" in mname.lower():
            ref_model = AutoModelForCausalLM.from_pretrained(
                mname,
                attn_implementation="eager",
                dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                mname,
                dtype=torch.bfloat16,
                device_map="auto"
            )

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
    parser.add_argument('--constrained_logp', action='store_true', help='whether to constrain logp')
    parser.add_argument('--instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--mname', type=str, default='meta-llama/Llama-3.2-3B', help='model name')
    parser.add_argument('--languages', nargs='+', default=['en', 'fr'], help='languages')
    parser.add_argument('--lang1_learning_strength', type=float, default=1.0, help='learning strength for language 1')
    parser.add_argument('--lang2_learning_strength', type=float, default=1.0, help='learning strength for language 2')
    parser.add_argument('--use_false_examples', action='store_true', help='whether to use false examples')
    parser.add_argument('--do_logging', action='store_true', help='whether to do MLflow logging')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--precompute_ref_log_probs', action='store_true', help='whether to precompute reference log probabilities')
    parser.add_argument('--sft_baseline', action='store_true', help='whether to use SFT baseline')

    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    instance_num = args.instance_num
    mname = args.mname
    languages = args.languages
    lang1_learning_strength = args.lang1_learning_strength
    lang2_learning_strength = args.lang2_learning_strength
    use_false_examples = args.use_false_examples
    precompute_ref_log_probs = args.precompute_ref_log_probs
    do_logging = args.do_logging
    constrained_logp = args.constrained_logp
    num_train_epochs = args.num_train_epochs
    sft_baseline = args.sft_baseline

    train_direction(
        seed=seed,
        beta=args.beta,
        learning_rate=args.learning_rate,
        constrained_logp=constrained_logp,
        sft_baseline=sft_baseline,
        num_train_epochs=num_train_epochs,
        dataset=dataset,
        instance_num=instance_num,
        mname=mname,
        languages=languages,
        lang1_learning_strength=lang1_learning_strength,
        lang2_learning_strength=lang2_learning_strength,
        use_false_examples=use_false_examples,
        precompute_ref_log_probs=precompute_ref_log_probs,
        do_logging=do_logging
    )
