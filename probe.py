import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import numpy as np
import shutil
import argparse
import random
import csv
import datasets

from tqdm import tqdm
import torch
from vllm import LLM as VLLM
from vllm import SamplingParams
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_dotenv(dotenv_path='./.env')
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

cache_dir = os.getenv("CACHE_DIR", "~/.cache")
cache_dir = os.path.expanduser(cache_dir)

api = HfApi()
hf_username = api.whoami(token=hf_token)["name"]

def success_repo(hub_model_id):
    try:    
        files = api.list_repo_files(repo_id=hub_model_id, repo_type="model")
        if 'tokenizer.json' in files: return True
        else: return False
    except Exception as e:
        return False
    
def predict_mask(model, answer_cand, prompt, mname):
    answer_pred_probs = dict()
    
    prompt_batch = []
    for answer in answer_cand:
        answer_cand_probs = []
        prompt_new = prompt.replace("<mask>", answer)
        # Fix the issue that Bloom Tokenizer will not automatically add the BOS token
        if "bloom" in mname: prompt_new = "<s>" + prompt_new
        prompt_batch.append(prompt_new)

    sampling_params = SamplingParams(
        max_tokens=1,  # must be >= 1
        temperature=0.0,
        logprobs=1,
        prompt_logprobs=0,
    )

    outputs = model.generate(prompt_batch, sampling_params, use_tqdm=False)

    for i, output in enumerate(outputs):
        # Get logprobs of prompt tokens only
        prompt_logprobs = output.prompt_logprobs

        # Compute negative log-likelihood (ignoring any None entries like BOS)
        nll = [list(lp.values())[0].logprob for lp in prompt_logprobs if lp is not None]
        avg_perplexity = -np.mean(nll)
        answer_pred_probs[answer_cand[i]] = avg_perplexity

    # return {cand: perplexity}, the lower the better
    return answer_pred_probs

# def probe(seed, mname):
def probe(
    seed: int = 0,
    dataset: str | None = None,
    beta: float = 1.0,
    train_instance_num: int = 5000,
    instance_num: int = 5000,
    mname: str | None = None,
    languages: list[str] | None = None,
) -> None:
    
    train_data = f"seed{seed}_sample{instance_num}_{dataset}"
    post_mname = f"{mname.replace('/', '-')}_{'-'.join(languages)}_1.0-1.0_{beta}"
    
    modelid = f"{train_data}_{post_mname}"
    hub_model_id = f"{hf_username}/{modelid}"


    print("Testing", hub_model_id)
    print(f"Number of GPU: {torch.cuda.device_count()}")

    save_path = f'./outputs/{train_data}/{post_mname}'

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Skip if the probing results already exist
    if os.path.exists(f'{save_path}/{languages[1]}_Accuracy.json') and os.path.exists(f'{save_path}/{languages[1]}_Accuracy.json'): 
        print("Probing results already exist.")
        return
    if os.path.exists(f'{save_path}/{languages[0]}_Accuracy.json') and os.path.exists(f'{save_path}/{languages[0]}_Accuracy.json'): 
        print(f"Probing results for {languages[0]} already exist.")
        languages = languages[1:]

    extra_kw = {"download_dir": cache_dir}
    model = VLLM(
        model=hub_model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        max_num_seqs=100,
        seed=2024,
        disable_custom_all_reduce=True,
        **extra_kw
    )

    raw_data = dict()
    if "mmmlu" in dataset.lower():
        data_name = "JRQi/MMMLU"
        dname = "mmmlu"
    elif "xcsqa" in dataset.lower():
        data_name = "JRQi/XCSQA"
        dname = "xcsqa"
    elif "bmlama" in dataset.lower():
        data_name = "JRQi/BMLAMA17"
        dname = "bmlama"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    for lang in languages:
        raw_data[lang] = datasets.load_dataset(data_name, lang, split="test").to_list()

    random.seed(seed)
    train_ids = random.sample(list(range(len(raw_data[languages[0]]))), train_instance_num)
    test_ids = list(set(range(len(raw_data[languages[0]]))) - set(train_ids))
    
    for lang in languages:
        data = [raw_data[lang][i] for i in test_ids]
        accuracy = 0

        # For saving probing results
        all_gold_indices = []
        all_ranked_indices = []

        for i, d in enumerate(tqdm(data)):
            prompt = d['Prompt'].strip()
            gold_ans_list = d['Ans']
            answer_cand = d['Candidate Ans']
            
            answer_pred_probs = predict_mask(model, answer_cand, prompt, mname)
            # {'Naples': 5.40697877407074, 'Rome': 5.137976503372192, ..., 'Mecca': 5.60733792998574}
            raw_candidates = list(answer_pred_probs.keys())
            # ['Naples', 'Rome', 'Mecca', 'Chicago', 'Armenia', 'London', 'Como', 'Scotland', 'Istanbul', 'Madrid']

            sorted_probs = sorted(answer_pred_probs.items(), key=lambda x: x[1], reverse=False)
            # [('Madrid', 4.722643804550171), ..., ('Como', 5.786579585075378)]
            ranked_candidates = [x[0] for x in sorted_probs]
            # ['Madrid', 'London', 'Rome', 'Chicago', 'Naples', 'Istanbul', 'Scotland', 'Armenia', 'Mecca', 'Como']

            # get the indices of all answers of gold_ans_list in raw_candidates
            
            gold_indices = [raw_candidates.index(ans) for ans in gold_ans_list]
            # get the indices of all candidates of ranked_candidates in raw_candidates
            ranked_indices = [raw_candidates.index(ans) for ans in ranked_candidates]
            # [9]
            # [9, 5, 1, 3, 0, 8, 7, 4, 2, 6]

            all_gold_indices.append(gold_indices)
            all_ranked_indices.append(ranked_indices)

            accuracy += sum(1 for gold_ans in gold_ans_list if gold_ans in ranked_candidates[:len(gold_ans_list)]) / len(gold_ans_list)            

        # Saving probing results to files
        with open(f'{save_path}/{lang}_GoldIndices.json', 'w') as f:
            json.dump(all_gold_indices, f)

        with open(f'{save_path}/{lang}_RankedIndices.json', 'w') as f:
            json.dump(all_ranked_indices, f)

        with open(f'{save_path}/{lang}_Accuracy.json', 'w') as f:
            json.dump(accuracy / len(data), f)

        # Print probing accuracy
        print(f'Probing Accuracy {lang}: {accuracy / len(data)}')
        print('====')
        
    # model_cache_dir = os.path.join(cache_dir, "models--" + hub_model_id.replace("/", "--"))
    # print(model_cache_dir)
    # if os.path.exists(model_cache_dir): shutil.rmtree(model_cache_dir)
    # model_cache_dir_hub = os.path.join(cache_dir, "hub/models--" + hub_model_id.replace("/", "--"))
    # if os.path.exists(model_cache_dir_hub): shutil.rmtree(model_cache_dir_hub)
    if os.path.exists(cache_dir): shutil.rmtree(cache_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed for data generation')
    parser.add_argument('--dataset', type=str, default='bmlama', help='dataset name')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value')
    parser.add_argument('--train_instance_num', type=int, default=5000, help='number of training instances')
    parser.add_argument('--instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--mname', type=str, default='meta-llama/Llama-3.2-3B', help='model name')
    parser.add_argument('--languages', nargs='+', default=['en', 'fr'], help='languages')

    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    beta = args.beta
    train_instance_num = args.train_instance_num
    instance_num = args.instance_num
    mname = args.mname
    languages = args.languages

    probe(
        seed=seed, dataset=dataset, beta=beta, train_instance_num=train_instance_num,
        instance_num=instance_num, mname=mname, languages=languages
    )