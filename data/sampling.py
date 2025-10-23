from tqdm import trange
import json
import os
import pathlib
from collections import defaultdict
import random
import argparse
import csv
import datasets


def sampling(seed, dataset, languages, train_instance_num=None, generate_instance_num=None, use_false_examples=False):
    assert train_instance_num > 0, "train_instance_num must be > 0"
    assert generate_instance_num % train_instance_num == 0 and generate_instance_num > 0, \
        "generate_instance_num must be a positive integer multiple of train_instance_num"

    data = dict()

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
        data[lang] = datasets.load_dataset(data_name, lang, split="test").to_list()

    random.seed(seed)
    train_ids = random.sample(list(range(len(data[languages[0]]))), train_instance_num)
    test_ids = list(set(range(len(data[languages[0]]))) - set(train_ids))

    # Keep original data for accessing by train_ids
    original_data = data.copy()
    
    # Calculate k = generate_instance_num / train_instance_num
    k = generate_instance_num // train_instance_num
    
    instances = []
    for iteration in trange(k):
        for train_id in train_ids:
            if len(languages) == 2:
                lang1, lang2 = languages
            else:
                lang1, lang2 = random.sample(languages, 2)
            
            # Use the specific train_id instead of random sampling
            random_query_1 = original_data[lang1][train_id]['Prompt'].strip()
            random_query_2 = original_data[lang2][train_id]['Prompt'].strip()

            ground_truth_1 = [original_data[lang1][train_id]["Ans"]]
            ground_truth_2 = [original_data[lang2][train_id]["Ans"]]

            answer_list_1 = eval(original_data[lang1][train_id]["Candidate Ans"])
            answer_list_2 = eval(original_data[lang2][train_id]["Candidate Ans"])

            # print(len(answer_list_2))

            if use_false_examples:
                # remove ground truth from the answer list
                answer_list_1 = [
                    ans for ans in answer_list_1 if ans not in ground_truth_1]
                answer_list_2 = [
                    ans for ans in answer_list_2 if ans not in ground_truth_2]

            if len(answer_list_1) < 2 or len(answer_list_2) < 2:
                continue

            random_chosen_id, random_rejected_id = random.sample(
                list(range(len(answer_list_1))), 2)

            random_chosen_1 = answer_list_1[random_chosen_id]
            random_chosen_2 = answer_list_2[random_chosen_id]

            random_rejected_1 = answer_list_1[random_rejected_id]
            random_rejected_2 = answer_list_2[random_rejected_id]

            instance = {
                "prompt_1": random_query_1,
                "chosen_1": random_chosen_1,
                "rejected_1": random_rejected_1,
                "prompt_2": random_query_2,
                "chosen_2": random_chosen_2,
                "rejected_2": random_rejected_2
            }
            instances.append(instance)


    if not use_false_examples:
        save_path = pathlib.Path(__file__).parent.parent / "data" / f"seed{seed}_sample{generate_instance_num}_{dname}" / f"{'-'.join(languages)}.json"
    else:
        save_path = pathlib.Path(__file__).parent.parent / "data" / f"seed{seed}_sample{generate_instance_num}_{dname}_false" / f"{'-'.join(languages)}.json"

    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent)

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(instances, f, ensure_ascii=True)
        """
        Example output:
        {
            "prompt_1": "The capital of France is",
            "chosen_1": " Paris.",
            "rejected_1": " Amsterdam.",
            "prompt_2": "法国的首都是",
            "chosen_2": "巴黎。",
            "rejected_2": "伦敦。",
        }
        """


# Example usage:
# python data/mmmlu.py --languages en fr --generate-instance-num 5000;
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default="mmmlu", help='dataset name')
    parser.add_argument('--langs', nargs='+', default=['en', 'fr'], help='languages')
    parser.add_argument('--train_instance_num', type=int, default=5000, help='number of training instances')
    parser.add_argument('--generate_instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--use_false_examples', action='store_true', help='whether to generate false examples')

    args = parser.parse_args()
    dataset = args.dataset
    languages = args.langs
    train_instance_num = args.train_instance_num
    generate_instance_num = args.generate_instance_num
    use_false_examples = args.use_false_examples
    seed = args.seed

    sampling(seed, dataset, languages, 
                 train_instance_num=train_instance_num,
                 generate_instance_num=generate_instance_num,
                 use_false_examples=use_false_examples)
