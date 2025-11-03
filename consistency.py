import json
import argparse
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def top1(data_en, data_non):
    clc = 0
    for i in range(len(data_en)):
        if data_en[i][0] == data_non[i][0]:
            clc += 1
    clc /= len(data_en)

    return clc

def onlycorrect(data_en, data_non, gold):
    clc = 0
    total = 0
    for i in range(len(data_en)):
        if data_en[i][0] in gold[i] or data_non[i][0] in gold[i]:
            total += 1
            if data_en[i][0] == data_non[i][0]: clc += 1
    clc /= total
    return clc

def rankc(data_en, data_non):
    clc = 0

    for i in range(len(data_en)):
        # Calculate the weight based on the order of the elements
        weight = softmax([len(data_en[i])-k for k in range(len(data_en[i]))])

        for j in range(len(data_en[i])):
            set1 = set(data_en[i][:j+1])
            set2 = set(data_non[i][:j+1])

            overlap = set1.intersection(set2)
            clc += weight[j] * (len(overlap)/len(set1))
    clc /= len(data_en)
    return float(clc)


def compute_consis_baseline(
    seed: int = 0,
    dataset: str = 'bmlama',
    instance_num: int = 5000,
    mname: str | None = None,
    metric: str = 'rankc',
) -> float:
    """Compute consistency between two languages."""

    train_data = f"seed{seed}_sample{instance_num}_{dataset}_baseline"

    if dataset == 'bmlama':
        langs = ['fr', 'nl', 'es', 'ru', 'ja', 'zh', 'ko', 'vi', 'el', 'hu', 'he', 'tr', 'ca', 'ar', 'uk', 'fa']
    elif dataset == 'mmmlu':
        langs = ['ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh', 'bn']
    elif dataset == 'xcsqa':
        langs = ['zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'ar', 'vi', 'hi', 'sw', 'ur']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    CLC_list = []
    for lang in langs:
        post_mname = f"{mname.replace('/', '-')}"

        try:
            data_en = json.load(open(f'./outputs/{train_data}/{post_mname}/en_RankedIndices.json', 'r'))
            data_non = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_RankedIndices.json', 'r'))
            
            assert len(data_en) == len(data_non), "Data length mismatch between English and non-English datasets."

            if metric == 'top1':
                CLC = top1(data_en, data_non)
            elif metric == 'rankc':
                CLC = rankc(data_en, data_non)
            elif metric == 'onlycorrect':
                correct = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_GoldIndices.json', 'r'))
                CLC = onlycorrect(data_en, data_non, correct)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            CLC = round(CLC*100, 2)
            CLC_list.append(CLC)
        except Exception as e:
            CLC_list.append(-100.00) # Append -100 for languages that fail to compute consistency

    return CLC_list


def compute_consis(
    seed: int = 0,
    dataset: str = 'bmlama',
    instance_num: int = 5000,
    mname: str | None = None,
    beta: float | None = None,
    metric: str = 'top1',
) -> float:
    """Compute consistency between two languages."""
    base_CLC_list = compute_consis_baseline(
        seed=seed,
        dataset=dataset,
        instance_num=instance_num,
        mname=mname,
        metric=metric
    )
    if not use_false_examples:
        train_data = f"seed{seed}_sample{instance_num}_{dataset}"
    else:
        train_data = f"seed{seed}_sample{instance_num}_{dataset}_false"

    if dataset == 'bmlama':
        langs = ['fr', 'nl', 'es', 'ru', 'ja', 'zh', 'ko', 'vi', 'el', 'hu', 'he', 'tr', 'ca', 'ar', 'uk', 'fa']
    elif dataset == 'mmmlu':
        langs = ['ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh', 'bn']
    elif dataset == 'xcsqa':
        langs = ['zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'ar', 'vi', 'hi', 'sw', 'ur']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    CLC_list = []
    for lang in langs:
        post_mname = f"{mname.replace('/', '-')}_{'-'.join(['en', lang])}_1.0-1.0_{beta}"

        try:
            data_en = json.load(open(f'./outputs/{train_data}/{post_mname}/en_RankedIndices.json', 'r'))
            data_non = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_RankedIndices.json', 'r'))

            assert len(data_en) == len(data_non), "Data length mismatch between English and non-English datasets."

            if metric == 'top1':
                CLC = top1(data_en, data_non)
            elif metric == 'rankc':
                CLC = rankc(data_en, data_non)
            elif metric == 'onlycorrect':
                correct = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_GoldIndices.json', 'r'))
                CLC = onlycorrect(data_en, data_non, correct)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            CLC = round(CLC*100, 2)
            CLC_list.append(CLC)
        except Exception as e:
            CLC_list.append(-100.00) # Append -100 for languages that fail to compute consistency

    print(f"+ \methodname" + " & " + " & ".join([f"{'+' if c > bc else ''}{c-bc:.2f}" if c != -100.00 else "Fail" for c, bc in zip(CLC_list, base_CLC_list)]) + r" \\")
    return CLC_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed for data generation')
    parser.add_argument('--dataset', type=str, default='bmlama', help='dataset name')
    parser.add_argument('--instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--mname', type=str, default='Qwen/Qwen2.5-3B', help='model name')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value')
    parser.add_argument('--metric', type=str, default='rankc', help='metric for consistency. Select from top1 and rankc')

    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    instance_num = args.instance_num
    mname = args.mname
    metric = args.metric
    beta = args.beta

    CLC = compute_consis(
        seed=seed, dataset=dataset,
        instance_num=instance_num, mname=mname,
        beta=beta,
        metric=metric
    )
