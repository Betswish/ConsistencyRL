import json
import argparse
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def compute_acc_baseline(
    seed: int = 0,
    dataset: str = 'bmlama',
    instance_num: int = 5000,
    mname: str | None = None,
    langs: list | None = None,
) -> float:
    """Compute consistency between two languages."""

    train_data = f"seed{seed}_sample{instance_num}_{dataset}_baseline"
    post_mname = f"{mname.replace('/', '-')}"

    try:
        acc_en = json.load(open(f'./outputs/{train_data}/{post_mname}/en_Accuracy.json', 'r'))
        acc_en = round(acc_en*100, 2)
    except Exception as e:
        acc_en = "-"  # Default value if error occurs

    acc_list = []
    for lang in ['en'] + langs:
        try:
            acc = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_Accuracy.json', 'r'))
            acc = round(acc*100, 2)
            acc_list.append(acc)
        except Exception as e:
            acc_list.append("-")  # Append "-" for languages that fail to compute consistency

    print(f"{mname.split('/')[-1]} & {' & '.join([str(acc) for acc in acc_list])} \\\\")
    return acc_list

def compute_acc(
    seed: int = 0,
    dataset: str = 'bmlama',
    instance_num: int = 5000,
    mname: str | None = None,
    beta: float | None = None,
    langs: list | None = None,
) -> float:
    train_data = f"seed{seed}_sample{instance_num}_{dataset}"

    """Compute consistency between two languages."""
    base_acc_non = compute_acc_baseline(
        seed=seed,
        dataset=dataset,
        instance_num=instance_num,
        mname=mname,
        langs=langs,
    )

    acc_list = []
    for lang in langs:
        post_mname = f"{mname.replace('/', '-')}_{'-'.join(['en', lang])}_1.0-1.0_{beta}"

        try:
            acc_en = json.load(open(f'./outputs/{train_data}/{post_mname}/en_Accuracy.json', 'r'))
            acc_non = json.load(open(f'./outputs/{train_data}/{post_mname}/{lang}_Accuracy.json', 'r'))

            acc_en = round(acc_en*100, 2)
            acc_non = round(acc_non*100, 2)

            acc_list.append((acc_en, acc_non))
        except Exception as e:
            acc_list.append(("-", "-"))  # Append "-" for languages that fail to compute consistency

    acc_list_en, acc_list_non = zip(*acc_list)
    acc_en = [x for x in acc_list_en if x != "-"]
    if len(acc_en) > 0:
        acc_en = np.mean(acc_en)
    else:
        acc_en = "-"
    acc_list_non = [acc_en] + list(acc_list_non)
    print(f"+ DCO & {' & '.join([f'${'+' if c > bc else ''}{c-bc:.2f}$' if c != '-' else '-' for c, bc in zip(acc_list_non, base_acc_non)])} \\\\")
    print()
    return acc_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed for data generation')
    parser.add_argument('--dataset', type=str, default='bmlama', help='dataset name')
    parser.add_argument('--instance_num', type=int, default=5000, help='number of instances')
    parser.add_argument('--mname', type=str, default='Qwen/Qwen2.5-3B', help='model name')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value')

    args = parser.parse_args()
    seed = args.seed
    dataset = args.dataset
    instance_num = args.instance_num
    mname = args.mname
    beta = args.beta
    
    if dataset == 'bmlama':
        langs = ['fr', 'nl', 'es', 'ru', 'ja', 'zh', 'ko', 'vi', 'el', 'hu', 'he', 'tr', 'ca', 'ar', 'uk', 'fa']
    elif dataset == 'mmmlu':
        langs = ['ar', 'de', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh', 'bn']
    elif dataset == 'xcsqa':
        langs = ['zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'pl', 'pt', 'ru', 'ar', 'vi', 'hi', 'sw', 'ur']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Langs & {' & '.join(['en'] + langs)} \\\\")

    CLC = compute_acc(
        seed=seed, dataset=dataset, langs=langs,
        instance_num=instance_num, mname=mname,
        beta=beta
    )
