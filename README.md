# OPTIMIZING LANGUAGE MODELS FOR CROSSLINGUAL KNOWLEDGE CONSISTENCY

<div align="center">

Authors (_* Equal contribution_):  
[Tianyu Liu*](https://rycolab.io/authors/tianyu/) • [Jirui Qi*](https://betswish.github.io/) • [Mrinmaya Sachan](https://rycolab.io/authors/mrinmaya/)
[Ryan Cotterell](https://rycolab.io/authors/ryan/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/) 

</div>


## (0) Environment Setup
### Install Dependencies from environment.yml
```bash
conda env create -f environment.yml
conda activate CRL
```

### Install trl
```bash
cd trl
pip install -e .
cd ..
```

### Set Huggingface Token

Modify `.env` and add your Huggingface token.

## (1) Data Preparation

For instance, to generate 5,000 instances for English and French languages, run the following command:
```bash
bash scripts/1_prepare_data_mmmlu.sh
bash scripts/1_prepare_data_xcsqa.sh
bash scripts/1_prepare_data_bmlama.sh
```

## (2) Train Your Models

To train your model, firstly ensure that your `hf_token` is stored in `.env` like `export HF_TOKEN="hf_xxxxxx"` (Override with your Hugging Face token). Once confirmed, run the following script to train your model. It will automatically upload the post-trained model onto Huggingface and also save a copy locally in `checkpoints/`.

```bash
bash scripts/2_train_mmmlu.sh
bash scripts/2_train_xcsqa.sh
bash scripts/2_train_bmlama.sh
```

## (3) Evaluate Post-Trained Models with DCO

For evaluation, run the following scripts to get the probing results of the untrained models:

```bash
bash scripts/3_eval_baseline_mmmlu.sh
bash scripts/3_eval_baseline_xcsqa.sh
bash scripts/3_eval_baseline_bmlama.sh
```


Run the following scripts to get the probing results of the post-trained models with DCO:

```bash
bash scripts/3_eval_mmmlu.sh
bash scripts/3_eval_xcsqa.sh
bash scripts/3_eval_bmlama.sh
```

## (4) Show Changes in Consistency & Accuracy

All scripts are tested and should be ready to run by one-click:

Scripts for computing consistency:
```bash
bash scripts/4_compute_clc_mmmlu.sh
bash scripts/4_compute_clc_xcsqa.sh
bash scripts/4_compute_clc_bmlama.sh
```

Scripts for computing accuracy:
```bash
bash scripts/4_compute_acc_mmmlu.sh
bash scripts/4_compute_acc_xcsqa.sh
bash scripts/4_compute_acc_bmlama.sh
```