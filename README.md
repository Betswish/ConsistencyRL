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

## (1) Data Preparation

For instance, to generate 5,000 instances for English and French languages, run the following command:
```bash
bash scripts/1_prepare_data_mmmlu.sh
bash scripts/1_prepare_data_xcsqa.sh
bash scripts/1_prepare_data_bmlama.sh
```

## (2) Train your model

To train your model, firstly write your `hf_token` into `.env`. Then, run the following script to train your model. It will automatically upload the trained model onto Huggingface and also save a copy locally in `checkpoints/`.

```bash
bash scripts/2_train_mmmlu.sh
bash scripts/2_train_xcsqa.sh
bash scripts/2_train_bmlama.sh
```


