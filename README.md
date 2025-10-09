# (Building in progress) OPTIMIZING LANGUAGE MODELS FOR CROSSLINGUAL KNOWLEDGE CONSISTENCY

<div align="center">

Authors (_* Equal contribution_):  
[Tianyu Liu*](https://rycolab.io/authors/tianyu/) • [Jirui Qi*](https://betswish.github.io/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/)  
[Mrinmaya Sachan](https://rycolab.io/authors/mrinmaya/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/) • [Ryan Cotterell](https://rycolab.io/authors/ryan/)

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
bash 1_prepare_data.sh
```

If you want to generate instances only with negative answers
```bash
bash 1_prepare_data_false.sh
```

## (2) Train your model

## (3) Test and evaluation

