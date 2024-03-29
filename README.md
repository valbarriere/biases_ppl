# Biases PPL

## Requirements

- Python version == 3.10

### CheckList library

This repository uses a custom version of the [CheckList library](https://github.com/marcotcr/checklist)

### Required packages

For managing Python packages, it is recommended to use either `virtualenv` or `conda`.

- For `virtualenv`, you can create a virtual environment using the following commands:

```bash
pip install virtualenv
virtualenv biases-ppl
source biases-ppl/bin/activate
```

- For `conda`, you can create a conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate biases-ppl
```

The required packages are listed in the `requirements.txt` file. You can install them using `pip`.

```bash
pip install -r requirements.txt
```

### Environment file

We use a `.env` file to store environment variables. Please copy the template from `public.env` to `.env` and fill in the required values.

```bash
cp public.env .env
```

### Entity Recognition Model

To run the experiments we use an entity recognition model. Download it using the Spacy library:

```bash
python -m spacy download xx_ent_wiki_sm
```

## Run the experiments

First you should run the biases_calculation_huggingfacehub_PPL.py script to perturb the data and calculate the biases and probas using the task finetuned model:

```bash
python biases_calculation_huggingfacehub_PPL.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--list_countries France United_Kingdom Ireland Spain Germany Italy Morocco \
India Canada Australia New_Zealand United_States South_Africa \
Portugal Hungary Poland Turkey \
--n_duplicates 50
```

Then you should run the PPL_Positivity.py script to calculate the PPL over the base model:
```bash
python PPL_Positivity.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--model_name cardiffnlp/twitter-xlm-roberta-base \
--list_gender male \
--verbose
```

Finally you can visualize the results using the given notebooks in the repo.