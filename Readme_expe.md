# Run the experiments

First you should run the biases_calculation_huggingfacehub_PPL.py script to perturb the data and calculate the biases and probas using the task finetuned model:

```bash
python biases_calculation_huggingfacehub_PPL.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--list_countries France United_Kingdom Ireland Spain Germany Italy Morocco \
India Canada Australia New_Zealand United_States South_Africa \
Portugal Hungary Poland Turkey \
--n_duplicates 50 \
--model_name cardiffnlp/twitter-xlm-roberta-base-sentiment
```

Then you should run the PPL_Positivity.py script to calculate the PPL over the base model:
```bash
python PPL_Positivity.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--base_model_name cardiffnlp/twitter-xlm-roberta-base
```

In order to calculate the global-level correlation between perplexity and classes outputs on raw sentences (creating Table 2 of the Paper):
```bash
python generate_table2.py \
--input_data_file one_language_data.tsv \
--list_model_name_PPL cardiffnlp/twitter-roberta-base cardiffnlp/twitter-xlm-roberta-base \
--list_model_name_task cardiffnlp/twitter-xlm-roberta-base-sentiment cardiffnlp/twitter-roberta-base-hate
```

In order to calculate the local-level correlation between perplexity and classes outputs on perturbated sentences (creating Table 3 of the Paper):
```bash
python generate_table3.py \
--data_tsv labeled_data.tsv \
--model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
--base_model_name cardiffnlp/twitter-xlm-roberta-base
```
