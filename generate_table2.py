"""
Script to generate the tables of correlations between PPL and Probability outputs at the global-level.
It does not use the perturbated examples. 

ex: 
python generate_table2.py \
--input_data_file one_language_data.tsv \
--list_model_name_PPL cardiffnlp/twitter-roberta-base cardiffnlp/twitter-xlm-roberta-base \
--list_model_name_task cardiffnlp/twitter-xlm-roberta-base-sentiment cardiffnlp/twitter-roberta-base-hate

Allows to create the Table 2 of the ACL24 paper

Author: Anonymous_Submission 01/24
"""
import pandas as pd
import pickle as pkl
import argparse
import os

from biases_calculation_huggingfacehub_PPL import prepare_data_and_model_from_scratch
from scipy.special import softmax
from utils import create_input_array

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from tqdm import tqdm

from scipy.stats import pearsonr
from dotenv import load_dotenv


load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR", None)
PATH_DATA = os.getenv("PATH_DATA", None)

def score(model, tokenizer, sentence):
    """
    Version more optimized
    Code from internet: https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
    """ 
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return loss
    
def return_label2idx(modelFilePath):
    """
    Return label2idx
    """
    config = AutoConfig.from_pretrained(modelFilePath)
    label2idx = {v.lower():k for k,v in config.id2label.items()}

    # Because label2idx is false for these 2 models... 
    emotion_task = modelFilePath == "cardiffnlp_roberta-base-tweet-emotion"
    sentiment_en_task = modelFilePath == "cardiffnlp_roberta-base-tweet-sentiment"
    if emotion_task:
        # there is an error with the label set in the model on the API
        label2idx = {'joy': 1, 'optimism': 2, 'anger': 0, 'sadness': 3}
    elif sentiment_en_task:
        label2idx = {'negative': 0, 'neutral': 1, 'positive': 2}

    return label2idx
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_corpus", help="The path of the folders containing the data", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--input_data_file", type=str, default="one_language_data.tsv")
    parser.add_argument("--list_model_name_PPL", help="countries to test, todo", type=str, default=['cardiffnlp/twitter-xlm-roberta-base', 
        'cardiffnlp/twitter-roberta-base'], nargs='+')
    parser.add_argument("--list_model_name_task", help="countries to test, todo", type=str, default=['cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'cardiffnlp/twitter-roberta-base-hate'], nargs='+')
    parser.add_argument("--verbose", help="verbose", default=False, action='store_true')
    args = parser.parse_args()

    ######################## Init ########################  

    list_model_name_PPL = args.list_model_name_PPL
    list_model_name_task = args.list_model_name_task
    input_data_file = args.input_data_file
    path_corpus = args.path_corpus

    input_data_path = os.path.join(path_corpus, input_data_file)
    df = pd.read_csv(input_data_path, sep='\t')
    list_text = {'Original' : df['tweet'].values}

    ######################## Proba ########################  

    path_dump_perturbed = os.path.join(path_corpus, f'Perturbed_{input_data_file}')

    for modelFilePath in list_model_name_task:

        model_name = modelFilePath.replace('/', '_')
        model, tokenizer, dict_lab, X_text, y = prepare_data_and_model_from_scratch(modelFilePath, input_data_file, path_corpus, unlabeled=True)

        for lan in list_text.keys():
            path_dump_proba_ini = ('/%s/ProbaIni_'%(model_name)).join(os.path.split(path_dump_perturbed)) + '_%s.pkl'%lan

            input_arrays = create_input_array(list_text[lan], tokenizer)
            try:
                proba_ini = softmax(model.predict(input_arrays).logits, axis=1)
            except:
                proba_ini = softmax(model.predict(input_arrays), axis=1)
            with open(path_dump_proba_ini, 'wb') as f:
                pkl.dump(proba_ini, f)

    ######################## PPL ########################  

    list_PPL = {model_name : {} for model_name in list_model_name_PPL}

    for model_name in list_model_name_PPL:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model.cuda()


        for lan, list_text_lan in list_text.items():
            list_PPL[model_name][lan] = []
            for sent in tqdm(list_text_lan):
                list_PPL[model_name][lan].append(score(model, tokenizer, sent).item())


    ######################## Pearson ########################     

    pearson_dic = {k : {} for k in list_model_name_task}
    probaini = {k : {} for k in list_model_name_task}

    for task in list_model_name_task:
        model_name = task.replace('/', '_')

        for lan in list_text.keys():
            path_dump_proba_ini = ('/%s/ProbaIni_'%(model_name)).join(os.path.split(path_dump_perturbed)) + '_%s.pkl'%lan
            with open(path_dump_proba_ini, 'rb') as f:        
                probaini[task][lan] = pkl.load(f)
            model_MLM_name = '-'.join(task.split('-')[:-1])

            label2idx = return_label2idx(task)
            for lab, idx_lab in label2idx.items(): 
                cor, _ = pearsonr(list_PPL[model_MLM_name][lan], probaini[task][lan][:,idx_lab])
                pearson_dic[task][(lan,lab)]  = cor

    df = [[pearson_dic[task][lan] for lan in pearson_dic[task].keys()] for task in probaini.keys()]
    df = pd.DataFrame(df, index=[task.replace('/', '_') for task in list_model_name_task])
    df.to_csv(path_corpus + '%s/Table2_Correlations_PPL_%s'%(model_name, input_data_file)+ '.tsv', sep='\t')