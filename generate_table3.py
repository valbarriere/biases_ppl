"""
Script to generate the tables of correlations between PPL and Probability outputs.
It uses the already-made pkl files of the PPL and the pkl files of the Probabilities.

ex: 
python generate_correlations.py \
--model_name cardiffnlp/twitter-roberta-base-offensive \
--model_name_PT cardiffnlp/twitter-roberta-base

Allows to create the Table 3 of the ACL24 paper

Author: Anonymous_Submission 01/24
"""
import os
import pickle as pkl
import numpy as np
from scipy.stats import pearsonr
import argparse 
import pandas as pd
from transformers import AutoConfig
from dotenv import load_dotenv


import warnings
warnings.filterwarnings("ignore")

load_dotenv()
PATH_DATA = os.getenv("PATH_DATA", None)


from sklearn.preprocessing import StandardScaler

def rescale(l):
    scaler = StandardScaler()
    return [kk for ll in scaler.fit_transform([[k] for k in l]) for kk in ll]

def main_local_level(args):

    name_file = args.data_tsv
    model_name = args.model_name.replace('/', '_')
    model_name_PT = args.model_name_PT.replace('/', '_')
    list_countries = args.list_countries
    list_gender = ['male'] if args.male_only else ['male', 'female']

    path_dump_perturbed = args.path_corpora + 'Perturbed_' + name_file
    path_dump_PPL = args.path_corpora + '%s/Perturbed_'%model_name_PT + name_file + '_PPL.pkl'

    with open(path_dump_perturbed, 'rb') as fp:
        perturbed_X_text = pkl.load(fp)

    with open(path_dump_PPL, 'rb') as f:
        list_PPL = pkl.load(f)

    proba = {ct: {} for ct in list_countries}
    for country in list_countries:
        for gender in list_gender:
            with open(args.path_corpora + '%s/Proba_Perturbed_%s_%s_%s.pkl'%(model_name, name_file, country, gender), 'rb') as f:
                proba[country][gender] = pkl.load(f)

    with open(args.path_corpora + '%s/ProbaIni_Perturbed_%s.pkl'%(model_name, name_file), 'rb') as f:
        proba_ini = pkl.load(f)

    n_samples = len(proba_ini)
    n_counterfactuals = len(proba[country][gender]) // n_samples
    proba_ini = np.repeat(proba_ini, n_counterfactuals, axis=0)
    PPL_ini = np.repeat(list_PPL['Original'], n_counterfactuals)
    list_PPL_without_offset = {ct : {gender: list_PPL[ct][gender] - PPL_ini for gender in list_gender} for ct in list_countries}
    
    # set label2idx
    config = AutoConfig.from_pretrained(args.model_name)
    label2idx = {v.lower():k for k,v in config.id2label.items()}

    # Because label2idx is false for these 2 models... 
    emotion_task = model_name == "cardiffnlp_roberta-base-tweet-emotion"
    sentiment_en_task = model_name == "cardiffnlp_roberta-base-tweet-sentiment"
    if emotion_task:
        # there is an error with the label set in the model on the API
        label2idx = {'joy': 1, 'optimism': 2, 'anger': 0, 'sadness': 3}
    elif sentiment_en_task:
        label2idx = {'negative': 0, 'neutral': 1, 'positive': 2}

    proba_counter = { k : {ct : {gender: proba[ct][gender][:,l] for gender in list_gender} for ct in list_countries} for k,l in label2idx.items()}
    proba_counter_without_offset = {v: {ct : {gender: proba_counter[v][ct][gender] - proba_ini[:,label2idx[v]] for gender in list_gender} for ct in list_countries} for v in proba_counter.keys()}
    
    list_proba = list(label2idx.keys())
    
    list_df = [('proba', 'country', 'gender', 'correlation')]
    for proba_string in list_proba:
        for country in list_countries:
            for gender in list_gender:
                cor, _ = pearsonr(proba_counter_without_offset[proba_string][country][gender], 
                    list_PPL_without_offset[country][gender])
                if args.verbose: print(proba_string, country, gender, np.round(100*cor, 2))
                list_df.append((proba_string, country, gender, np.round(100*cor, 2)))


    list_PPL_without_offset_country = {gender : np.concatenate([list_PPL_without_offset[ct][gender] for ct in list_countries]) for gender in list_gender}
    proba_counter_without_offset_country = {v: {gender : np.concatenate([proba_counter_without_offset[v][ct][gender] for ct in list_countries]) for gender in list_gender} for v in proba_counter.keys()} 
    
    for proba_string in list_proba:
        for gender in list_gender:
            cor, _ = pearsonr(proba_counter_without_offset_country[proba_string][gender], 
                list_PPL_without_offset_country[gender])
            if args.verbose: print(proba_string, gender, cor)
            list_df.append((proba_string, 'Overall', gender, np.round(100*cor, 2)))

    df = pd.DataFrame(list_df[1:], columns=list_df[0])
    output_path = args.path_corpora + '%s/Table3_Correlations_PPL_%s'%(model_name, name_file)+ '.tsv'
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Local correlation data written to {output_path}")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="The name of the model name", type=str, default="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                 )
    parser.add_argument("--model_name_PT", help="The name of the pre-trained model name (if different)", type=str, default="cardiffnlp/twitter-xlm-roberta-base", 
                 )
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--data_tsv", type=str, default="Eurotweets_English_val_without_line_return.tsv_clean_test")
    parser.add_argument("--list_countries", help="countries to test, todo", type=str, default=['United_Kingdom', 'Ireland', 'United_States', 'Canada', 'Australia', 'New_Zealand', 
    'South_Africa', 'India', 'Germany', 'France', 'Spain', 'Italy', 'Portugal', 'Hungary', 'Poland', 'Turkey', 'Morocco'], nargs='+')
    parser.add_argument("--proba_only", help="Proba only", default=False, action='store_true')
    parser.add_argument("--verbose", help="verbose", default=False, action='store_true')
    parser.add_argument("--male_only", help="male only", default=True, type=bool)
    args = parser.parse_args()
    main_local_level(args)