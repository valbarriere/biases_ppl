"""
Script to measure the PPL of a model. To use on a (M)LM, not on a classifier!

Use it like: 
python ../PPL_Positivity.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--model_name cardiffnlp/twitter-xlm-roberta-base \
--list_gender male \
--verbose

Author: Valentin Barriere, 01/24
"""
import os
import torch
import argparse
import pickle as pkl
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForMaskedLM

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
        if torch.cuda.is_available():
            loss = model(masked_input.cuda(), labels=labels.cuda()).loss
        else:
            loss = model(masked_input, labels=labels).loss
    return loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="The name of the model", type=str, default="cardiffnlp/twitter-xlm-roberta-base")
    parser.add_argument("-c", "--name_corpora", help="The name of the folder containing the corpora", type=str, 
                        default="Biases")
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--data_tsv", type=str, help="The tsv file containing the data", default="tweets_test_spanish_val.tsv")
    parser.add_argument("--list_countries", help="Countries to test", type=str, default=[], nargs='+')
    parser.add_argument("--list_gender", help="Gender to test", type=str, default=[], nargs='+')
    parser.add_argument("--n_duplicates", help="How many n_duplicates", type=int, default=10)
    parser.add_argument("--verbose", help="verbose", default=False, action='store_true')
    parser.add_argument("--existing_dic", help="If already started experiments before", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_existing_dic", help="if already started experiments before", default=True)
                       
    
    args = parser.parse_args()
    use_existing_dic = args.existing_dic
    path_data = os.path.join(args.path_corpora, args.name_corpora)

    model_name_underscore = args.model_name.replace('/', '_')
    path_dump_perturbed_ppl = os.path.join(path_data, model_name_underscore)
    os.makedirs(path_dump_perturbed_ppl, exist_ok=True)
    perturbed_file_prefix = f"Perturbed_{args.data_tsv}"

    path_dump_perturbed_ppl = os.path.join(path_dump_perturbed_ppl, perturbed_file_prefix)
    path_dump_perturbed = os.path.join(path_data, f"Perturbed_{args.data_tsv}")
    with open(path_dump_perturbed, 'rb') as fp:
        perturbed_X_text = pkl.load(fp)

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if torch.cuda.is_available():
        model.cuda()

    proba = {}

    list_countries = args.list_countries if len(args.list_countries) else [k for k in perturbed_X_text.keys() if k != 'Original']
    if args.verbose: print('list_countries', list_countries)
    list_gender = args.list_gender if len(args.list_gender) else ['male', 'female']
    list_PPL = {ct : {'male':[],'female':[]} for ct in list_countries}
    list_PPL['Original'] = []
    
    # to start if program had stoped before
    if use_existing_dic:
        if os.path.isfile(path_dump_perturbed_ppl + '_PPL_buff.pkl'):
            with open(path_dump_perturbed_ppl + '_PPL_buff.pkl', 'rb') as f:
                list_PPL_ini = pkl.load(f)
            for key in list_PPL_ini.keys():
                list_PPL[key] = list_PPL_ini[key]
    
    # if not already in the existing dic
    if len(list_PPL['Original']) == 0:
        for sent in tqdm(perturbed_X_text['Original'][0]):
            list_PPL['Original'].append(score(model, tokenizer, sent).item())
        
    for country in list_countries:
        for gender in list_gender:
            if args.verbose: print(country, gender)
            # if not already in the existing dic
            if len(list_PPL[country][gender]) == 0:
                for sent in tqdm(perturbed_X_text[country][gender][0]):
                    list_PPL[country][gender].append(score(model, tokenizer, sent).item())
                # buff
                with open(path_dump_perturbed_ppl + '_PPL_buff.pkl', 'wb') as f:
                    pkl.dump(list_PPL, f)
        
    with open(path_dump_perturbed_ppl + '_PPL.pkl', 'wb') as f:
        pkl.dump(list_PPL, f)

    os.remove(path_dump_perturbed_ppl + '_PPL_buff.pkl')