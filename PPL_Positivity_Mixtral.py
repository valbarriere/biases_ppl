"""
Script to measure the PPL of a model. To use on a (M)LM, not on a classifier!

Use it like: 
python ../PPL_Positivity.py \
--data_tsv Eurotweets_English_val_without_line_return.tsv_clean_test \
--model_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
--list_gender male \
--verbose

France United_Kingdom Ireland Spain Germany Italy Morocco India 
Canada Australia New_Zealand United_States South_Africa Portugal Hungary Poland
Turkey Original'

Author: Valentin Barriere, 01/24
"""
import pickle as pkl
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm

import os
import argparse
# PATH_DATA = '/eos/jeodpp/data/projects/REFOCUS/data/Valentin/'
# PATH_DATA = '/home/barriva/data/'
PATH_DATA = './' # Move it to .env?
CACHE_DIR = '/workspace1/sebcif/hfcache/' # same here?
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="The name of the model name", type=str, default="cardiffnlp/twitter-xlm-roberta-base", 
                 )
    #parser.add_argument("-c", "--name_corpora", help="The name of the folder containing the corpora", type=str, 
                        # default="Biases")
    #parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                       # default=PATH_DATA)
    parser.add_argument("--data_tsv", type=str, default="tweets_test_spanish_val.tsv")
    parser.add_argument("--list_countries", help="countries to test", type=str, default=[], nargs='+')
    parser.add_argument("--list_gender", help="gender to test", type=str, default=[], nargs='+')
    parser.add_argument("--n_duplicates", help="how many n_duplicates", type=int, default=10)
    parser.add_argument("--verbose", help="verbose", default=False, action='store_true')
    parser.add_argument("--use_existing_dic", help="if already started experiments before", default=True, 
                        action='store_false'
                       )
    
    args = parser.parse_args()
    
    use_existing_dic = args.use_existing_dic
    path_data = PATH_DATA
    path_dump_perturbed = path_data + 'Perturbed_%s'%args.data_tsv
    model_folder = path_data + f"{args.model_name.replace('/', '_')}/"
    path_dump_perturbed_ppl = f"Perturbed_{args.data_tsv}"
    # model_name = "cardiffnlp/twitter-roberta-base"
    model_name = args.model_name
    
    with open(path_dump_perturbed, 'rb') as fp:
        perturbed_X_text = pkl.load(fp)
    model_inst = "Determine the sentiment of the tweet below by selecting one word: 'negative', 'neutral', or 'positive'. Keep your response succinct, avoiding explanations."
    inst_format = "[INST] {instruction} [/INST]\n" if model_inst else ""
    input_format = "Tweet:{tweet}\nSentiment:"
    formated_inst = inst_format.format(instruction=model_inst)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quantization_config,
                                                device_map=DEVICE_MAP,
                                                cache_dir=CACHE_DIR)

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

    def score2(model, tokenizer, sentence):
        """
        HuggingFace code:
        """
        encodings = tokenizer(sentence, return_tensors='pt')
        max_length = 4032
        stride = 4032
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE_MAP)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        nll_mean = torch.stack(nlls).mean() 
        # ppl = torch.exp(torch.stack(nlls).mean())
        # return ppl
        return nll_mean

    #country = 'United_Kingdom'
    #gender = 'male'
    proba = {}

    list_countries = args.list_countries if len(args.list_countries) else [k for k in perturbed_X_text.keys() if k != 'Original']
    country_verif = [c in perturbed_X_text.keys() for c in list_countries]
    assert all(country_verif)
    # list_countries = [k for k in perturbed_X_text.keys() if k != 'Original']
    if args.verbose: print('list_countries', list_countries)
    # list_countries = ['France','United_Kingdom','Ireland','Spain','Germany','Italy',
    #  'Morocco','Hungary','Poland','Estonia','Finland','Portugal','India','Russia','Turkey']

    # list_gender = ['male', 'female']
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
    
    # print(list_PPL_ini.keys(),list_PPL['Original'])
    # if not already in the existing dic
    # if len(list_PPL['Original']) == 0:
    #     for sent in tqdm(perturbed_X_text['Original'][0]):
    #         formated_input = input_format.format(tweet=sent)
    #         sent = formated_inst + formated_input
    #         _score = score2(model, tokenizer, sent).item()
    #         list_PPL['Original'].append(_score)
    
    for country in list_countries:
        for gender in list_gender:
            if args.verbose: print(country, gender)
            # if not already in the existing dic
            if len(list_PPL[country][gender]) == 0:
                for sent in tqdm(perturbed_X_text[country][gender][0]):
                    formated_input = input_format.format(tweet=sent)
                    sent = formated_inst + formated_input
                    _score = score2(model, tokenizer, sent).item()
                    list_PPL[country][gender].append(_score)
                # buff
                with open(path_dump_perturbed_ppl + '_PPL_buff_{list_countries[0]}.pkl', 'wb') as f: # Here it fails if folder doesnt exist
                    pkl.dump(list_PPL, f)
    with open(path_dump_perturbed_ppl + f'_PPL_{list_countries[0]}.pkl', 'wb') as f:
        pkl.dump(list_PPL, f)

    os.remove(path_dump_perturbed_ppl + '_PPL_buff_{list_countries[0]}.pkl')