"""
Soit je compare les changements par rapport aux predictions d'avant, en calculant:
- la diff moyenne de proba pos 
- la diff moyenne de proba neg 
- diff entre Ppos-Pneg en moyenne 
- la difference des F1-pro, F1-neg ? 
- le pourcentage de prediction differentes des predictions normales sur les positif/negatif (i.e. #FP et FN sur pos/neg quand on compare par rapport aux pred originales)

Valentin Barriere, 02/22
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# Path containing the CountryGenderNamePerturbation.py script, if in the same folder, you can comment this line
sys.path.insert(1, '/home/barriva/Valentin_code/Biases/')

from Transformers_data import create_input_array, loadTsvData
from CountryGenderNamePerturbation import PerturbedExamples
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle as pkl

from transformers import *
CACHE_DIR = '/home/barriva/data/.cache/torch/transformers'
proxies = None

import argparse
PATH_DATA = '/eos/jeodpp/data/projects/REFOCUS/data/Valentin/'

from scipy.special import kl_div


def _kl_div_scipy(P,Q):
    """
    2 options: mean of the KL on each example or mean of the probabilities and KL global 
    """
    return np.sum(kl_div(P,Q), axis=1)

def _kl_div(P,Q, mean_of_divs=True):
    """
    2 options: mean of the KL on each example or mean of the probabilities and KL global 
    """
    P=np.array(P)
    Q=np.array(Q)

    # mean of the divs 
    if mean_of_divs:
        kl = np.mean(np.sum((P * np.log(P / Q)), axis=1))
    else:
        # div of the means
        P=np.mean(P, axis=1)
        Q=np.mean(Q, axis=1)
        kl = np.sum((P * np.log(P / Q)))

    return kl

def symetric_kl(P,Q, mean_of_divs=True):
    return (_kl_div(P,Q,mean_of_divs=mean_of_divs) + _kl_div(Q,P,mean_of_divs=mean_of_divs))/2

# import tweetnlp


# Will never use this setting in the end, there is just one model that I'm interested in, and I will dl the datasets
# def prepare_data_and_model_tweetnlp(modelFilePath, task='sentiment'):
#     """
#     """

#     model = tweetnlp.load_model('hate')
#     # model.hate('Whoever just unfollowed me you a bitch', return_probability=True)

#     # change model.predict() so it outputs a vector of proba in line with a dict_lab
#     # Also see what is the output when a lot of inputs 

#     dataset, label2id = tweetnlp.load_dataset(task)
#     X_text, y = dataset, label2id
    
#     return model, tokenizer, X_text, y

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
# text = preprocess(text)

# Andrazp/multilingual-hate-speech-robacofi
# was jointly fine-tuned on five languages, namely Arabic, Croatian, English, German and Slovenian

def prepare_data_and_model_from_scratch(
    modelFilePath='cardiffnlp/twitter-xlm-roberta-base-sentiment',
    input_data_File=None, 
    path_corpus=None, 
    model_gen=TFAutoModelForSequenceClassification, 
    tok_gen=AutoTokenizer,
    task='sentiment',
    ):
    """
    """
 
    tokenizer = tok_gen.from_pretrained(modelFilePath, proxies=proxies, cache_dir = CACHE_DIR)
    model = model_gen.from_pretrained(modelFilePath, proxies=proxies, cache_dir = CACHE_DIR)

    config = AutoConfig.from_pretrained(modelFilePath)
    dict_lab = {v.lower():k for k,v in config.id2label.items()}

    # Finally I will always use path_corpus, it's easier 
    if path_corpus:
        X_text, y = loadTsvData(path_corpus + input_data_File, dict_lab, multi_labels = False, cumsum_label_vectors = False)
    # else:
        # dataset, label2id = tweetnlp.load_dataset(task)
        # X_text, y = dataset, label2id
    
    return model, tokenizer, dict_lab, X_text, y


def prepare_data_and_model(modelFilePath, model, input_data_File, dict_lab, path_corpus):
    """
    """
    model.load_weights(modelFilePath)
    X_text, y = loadTsvData(path_corpus + input_data_File, dict_lab, multi_labels = False, cumsum_label_vectors = False)
    
    return model, X_text, y
    

def _calculate_sentiment_bias(model, X_text, y, tokenizer, dict_lab, list_countries=[], n_duplicates=10, 
                              dict_pos_neg = {'positive' : 'yes', 'negative' : 'no'}):
    """
    The function itself, taking model, X_text, y as inputs
    """

    str_pos = 'positive'
    str_neg = 'negative'

    # if not positive negative, then you need a mapping between the variables and pos/neg
    if not (('positive' in dict_lab.keys()) and ('negative' in dict_lab.keys())):
        # check that all the labels are in the dict_pos_neg so that can be a direct mapping
        for k in dict_pos_neg.values():
            assert k in list(dict_lab.keys()), "dict_pos_neg does not contain the string names of positive/negative classes"
        
        print('new mapping:', dict_pos_neg)
        str_pos = dict_pos_neg[str_pos]
        str_neg = dict_pos_neg[str_neg]

    perturber = PerturbedExamples(list_countries) if len(list_countries) else PerturbedExamples()
    print("Perturbing examples...")
    perturbed_X_text = perturber.all_countries(X_text, y, n_duplicates)
    nb_sent_modified = len(perturbed_X_text['Original'][0])
    print('...Obtaining %d perturbated sentences' % nb_sent_modified)

    dict_results = {country : {} for country in perturbed_X_text.keys()}
    
    path_dump_perturbed = '/eos/jeodpp/data/projects/REFOCUS/data/Valentin/Biases/Perturbed_text_stances_en.pkl'
    with open(path_dump_perturbed, 'wb') as fp:
        pkl.dump(perturbed_X_text, fp)
    
    if nb_sent_modified > 1:
        print("Calculating probas...")
        # print(perturbed_X_text['Original'][0])

        input_arrays = create_input_array(perturbed_X_text['Original'][0], tokenizer)
        try: # for TFAutoModelForSequenceClassification
            proba_ini = softmax(model.predict(input_arrays), axis=1)
        except:
            proba_ini = softmax(model.predict(input_arrays)[0], axis=1)
        y_hat_ini = [kkk for kk in [n_duplicates*[k] for k in np.argmax(proba_ini, axis=1)] for kkk in kk]

        # mean of the difference between pos and neg
        dict_results['Original'] = 100*np.mean(proba_ini[:,dict_lab[str_pos]] - proba_ini[:,dict_lab[str_neg]])

        for country, value_gender in tqdm(perturbed_X_text.items()):
            if country != 'Original':
                for gender, (examples, labels) in value_gender.items():
                    input_arrays = create_input_array(examples, tokenizer)
                    # print(model.predict(input_arrays))
                    try:
                        proba = softmax(model.predict(input_arrays), axis=1)
                    except: # for TFAutoModelForSequenceClassification
                        proba = softmax(model.predict(input_arrays)[0], axis=1)
                    dict_results[country][gender] = {'proba' : np.round(100*np.mean(proba[:,dict_lab[str_pos]] - proba[:,dict_lab[str_neg]]) - dict_results['Original'],2)}
                    y_hat_pred = np.argmax(proba, axis=1)
                    CM = confusion_matrix(y_hat_ini, y_hat_pred)

                    path_dump_CM = '/eos/jeodpp/data/projects/REFOCUS/data/Valentin/Biases/CM_stances_en'
                    with open(path_dump_CM + '_%s.pkl'%country, 'wb') as fp:
                        pkl.dump(CM, fp)
                    # gives the percentage of increasing/decreasing 
                    # the number of new prediction - the number of past prediction / the number of past prediction
                    percentage_changing = (np.sum(CM, axis=0)-np.sum(CM, axis=1)) / np.sum(CM, axis=1)

                    # case where there was only one class outputed by the system, i.e. if very few entities were detected
                    if len(CM) == 1:
                        print(percentage_changing)
                        percentage_changing = [1 for _ in dict_lab.values()]

                    # print(y_hat_ini, y_hat_pred, CM)
                    for lab, idx_lab in dict_lab.items():
                        dict_results[country][gender][lab] = 100*np.round(percentage_changing[idx_lab],2)

                    dict_results[country][gender]['KL_sym'] = symetric_kl(np.repeat(proba_ini, n_duplicates, axis=0), proba)
                    dict_results[country][gender]['KL_sym_mean'] = symetric_kl(np.repeat(proba_ini, n_duplicates, axis=0), proba, mean_of_divs=False)
                    # dict_results[country][gender]['Sinkhorn'] = kl_div(proba_ini, proba)

        # creating the df
        del dict_results['Original']
        midx = pd.MultiIndex.from_product([list(dict_results.values())[0].keys(), list(list(dict_results.values())[0].values())[0].keys()])
        df_bias = pd.DataFrame([[ key3 for key2 in key1.values() for key3 in key2.values()] for key1 in dict_results.values()], index=dict_results.keys(),columns=midx)

    else:
        print('Not even one sentence detected with an entity... Return empty dataframe')
        df_bias = pd.DataFrame([[]])
        
    df_bias['n_sent_modified'] = nb_sent_modified
    return df_bias


def calculate_sentiment_bias(modelFilePath, model, input_data_File, tokenizer, dict_lab, path_corpus, list_countries=[], n_duplicates=10):
    """
    Return the sentiment bias 
    """
    
    model, X_text, y = prepare_data_and_model(modelFilePath, model, input_data_File, dict_lab, path_corpus)
    
    return _calculate_sentiment_bias(model, X_text, y, tokenizer, dict_lab, list_countries=list_countries, n_duplicates=n_duplicates)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="The name of the model name", type=str, default="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                 )
    parser.add_argument("-c", "--name_corpora", help="The name of the folder containing the corpora", type=str, 
                        default="Biases")
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--data_tsv", type=str, default="tweets_test_spanish_val.tsv")
    parser.add_argument("--list_countries", help="countries to test", type=str, default=['United_Kingdom', "France", 'Spain', 'Germany'], nargs='+')
    parser.add_argument("--n_duplicates", help="how many n_duplicates", type=int, default=10)
    parser.add_argument("--test", help="test", default=False, action='store_true')
    args = parser.parse_args()

    modelFilePath = args.model_name
    path_corpus = args.path_corpora + args.name_corpora + '/'
    input_data_File = args.data_tsv


    model, tokenizer, dict_lab, X_text, y = prepare_data_and_model_from_scratch(modelFilePath, input_data_File, path_corpus)    

    if args.test:
        X_text = X_text[:200]

    df_bias = _calculate_sentiment_bias(model, X_text, y, tokenizer, dict_lab, list_countries=args.list_countries, n_duplicates=args.n_duplicates)
    if not args.test:
        df_bias.to_csv(path_corpus + 'biases_' + input_data_File, sep='\t')
    else:
        print(df_bias)