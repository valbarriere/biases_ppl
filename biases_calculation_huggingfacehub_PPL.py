"""
Use it like:
python biases_calculation_huggingfacehub_PPL.py \
--name_corpora PPL_Positivity \
--data_tsv labeled_data.tsv \
--list_countries France United_Kingdom Ireland Spain Germany Italy Morocco \
India Canada Australia New_Zealand United_States South_Africa \
Portugal Hungary Poland Turkey \
--n_duplicates 50

Author: Anonymous_Submission 01/24
"""
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from utils import create_input_array, loadTsvData
from CountryGenderNamePerturbation import PerturbedExamples

load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR", None)
PATH_DATA = os.getenv("PATH_DATA", None)
proxies = None



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


def prepare_data_and_model_from_scratch(
    model_file_path='cardiffnlp/twitter-xlm-roberta-base-sentiment',
    input_data_file=None, 
    path_corpus=None, 
    model_gen=TFAutoModelForSequenceClassification, 
    tok_gen=AutoTokenizer,
    unlabeled=False
    ):
    """
    Load the model, tokenizer, label2id and the tweets with labels
    """
    tokenizer = tok_gen.from_pretrained(model_file_path, proxies=proxies, cache_dir = CACHE_DIR)
    model = model_gen.from_pretrained(model_file_path, proxies=proxies, cache_dir = CACHE_DIR)

    config = AutoConfig.from_pretrained(model_file_path)
    dict_lab = {k.lower():v for k,v in config.label2id.items()}
    X_text, y = loadTsvData(os.path.join(path_corpus, input_data_file), dict_lab, multi_labels = False, cumsum_label_vectors = False, unlabeled=unlabeled)
    
    return model, tokenizer, dict_lab, X_text, y


def sum_pos_neg_emotions(proba):
    """
    for the model cardiffnlp/roberta-base-tweet-emotion 
    "id2label": {
        "0": "joy",
        "1": "optimism",
        "2": "anger",
        "3": "sadness"
      }
    FAUX --> 
    0	anger
    1	joy
    2	optimism
    3	sadness
    """
    return np.concatenate((proba, proba[:,1:2] + proba[:,2:3], proba[:,0:1] + proba[:,3:4]), axis=1)

def create_dict_and_mapping_labels(dict_lab, dict_pos_neg, model_name):
    """
    WARNING: 
    To deal with errors contained in the config.labels2id of your model
    Ex: for roberta-base-tweet-sentiment no label, for roberta-base-tweet-emotion wrong columns
    You need to add the classes of your config.labels2id in dict_pos_neg, or to have a "positive" and a "negative" class
    """

    str_pos = 'positive'
    str_neg = 'negative'

    # used later, when emotion do not calculate the overall 'positive'/'negative' because the code does not work (future todo)
    dict_lab_ini = dict_lab.copy()
    # if emotion
    emotion_task = model_name == "cardiffnlp_roberta-base-tweet-emotion"
    sentiment_en_task = model_name == "cardiffnlp_roberta-base-tweet-sentiment"
    if emotion_task:
        # there is an error with the label set in the model on the API
        dict_lab = {'joy': 1, 'optimism': 2, 'anger': 0, 'sadness': 3}
        dict_lab_ini = dict_lab.copy()
        dict_lab['positive'] = 4
        dict_lab['negative'] = 5
    elif sentiment_en_task:
        dict_lab = {'negative': 0, 'neutral': 1, 'positive': 2}
        dict_lab_ini = dict_lab.copy()

    # if not positive negative, then you need a mapping between the variables and pos/neg
    if not (('positive' in dict_lab.keys()) and ('negative' in dict_lab.keys())):

        # check that all the labels are in the dict_pos_neg so that can be a direct mapping
        list_labels = dict_pos_neg[str_pos]
        assert sum([lab in list(dict_lab.keys()) for lab in list_labels]), f"dict_pos_neg does not contain the string names of {str_pos} class"
        for lab in list_labels:
            if lab in list(dict_lab.keys()):
                str_pos = lab

        list_labels = dict_pos_neg[str_neg]
        assert sum([lab in list(dict_lab.keys()) for lab in list_labels]), f"dict_pos_neg does not contain the string names of {str_neg} class"
        for lab in list_labels:
            if lab in list(dict_lab.keys()):
                str_neg = lab

        print('new mapping:', str_pos, str_neg)

    return dict_lab, dict_lab_ini, str_pos, str_neg


def _calculate_sentiment_bias(model, X_text, y, tokenizer, dict_lab, list_countries=[], n_duplicates=10, 
                              path_dump_perturbed=None,
                              dict_pos_neg = {'positive' : ['yes', 'non-hate', 'non-offensive', 'joy'], 'negative' : ['no', 'hate', 'offensive', 'sadness']}, 
                              perturb=True,
                              model_name='model', use_existing_dic=True, male_only=False, emotion_task=False):
    """
    The function itself, taking model, X_text, y as inputs
    """
    
    dict_lab, dict_lab_ini, str_pos, str_neg = create_dict_and_mapping_labels(dict_lab, dict_pos_neg, model_name)

    if perturb:
        perturber = PerturbedExamples(list_countries) if len(list_countries) else PerturbedExamples()
        print("Perturbing examples...")
        perturbed_X_text = perturber.all_countries(X_text, y, n_duplicates)

        # take the dictionnary if it has been calculated before
        if use_existing_dic:
            if os.path.isfile(path_dump_perturbed):
                with open(path_dump_perturbed, 'rb') as fp:
                    perturbed_X_text_existing = pkl.load(fp)
                for key in perturbed_X_text_existing:
                    perturbed_X_text[key] = perturbed_X_text_existing[key]
        # Save the sentences obtained for reuse later
        with open(path_dump_perturbed, 'wb') as fp:
            pkl.dump(perturbed_X_text, fp)
        
    else:
        print(f'Loading already existing {path_dump_perturbed}')
        with open(path_dump_perturbed, 'rb') as fp:
            perturbed_X_text = pkl.load(fp)
    
    nb_sent_modified = len(perturbed_X_text['Original'][0])
    print(f'...Obtaining {nb_sent_modified} perturbated sentences')
    
    dict_results = {country : {} for country in perturbed_X_text.keys() if country != 'Original'}

    if nb_sent_modified > 1:
        
        # Avoid to redo probaini if already done
        path_dump_proba_ini = ('/%s/ProbaIni_'%(model_name)).join(os.path.split(path_dump_perturbed)) + '.pkl'
        if not os.path.isfile(path_dump_proba_ini):
            print("Calculating probas...")

            input_arrays = create_input_array(perturbed_X_text['Original'][0], tokenizer)
            try: # for TFAutoModelForSequenceClassification
                proba_ini = softmax(model.predict(input_arrays), axis=1)
            except:
                proba_ini = softmax(model.predict(input_arrays)[0], axis=1)
            # create the folder
            os.makedirs(os.path.split(path_dump_perturbed)[0]+'/%s/'%model_name, exist_ok=True)

            # Dump the proba, with the country and the gender. So I can use them sentence by sentence to check on the link between Pos and PPL
            with open(path_dump_proba_ini, 'wb') as fp:
                pkl.dump(proba_ini, fp)
        else:
            print('Loading ProbaIni...')
            with open(path_dump_proba_ini, 'rb') as f:
                proba_ini = pkl.load(f)
        
        # remove the new columns added as 'main positive' and 'main negative'
        if emotion_task:
            proba_ini = sum_pos_neg_emotions(proba_ini)
            y_hat_ini = np.repeat(np.argmax(proba_ini[:,:4], axis=1), n_duplicates)
        else:
            y_hat_ini = np.repeat(np.argmax(proba_ini, axis=1), n_duplicates)
            
        # mean of the difference between pos and neg
        dict_results['OriginalDelta'] = 100*np.mean(proba_ini[:,dict_lab[str_pos]] - proba_ini[:,dict_lab[str_neg]])

        for country, value_gender in tqdm(perturbed_X_text.items()):
            if country != 'Original':
                for gender, (examples, labels) in value_gender.items():
                    
                    # if only doing it for male
                    if ((male_only and (gender == 'male')) or (not male_only)):
                        
                        path_dump_proba = ('/%s/Proba_'%(model_name)).join(os.path.split(path_dump_perturbed)) + '_%s_%s'%(country, gender) + '.pkl'
                        if os.path.isfile(path_dump_proba):
                            print(path_dump_proba, 'already existing... loading it!')
                            with open(path_dump_proba, 'rb') as fp:
                                proba = pkl.load(fp)
                        else:
                            print(path_dump_proba, 'NOT existing... making it!')
                            input_arrays = create_input_array(examples, tokenizer)
                            try:
                                proba = softmax(model.predict(input_arrays), axis=1)
                            except: # for TFAutoModelForSequenceClassification
                                proba = softmax(model.predict(input_arrays)[0], axis=1)

                            if emotion_task:
                                proba = sum_pos_neg_emotions(proba)
                            # Dump the proba, with the country and the gender. So I can use them sentence by sentence to check on the link between Pos and PPL
                            with open(path_dump_proba, 'wb') as fp:
                                pkl.dump(proba, fp)

                        dict_results[country][gender] = {'proba' : np.round(100*np.mean(proba[:,dict_lab[str_pos]] - proba[:,dict_lab[str_neg]]) - dict_results['OriginalDelta'],2)}
                        
                        # only do this on the basic emotions
                        if emotion_task:
                            y_hat_pred = np.argmax(proba[:,:4], axis=1)
                        else:
                            y_hat_pred = np.argmax(proba, axis=1)
                        CM = confusion_matrix(y_hat_ini, y_hat_pred)

                        # gives the percentage of increasing/decreasing 
                        # the number of new prediction - the number of past prediction / the number of past prediction
                        percentage_changing = (np.sum(CM, axis=0)-np.sum(CM, axis=1)) / np.sum(CM, axis=1)

                        # case where there was only one class outputed by the system, i.e. if very few entities were detected
                        if len(CM) == 1:
                            print(percentage_changing)
                            percentage_changing = [1 for _ in dict_lab.values()]
                        
                        for lab, idx_lab in dict_lab_ini.items():
                            dict_results[country][gender][lab] = 100*np.round(percentage_changing[idx_lab],3)

                        dict_results[country][gender]['KL_sym'] = symetric_kl(np.repeat(proba_ini, n_duplicates, axis=0), proba)
                        dict_results[country][gender]['KL_sym_mean'] = symetric_kl(np.repeat(proba_ini, n_duplicates, axis=0), proba, mean_of_divs=False)

        # creating the df
        del dict_results['OriginalDelta']
        midx = pd.MultiIndex.from_product([list(dict_results.values())[0].keys(), list(list(dict_results.values())[0].values())[0].keys()])
        df_bias = pd.DataFrame([[ key3 for key2 in key1.values() for key3 in key2.values()] for key1 in dict_results.values()], index=dict_results.keys(),columns=midx)

    else:
        print('Not even one sentence detected with an entity... Return empty dataframe')
        df_bias = pd.DataFrame([[]])
        
    df_bias['n_sent_modified'] = nb_sent_modified
    return df_bias


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="The name of the model", type=str, default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    parser.add_argument("-c", "--name_corpora", help="The name of the folder containing the corpora", type=str, 
                        default="Biases")
    parser.add_argument("--path_corpora", help="The path of the folders containing all the corpora", type=str, 
                        default=PATH_DATA)
    parser.add_argument("--data_tsv", type=str, help="The tsv file containing the data", default="tweets_test_spanish_val.tsv")
    parser.add_argument("--list_countries", help="Countries to test", type=str, default=['United_Kingdom', "France", 'Spain', 'Germany'], nargs='+')
    parser.add_argument("--n_duplicates", help="How many n_duplicates", type=int, default=10)
    parser.add_argument("--test", help="test", default=False, action='store_true')
    parser.add_argument("--proba_only", help="Calculate probability only", default=False, action='store_true')
    parser.add_argument("--male_only", help="Use male only", default=True, action='store_false')
    parser.add_argument("--perturb", help="Apply perturbation to data", default=True, action='store_false')
    parser.add_argument("--emotion_task", help="If the task is emotion classification", default=False, action='store_true')
    args = parser.parse_args()

    model_file_path = args.model_name
    path_corpus = os.path.join(args.path_corpora, args.name_corpora)
    input_data_file = args.data_tsv

    model, tokenizer, dict_lab, X_text, y = prepare_data_and_model_from_scratch(model_file_path, input_data_file, path_corpus)    

    model_name_underscore = model_file_path.replace('/', '_')
    if args.test:
        X_text = X_text[:200]
    df_bias = _calculate_sentiment_bias(model, X_text, y, tokenizer, dict_lab,
                                        list_countries=args.list_countries,
                                        n_duplicates=args.n_duplicates,
                                        path_dump_perturbed = os.path.join(path_corpus, f'Perturbed_{args.data_tsv}'),
                                        model_name=model_name_underscore,
                                        male_only=args.male_only,
                                        emotion_task=False)
    if args.test:
        print(df_bias)
    else:
        out_file_name = f'{model_name_underscore}biases_{input_data_file}.tsv'
        df_bias.to_csv(os.path.join(path_corpus, out_file_name), sep='\t')